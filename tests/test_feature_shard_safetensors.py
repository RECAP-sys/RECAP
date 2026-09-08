from __future__ import annotations

import json
import os

import pytest
import torch

from cloud.feature_cache import FeatureShardRef, FeatureShardStore
from model_management.payload import boundary_payload_from_tensors
from tests.test_feature_shard_common import make_entries, make_folded_entries, runtime_context

pytest.importorskip("safetensors")


def test_safetensors_shard_writes_stacked_leaves_and_reads_batch(tmp_path) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="safetensors_shard",
        shard_dtype="float16",
        shard_max_samples=8,
    )
    written = store.write_entries(
        make_entries(3),
        runtime_context=runtime_context(),
        generation="gen_000001",
        source="test",
    )
    refs = [entry["feature_ref"] for entry in written]
    assert all(isinstance(ref, FeatureShardRef) for ref in refs)
    assert len({refs[0].shard_path}) == 1
    assert os.path.exists(refs[0].shard_path or "")
    assert os.path.exists(refs[0].index_path)
    assert refs[0].leaf_keys == ["leaf_0", "leaf_1"]
    assert [ref.row_id for ref in refs] == [0, 1, 2]
    with open(refs[0].index_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert "format_version" not in metadata
    assert metadata["leaf_specs"]["leaf_0"]["storage_layout"] == "sample_axis"
    assert metadata["leaf_specs"]["leaf_0"]["feature_shape_without_batch"] == [2, 3]
    try:
        from safetensors import safe_open
    except ModuleNotFoundError:  # pragma: no cover - guarded by importorskip
        pytest.skip("safetensors unavailable")
    with safe_open(refs[0].shard_path or "", framework="pt", device="cpu") as handle:
        assert tuple(handle.get_tensor("leaf_0").shape) == (3, 1, 2, 3)

    batch = store.read_batch([refs[2], refs[0]])
    assert list(batch.tensors) == ["boundary", "skip"]
    assert tuple(batch.tensors["boundary"].shape) == (2, 2, 3)
    assert torch.equal(
        batch.tensors["boundary"][:, 0, 0], torch.tensor([2.0, 0.0], dtype=torch.float16)
    )


def test_safetensors_folded_batch_reads_with_symbolic_multiplier(tmp_path) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="safetensors_shard",
        shard_dtype="float16",
        shard_max_samples=8,
    )
    written = store.write_entries(
        make_folded_entries(2),
        runtime_context=runtime_context("folded-layout"),
        generation="gen_folded",
        source="test",
    )
    refs = [entry["feature_ref"] for entry in written]

    batch = store.read_batch([refs[1], refs[0]])

    assert batch.batch_size == 2
    assert tuple(batch.tensors["folded"].shape) == (8, 2)
    assert torch.equal(
        batch.tensors["folded"][:, 0],
        torch.tensor([100, 102, 104, 106, 0, 2, 4, 6], dtype=torch.float16),
    )
    with open(refs[0].index_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["leaf_specs"]["leaf_0"]["sample_shape"] == [4, 2]
    assert metadata["leaf_specs"]["leaf_0"]["feature_shape_without_batch"] == [2]


def test_safetensors_shape_bucket_split(tmp_path) -> None:
    entries = make_entries(1)
    entries.append(
        {
            "sample": {"sample_id": "different", "labels": {"boxes": [], "labels": []}},
            "record": {
                "intermediate": boundary_payload_from_tensors(
                    {"boundary": torch.zeros(1, 4, 3), "skip": torch.zeros(1, 1, 2)},
                    split_id="after:test",
                    graph_signature="test-graph",
                    batch_size=1,
                )
            },
        }
    )
    store = FeatureShardStore(
        str(tmp_path), storage_format="safetensors_shard", shard_max_samples=8
    )
    written = store.write_entries(
        entries, runtime_context=runtime_context(), generation="gen", source="test"
    )
    assert len({entry["feature_ref"].shard_id for entry in written}) == 2
