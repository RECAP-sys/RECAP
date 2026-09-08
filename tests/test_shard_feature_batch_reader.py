from __future__ import annotations

import json

import numpy as np
import torch

from cloud.feature_cache import (
    NPY_MEMMAP_SHARD,
    FeatureShardMetadata,
    FeatureShardPayloadCache,
    FeatureShardRef,
    FeatureShardStore,
    ShardFeatureBatchReader,
)
from tests.test_feature_shard_common import make_entries, runtime_context


def test_shard_feature_batch_reader_groups_rows_and_uses_cache(tmp_path, monkeypatch) -> None:
    store = FeatureShardStore(
        str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_dtype="float16",
        shard_max_samples=2,
        payload_cache_enabled=False,
    )
    refs = [
        entry["feature_ref"]
        for entry in store.write_entries(
            make_entries(4),
            runtime_context=runtime_context(),
            generation="gen",
            source="test",
        )
    ]
    cache = FeatureShardPayloadCache(enabled=True)
    reader = ShardFeatureBatchReader(payload_cache=cache)

    def fail_load(*_args, **_kwargs):
        raise AssertionError("torch.load must not be used by shard reader")

    monkeypatch.setattr(torch, "load", fail_load)
    first = reader.read_batch([refs[0], refs[1]])
    second = reader.read_batch([refs[0], refs[1]])
    assert first.batch_size == 2
    assert second.batch_size == 2
    assert cache.hits == 1
    assert cache.misses == 1


def test_shard_feature_batch_reader_restores_legacy_flat_folded_rows(tmp_path) -> None:
    shard_dir = tmp_path / "legacy"
    shard_dir.mkdir()
    shard_id = "legacy-folded"
    shard = torch.arange(16, dtype=torch.float16).reshape(8, 2).numpy()
    np.save(shard_dir / "leaf_0.npy", shard)
    leaf_specs = {
        "leaf_0": {
            "original_label": "folded",
            "shape": [4, 2],
            "sample_shape": [2],
            "dtype": "torch.float16",
            "schema": {
                "canonical_id": "folded",
                "torchlens_label": "folded",
                "module_path": "fake",
                "op_type": "reshape",
                "shape": ["B*4", 2],
                "dtype": "torch.float16",
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            },
        }
    }
    index_path = shard_dir / f"{shard_id}.index.json"
    meta_path = shard_dir / f"{shard_id}.meta.json"
    metadata = FeatureShardMetadata(
        storage_format=NPY_MEMMAP_SHARD,
        model_id="unit",
        model_family="unit",
        split_config_id="split",
        feature_layout_id="layout",
        contract_id="contract",
        boundary_id="after:folded",
        boundary_schema_hash="schema",
        passthrough_schema_hash=None,
        preprocessing_fingerprint=None,
        dtype="float16",
        shape_bucket="legacy",
        num_samples=2,
        leaf_specs=leaf_specs,
        sample_to_row={"sample-0": 0, "sample-1": 1},
        payload_kind="boundary_payload",
        shard_id=shard_id,
        shard_dir=str(shard_dir),
        index_path=str(index_path),
    )
    payload = metadata.to_dict()
    payload["metadata_path"] = str(meta_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    refs = [
        FeatureShardRef(
            storage_format=NPY_MEMMAP_SHARD,
            shard_id=shard_id,
            shard_path=None,
            shard_dir=str(shard_dir),
            index_path=str(index_path),
            row_id=index,
            sample_id=f"sample-{index}",
            feature_layout_id="layout",
            contract_id="contract",
            boundary_id="after:folded",
            payload_kind="boundary_payload",
            dtype="float16",
            shape_bucket="legacy",
            leaf_keys=["leaf_0"],
        )
        for index in range(2)
    ]

    batch = ShardFeatureBatchReader().read_batch([refs[1], refs[0]])

    assert batch.batch_size == 2
    assert tuple(batch.tensors["folded"].shape) == (8, 2)
    assert torch.equal(
        batch.tensors["folded"][:, 0],
        torch.tensor([8, 10, 12, 14, 0, 2, 4, 6], dtype=torch.float16),
    )
