from __future__ import annotations

import json
import os

import torch

from cloud.feature_cache import FeatureCacheMaterializer, FeatureShardStore
from tests.test_feature_shard_common import make_entries, runtime_context


def test_training_cache_view_contains_only_shard_refs_and_no_feature_files(
    tmp_path, monkeypatch
) -> None:
    store = FeatureShardStore(str(tmp_path / "shards"), storage_format="npy_memmap_shard")
    written = store.write_entries(
        make_entries(3),
        runtime_context=runtime_context(),
        generation="gen_000001",
        source="test",
    )

    def fail_load(*_args, **_kwargs):
        raise AssertionError("view materialization must not load feature payloads")

    monkeypatch.setattr(torch, "load", fail_load)
    result = FeatureCacheMaterializer(
        store,
        view_root_dir=str(tmp_path / "views"),
    ).write_training_view(
        view_id="view-a",
        generation="gen_000001",
        feature_layout_id="layout-a",
        contract_id="contract-a",
        feature_abi_id="feature-abi-a",
        runtime_identity_id="runtime-identity-a",
        entries=written,
    )
    assert result.view is not None
    assert {sample.sample_id for sample in result.view.samples} == {
        "sample-0",
        "sample-1",
        "sample-2",
    }
    view_dir = os.path.dirname(result.view.manifest_path)
    assert not os.path.exists(os.path.join(view_dir, "features"))
    manifest = json.loads(open(result.view.manifest_path, encoding="utf-8").read())
    assert manifest["feature_abi_id"] == "feature-abi-a"
    assert manifest["runtime_identity_id"] == "runtime-identity-a"
    assert all(
        sample["feature_ref"]["storage_format"] == "npy_memmap_shard"
        for sample in manifest["samples"]
    )
    assert result.stats.files_copied == 0
    assert result.stats.bytes_copied == 0
