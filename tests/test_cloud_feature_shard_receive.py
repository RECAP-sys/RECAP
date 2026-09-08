from __future__ import annotations

import os

from cloud.feature_cache import FeatureShardRef, FeatureShardStore
from edge.feature_shard import write_feature_label_shards
from tests.test_feature_shard_common import make_entries, runtime_context


def test_cloud_receive_registers_uploaded_npy_shard_without_pt(tmp_path) -> None:
    edge_root, shards, _labels = write_feature_label_shards(
        output_root=str(tmp_path / "edge"),
        storage_format="npy_memmap_shard",
        shard_max_samples=8,
        shard_dtype="float16",
        runtime_context=runtime_context(),
        generation="upload",
        entries=make_entries(2),
    )
    manifest = {
        "model_id": "yolo26n",
        "request_id": "upload",
        "runtime_contract": {"feature_layout_id": "layout-a", "contract_id": "contract-a"},
        "shards": shards,
    }
    store = FeatureShardStore(str(tmp_path / "cloud"), storage_format="npy_memmap_shard")
    registered = store.import_shard_bundle(
        bundle_root=edge_root,
        manifest=manifest,
        shard_entries=shards,
    )
    refs = [entry["feature_ref"] for entry in registered]
    assert len(refs) == 2
    assert all(isinstance(ref, FeatureShardRef) for ref in refs)
    assert store.read_batch(refs).batch_size == 2
    assert not any(
        filename.endswith(".pt")
        for _root, _dirs, files in os.walk(tmp_path / "cloud")
        for filename in files
    )
