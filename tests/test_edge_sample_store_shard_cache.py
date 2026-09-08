from __future__ import annotations

import os

import torch

from edge.sample_quality import HIGH_QUALITY
from edge.sample_store import EdgeSampleStore
from model_management.payload import boundary_payload_from_tensors


def test_edge_sample_store_caches_features_as_shard_refs(tmp_path) -> None:
    store = EdgeSampleStore(str(tmp_path / "edge_store"))
    payload = boundary_payload_from_tensors(
        {"boundary": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)},
        split_id="after:test",
        graph_signature="edge-store-test",
        batch_size=1,
    )

    record = store.store_sample(
        sample_id="sample-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="split-a",
        model_id="yolo26n",
        model_version="1",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
        input_image_size=[320, 320],
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
    )

    assert isinstance(record.feature_ref, dict)
    assert record.has_feature is True
    assert record.feature_bytes > 0
    assert not any(
        filename.endswith(".pt")
        for _root, _dirs, files in os.walk(store.root_dir)
        for filename in files
    )

    loaded = store.load_intermediate(record)
    assert torch.equal(loaded.tensors["boundary"], payload.tensors["boundary"])


def test_edge_sample_store_supports_safetensors_for_online_single_samples(tmp_path) -> None:
    store = EdgeSampleStore(
        str(tmp_path / "edge_store"),
        feature_storage_format="safetensors_shard",
    )
    payload = boundary_payload_from_tensors(
        {"boundary": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)},
        split_id="after:test",
        graph_signature="edge-store-test",
        batch_size=1,
    )

    record = store.store_sample(
        sample_id="sample-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="split-a",
        model_id="yolo26n",
        model_version="1",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
    )

    assert record.feature_ref is not None
    assert record.feature_ref["storage_format"] == "safetensors_shard"
    assert record.feature_ref["shard_path"].endswith(".safetensors")
    loaded = store.load_intermediate(record)
    assert torch.equal(loaded.tensors["boundary"], payload.tensors["boundary"])
