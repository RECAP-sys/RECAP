from __future__ import annotations

import json
import os
import zipfile

import pytest
import torch

from edge.feature_shard import write_feature_label_shards
from edge.sample_quality import HIGH_QUALITY
from edge.sample_store import EdgeSampleStore
from edge.sample_sync import pack_high_quality_sync_bundle_to_file
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import SplitRuntimeContract
from tests.test_feature_shard_common import make_entries, runtime_context


def test_edge_feature_shard_writer_outputs_manifest_labels_and_no_pt(tmp_path) -> None:
    root, shards, labels_by_shard = write_feature_label_shards(
        output_root=str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_max_samples=2,
        shard_dtype="float16",
        runtime_context=runtime_context(),
        generation="upload",
        entries=make_entries(3),
    )
    assert len(shards) == 2
    assert sum(shard["sample_count"] for shard in shards) == 3
    assert labels_by_shard
    for shard in shards:
        assert os.path.exists(os.path.join(root, shard["label_file"]))
        assert "shard_dir" in shard
        assert shard["sample_ids"]
    assert not any(
        filename.endswith(".pt") for _root, _dirs, files in os.walk(root) for filename in files
    )


def test_edge_feature_shard_writer_preserves_dtype_when_unset(tmp_path) -> None:
    root, shards, _labels_by_shard = write_feature_label_shards(
        output_root=str(tmp_path),
        storage_format="npy_memmap_shard",
        shard_max_samples=2,
        shard_dtype=None,
        runtime_context=runtime_context(),
        generation="upload",
        entries=make_entries(2, dtype=torch.float32),
    )

    shard = shards[0]
    index_path = os.path.join(root, shard["shard_dir"], shard["index_file_name"])
    with open(index_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["dtype"] == "float32"
    assert {leaf["dtype"] for leaf in metadata["leaf_specs"].values()} == {"torch.float32"}


def test_high_quality_feature_label_shard_writes_runtime_contract_feature_abi(tmp_path) -> None:
    tensors = {
        "boundary": torch.zeros((1, 2, 3), dtype=torch.float16),
        "skip": torch.zeros((1, 1, 2), dtype=torch.float16),
    }
    contract = SplitRuntimeContract.create(
        edge_id=1,
        model_id="yolo26n",
        split_config_id="split-a",
        canonical_split_key="after:test",
        edge_split_id="after:test",
        cloud_batch_split_id="after:test",
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=list(tensors),
        front_version="1",
        feature_tensors=tensors,
        runtime_identity={"graph_signature": "edge-abi-writer"},
    )
    payload = boundary_payload_from_tensors(
        tensors,
        split_id="after:test",
        graph_signature="edge-abi-writer",
        batch_size=1,
    )
    store = EdgeSampleStore(str(tmp_path / "edge_store"))
    record = store.store_sample(
        sample_id="sample-abi",
        frame_index=1,
        confidence=0.9,
        split_config_id="split-a",
        model_id="yolo26n",
        model_version="1",
        front_version="1",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
        input_image_size=[320, 320],
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
        runtime_contract=contract.to_dict(),
    )

    zip_path, manifest, _stats = pack_high_quality_sync_bundle_to_file(
        store,
        [record],
        edge_id=1,
        shard_size=8,
        storage_format="npy_memmap_shard",
        request_id="upload-abi",
        split_context={
            "model_id": contract.model_id,
            "model_version": "1",
            "edge_session_id": "session-a",
            "front_version": contract.front_version,
            "split_config_id": contract.split_config_id,
            "canonical_split_key": contract.canonical_split_key,
            "edge_split_id": contract.edge_split_id,
            "input_tensor_shape": list(contract.input_tensor_shape),
            "input_resize_mode": contract.input_resize_mode,
            "runtime_contract": contract.to_dict(),
        },
        output_dir=str(tmp_path / "sync"),
    )
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            bundle_manifest = json.loads(archive.read("bundle_manifest.json"))
            shard = bundle_manifest["shards"][0]
            index_payload = json.loads(
                archive.read(f"{shard['shard_dir']}/{shard['index_file_name']}")
            )
            meta_payload = json.loads(
                archive.read(f"{shard['shard_dir']}/{shard['meta_file_name']}")
            )
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    assert manifest["feature_abi_id"] == contract.feature_abi_id
    assert manifest["edge_session_id"] == "session-a"
    assert bundle_manifest["edge_session_id"] == "session-a"
    assert bundle_manifest["runtime_contract"]["feature_abi_id"] == contract.feature_abi_id
    assert index_payload["feature_abi_id"] == contract.feature_abi_id
    assert meta_payload["feature_abi_id"] == contract.feature_abi_id
    assert meta_payload["metadata"]["feature_abi_spec"] == contract.feature_abi_spec


def test_split_runtime_contract_rejects_missing_current_identity() -> None:
    tensors = {"boundary": torch.zeros((1, 2, 3), dtype=torch.float16)}
    contract = SplitRuntimeContract.create(
        edge_id=1,
        model_id="yolo26n",
        split_config_id="split-a",
        canonical_split_key="after:test",
        edge_split_id="after:test",
        cloud_batch_split_id="after:test",
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=list(tensors),
        front_version="1",
        feature_tensors=tensors,
        runtime_identity={"graph_signature": "current-contract"},
    )
    payload = contract.to_dict()
    payload.pop("feature_abi_id")

    with pytest.raises(ValueError, match="feature_abi_id"):
        SplitRuntimeContract.from_dict(payload)


def test_high_quality_safetensors_bundle_creates_nested_shard_paths(tmp_path) -> None:
    pytest.importorskip("safetensors")
    tensors = {
        "boundary": torch.zeros((1, 2, 3), dtype=torch.float16),
        "skip": torch.zeros((1, 1, 2), dtype=torch.float16),
    }
    contract = SplitRuntimeContract.create(
        edge_id=1,
        model_id="yolo26n",
        split_config_id="split-a",
        canonical_split_key="after:test",
        edge_split_id="after:test",
        cloud_batch_split_id="after:test",
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=list(tensors),
        front_version="1",
        feature_tensors=tensors,
        runtime_identity={"graph_signature": "edge-safetensors-writer"},
    )
    payload = boundary_payload_from_tensors(
        tensors,
        split_id="after:test",
        graph_signature="edge-safetensors-writer",
        batch_size=1,
    )
    store = EdgeSampleStore(str(tmp_path / "edge_store"))
    record = store.store_sample(
        sample_id="sample-safetensors",
        frame_index=1,
        confidence=0.9,
        split_config_id="split-a",
        model_id="yolo26n",
        model_version="1",
        front_version="1",
        quality_bucket=HIGH_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
        input_image_size=[320, 320],
        input_tensor_shape=[1, 3, 320, 320],
        input_resize_mode="direct_resize",
        runtime_contract=contract.to_dict(),
    )

    zip_path, manifest, _stats = pack_high_quality_sync_bundle_to_file(
        store,
        [record],
        edge_id=1,
        shard_size=8,
        storage_format="safetensors_shard",
        request_id="upload-safetensors",
        split_context={
            "model_id": contract.model_id,
            "model_version": "1",
            "front_version": contract.front_version,
            "split_config_id": contract.split_config_id,
            "canonical_split_key": contract.canonical_split_key,
            "edge_split_id": contract.edge_split_id,
            "input_tensor_shape": list(contract.input_tensor_shape),
            "input_resize_mode": contract.input_resize_mode,
            "runtime_contract": contract.to_dict(),
        },
        output_dir=str(tmp_path / "sync"),
    )
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = set(archive.namelist())
            shard = manifest["shards"][0]
            assert shard["shard_file"] in names
            assert shard["index_file"] in names
            assert shard["meta_file"] in names
            assert shard["label_file"] in names
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
