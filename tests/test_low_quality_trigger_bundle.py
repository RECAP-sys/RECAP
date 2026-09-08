from __future__ import annotations

import io
import os
import zipfile
from types import SimpleNamespace

import numpy as np
import torch

from cloud.feature_cache import FeatureShardStore
from cloud.feature_cache.path_utils import fs_path
from cloud.feature_cache.types import SAFETENSORS_SHARD
from cloud.ingest import materialize_low_quality_trigger_bundle
from edge.sample_quality import LOW_QUALITY
from edge.sample_store import EdgeSampleStore
from edge.transmit import (
    _select_low_quality_trigger_records,
    measure_trigger_bundle_payload,
    pack_low_quality_trigger_bundle_to_file,
)
from model_management.fixed_split import SplitPlan
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import build_runtime_contract


def test_trigger_bundle_payload_measurement_accounts_for_all_bytes() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("raw_shards/raw.tar", b"raw" * 100)
        archive.writestr("feature_shards/features.bin", b"feature" * 100)
        archive.writestr("trigger_manifest.json", b'{}')

    metrics = measure_trigger_bundle_payload(buffer.getvalue())

    assert metrics["raw_frame_bytes"] > 0
    assert metrics["feature_bytes"] > 0
    assert metrics["prediction_metadata_bytes"] > 0
    assert (
        metrics["raw_frame_bytes"] + metrics["feature_bytes"] + metrics["prediction_metadata_bytes"]
        == metrics["total_upload_bytes"]
    )


def _runtime_contract() -> dict[str, object]:
    return build_runtime_contract(
        logical_split_id="after:test",
        trace_signature="trace-a",
        trace_device_type="cpu",
        runtime_backend="torchlens_native",
        boundary_tensor_labels=["boundary"],
        boundary_schema={
            "boundary": {
                "canonical_id": "boundary",
                "torchlens_label": "boundary",
                "module_path": "fake",
                "op_type": "conv",
                "shape": (1, 2, 3),
                "dtype": "torch.float32",
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            }
        },
        model_id="yolo26n",
        model_version="1",
        input_tensor_shape=[1, 3, 32, 32],
        input_resize_mode="direct_resize",
        feature_layout={"boundary": {"shape": [1, 2, 3], "dtype": "float32"}},
    )


def _split_plan(runtime_contract: dict[str, object]) -> SplitPlan:
    return SplitPlan(
        split_config_id="split-a",
        model_name="yolo26n",
        candidate_id="after:test",
        split_index=1,
        split_label="after:test",
        boundary_tensor_labels=["boundary"],
        payload_bytes=24,
        privacy_metric=0.0,
        privacy_risk=0.0,
        layer_freezing_ratio=0.0,
        canonical_split_key="after:test",
        edge_split_id="after:test",
        input_tensor_shape=[1, 3, 32, 32],
        input_resize_mode="direct_resize",
        front_version="0",
        runtime_contract=runtime_contract,
    )


def test_raw_feature_selection_counts_shared_artifacts_once(tmp_path) -> None:
    runtime_contract = _runtime_contract()
    split_plan = _split_plan(runtime_contract)
    root = tmp_path / "edge_store"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "one.jpg").write_bytes(b"a")
    (raw_dir / "two.jpg").write_bytes(b"b")

    feature_dir = root / "features"
    feature_dir.mkdir()
    shard_path = feature_dir / "shared.safetensors"
    index_path = feature_dir / "shared.index.json"
    meta_path = feature_dir / "shared.meta.json"
    shard_path.write_bytes(b"x" * 20)
    index_path.write_bytes(b"{}")
    meta_path.write_bytes(b"{}")

    feature_ref = {
        "storage_format": SAFETENSORS_SHARD,
        "shard_id": "shared",
        "shard_path": str(shard_path),
        "index_path": str(index_path),
        "feature_layout_id": runtime_contract["feature_layout_id"],
    }
    records = [
        SimpleNamespace(
            sample_id="low-1",
            quality_bucket=LOW_QUALITY,
            timestamp="2026-01-01T00:00:00Z",
            raw_relpath="raw/one.jpg",
            feature_ref=feature_ref,
            in_drift_window=False,
        ),
        SimpleNamespace(
            sample_id="low-2",
            quality_bucket=LOW_QUALITY,
            timestamp="2026-01-01T00:00:01Z",
            raw_relpath="raw/two.jpg",
            feature_ref=feature_ref,
            in_drift_window=False,
        ),
    ]

    selected, stats = _select_low_quality_trigger_records(
        SimpleNamespace(root_dir=str(root)),
        records,
        send_low_conf_features=True,
        split_plan=split_plan,
        bundle_cap_bytes=26,
    )

    assert [record.sample_id for record in selected] == ["low-1", "low-2"]
    assert stats["source_total_bytes"] == 26


def test_trigger_sample_cap_keeps_most_recent_drift_samples(tmp_path) -> None:
    root = tmp_path / "edge_store"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    records = []
    for index in range(1, 4):
        (raw_dir / f"{index}.jpg").write_bytes(bytes([index]))
        records.append(
            SimpleNamespace(
                sample_id=f"low-{index}",
                quality_bucket=LOW_QUALITY,
                timestamp=f"2026-01-01T00:00:0{index}Z",
                raw_relpath=f"raw/{index}.jpg",
                feature_ref=None,
                in_drift_window=True,
            )
        )

    selected, stats = _select_low_quality_trigger_records(
        SimpleNamespace(root_dir=str(root)),
        records,
        send_low_conf_features=False,
        bundle_cap_bytes=None,
        max_samples=2,
    )

    assert [record.sample_id for record in selected] == ["low-2", "low-3"]
    assert stats["max_samples"] == 2
    assert stats["omitted_sample_count"] == 1


def test_raw_feature_low_quality_bundle_imports_current_feature_ref(tmp_path) -> None:
    runtime_contract = _runtime_contract()
    split_plan = _split_plan(runtime_contract)
    store = EdgeSampleStore(str(tmp_path / "edge_store"))
    payload = boundary_payload_from_tensors(
        {"boundary": torch.ones((1, 2, 3), dtype=torch.float32)},
        split_id="after:test",
        graph_signature="raw-feature-test",
        batch_size=1,
    )
    store.store_sample(
        sample_id="low-1",
        frame_index=1,
        confidence=0.1,
        split_config_id="split-a",
        model_id="yolo26n",
        model_version="1",
        quality_bucket=LOW_QUALITY,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
        raw_frame=np.zeros((32, 32, 3), dtype=np.uint8),
        input_image_size=[32, 32],
        input_tensor_shape=[1, 3, 32, 32],
        input_resize_mode="direct_resize",
        runtime_contract=runtime_contract,
    )

    zip_path, manifest, stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=7,
        send_low_conf_features=True,
        split_plan=split_plan,
        model_id="yolo26n",
        model_version="1",
        output_dir=str(tmp_path),
    )

    assert manifest["upload_mode"] == "raw+feature"
    assert manifest["sample_ids"] == ["low-1"]
    assert manifest["raw_shards"][0]["sample_ids"] == ["low-1"]
    assert stats["feature_shard_count"] == 1
    assert manifest["feature_shards"]

    workspace = tmp_path / "cloud_workspace"
    workspace.mkdir()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(workspace)

    cloud_store = FeatureShardStore(str(tmp_path / "cloud_features"))
    materialized = materialize_low_quality_trigger_bundle(
        str(workspace),
        feature_store=cloud_store,
    )

    assert materialized is not None
    sample = materialized["samples"][0]
    assert sample["raw_relpath"].startswith("low_quality_staging/raw/")
    assert sample["feature_layout_id"] == runtime_contract["feature_layout_id"]
    feature_ref = sample["feature_ref"]
    assert os.path.exists(fs_path(feature_ref["index_path"]))
    assert str(feature_ref["index_path"]).startswith(str(tmp_path / "cloud_features"))


def test_uploaded_low_quality_samples_can_be_deleted_from_edge_store(tmp_path) -> None:
    runtime_contract = _runtime_contract()
    split_plan = _split_plan(runtime_contract)
    store = EdgeSampleStore(str(tmp_path / "edge_store"))
    payload = boundary_payload_from_tensors(
        {"boundary": torch.ones((1, 2, 3), dtype=torch.float32)},
        split_id="after:test",
        graph_signature="raw-feature-test",
        batch_size=1,
    )
    for index in range(1, 4):
        store.store_sample(
            sample_id=f"low-{index}",
            frame_index=index,
            confidence=0.1,
            split_config_id="split-a",
            model_id="yolo26n",
            model_version="1",
            quality_bucket=LOW_QUALITY,
            inference_result={"boxes": [], "labels": [], "scores": []},
            intermediate=payload,
            raw_frame=np.zeros((32, 32, 3), dtype=np.uint8),
            input_image_size=[32, 32],
            input_tensor_shape=[1, 3, 32, 32],
            input_resize_mode="direct_resize",
            runtime_contract=runtime_contract,
        )

    assert store.delete_samples(["low-1", "low-2"], quality_bucket=LOW_QUALITY) == 2
    assert store.stats()["low_quality_count"] == 1

    zip_path, manifest, stats = pack_low_quality_trigger_bundle_to_file(
        store,
        edge_id=7,
        send_low_conf_features=False,
        split_plan=split_plan,
        model_id="yolo26n",
        model_version="1",
        output_dir=str(tmp_path),
    )

    assert os.path.exists(zip_path)
    assert stats["sample_count"] == 1
    assert manifest["sample_ids"] == ["low-3"]
    assert [record.sample_id for record in store.list_records(quality_bucket=LOW_QUALITY)] == [
        "low-3"
    ]
