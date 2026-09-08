from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from cloud.feature_cache import FeatureShardStore
from cloud_server import CloudContinualLearner
from model_management.payload import boundary_payload_from_tensors
from model_management.split_contract import SplitRuntimeContract
from model_management.split_runtime.torchlens_forward_guard import torchlens_forward_guard


class FakeLargeOD:
    def __init__(self) -> None:
        self.model_name = "fake_teacher"
        self.model = SimpleNamespace(label_schema="coco_91", class_names=["person"])
        self.batch_calls = 0

    def large_inference_batch(self, images, threshold=None):
        del threshold
        self.batch_calls += 1
        return [([[1, 2, 3, 4]], [1], [0.9]) for _image in images]

    def large_inference(self, image, threshold=None):
        del image, threshold
        return ([[1, 2, 3, 4]], [1], [0.9])


def _config(tmp_path, *, async_enabled: bool):
    teacher_annotation = SimpleNamespace(
        async_enabled=async_enabled,
        cache_enabled=True,
        wait_timeout_sec=0.0,
        worker_batch_size=4,
        worker_max_queue_size=64,
        worker_max_retries=0,
        oom_retry_enabled=True,
        min_worker_batch_size=1,
        cache_root_dir=str(tmp_path / "teacher_label_cache"),
    )
    continual_learning = SimpleNamespace(
        num_epoch=1,
        max_concurrent_jobs=1,
        batch_size=2,
        trace_batch_size=1,
        feature_cache_mode="auto",
        teacher_annotation_threshold=0.5,
        teacher_batch_size=2,
        teacher_annotation=teacher_annotation,
        proxy_eval_interval_rounds=1,
        proxy_eval_patience=0,
        proxy_eval_min_delta=0.0,
        proxy_eval_max_samples=0,
        proxy_eval_frame_cache_enabled=True,
        split_learning_rate=1e-3,
        wrapper_fixed_split_learning_rate=3e-5,
        tinynext_fixed_split_learning_rate=1e-3,
        rfdetr_fixed_split_learning_rate=1e-4,
        tinynext_fixed_split_target_steps_per_round=4,
        yolo_fixed_split_target_steps_per_round=4,
        rfdetr_fixed_split_target_steps_per_round=4,
        recent_training_window_root=str(tmp_path / "recent_training_windows"),
        split_contract_root=str(tmp_path / "contracts"),
    )
    return SimpleNamespace(
        edge_model_name="yolo26n",
        golden="fake_teacher",
        weights_path="",
        workspace_root=str(tmp_path / "workspace"),
        continual_learning=continual_learning,
        training_frame_count=2,
        tinynext_input_size=320,
    )


def _write_image(frame_dir, sample_id: str) -> str:
    frame_dir.mkdir(parents=True, exist_ok=True)
    path = frame_dir / f"{sample_id}.jpg"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)
    return str(path)


def test_injected_shared_teacher_service_skips_local_teacher_model(tmp_path) -> None:
    shared_service = SimpleNamespace()
    learner = CloudContinualLearner(
        _config(tmp_path, async_enabled=True),
        None,
        teacher_annotation_service=shared_service,
        teacher_annotation_metadata={
            "teacher_model_name": "rtdetr_x",
            "teacher_weights_fingerprint": "shared-fingerprint",
            "teacher_label_schema": "coco_91",
            "teacher_num_classes": 91,
            "teacher_class_names": ["person"],
        },
    )
    try:
        assert learner.teacher_annotation_worker is None
        assert learner.teacher_annotation_service is shared_service
        assert learner._teacher_model_name() == "rtdetr_x"
        assert learner._teacher_weights_fingerprint() == "shared-fingerprint"
        assert learner._teacher_num_classes() == 91
    finally:
        learner.close()


def _boundary_and_contract():
    boundary = boundary_payload_from_tensors(
        {"feat": torch.randn(1, 4)},
        split_id="after:fake",
        graph_signature="teacher-integration",
        batch_size=1,
        schema={
            "feat": {
                "canonical_id": "feat",
                "torchlens_label": "feat",
                "module_path": "fake.feat",
                "op_type": "linear",
                "shape": (1, 4),
                "dtype": torch.float32,
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            }
        },
    )
    contract = SplitRuntimeContract.create(
        edge_id=1,
        model_id="yolo26n",
        split_config_id="split-a",
        canonical_split_key="after:fake",
        edge_split_id="after:fake",
        cloud_batch_split_id="after:fake",
        input_tensor_shape=[1, 3, 16, 16],
        input_resize_mode="direct_resize",
        boundary_tensor_labels=["feat"],
        front_version="0",
        feature_tensors=dict(boundary.tensors),
        runtime_identity={"graph_signature": "teacher-integration"},
    )
    return boundary, contract


def _candidate_with_shard_ref(
    tmp_path,
    *,
    sample_id: str,
    boundary,
    contract: SplitRuntimeContract,
    labels: dict,
) -> dict:
    store = FeatureShardStore(str(tmp_path / "shards"), storage_format="npy_memmap_shard")
    written = store.write_entries(
        [
            {
                "sample": {"sample_id": sample_id},
                "record": {"intermediate": boundary},
            }
        ],
        runtime_context={
            "model_id": contract.model_id,
            "model_family": "test",
            "split_config_id": contract.split_config_id,
            "contract_id": contract.contract_id,
            "feature_layout_id": contract.feature_layout_id,
            "boundary_id": contract.cloud_batch_split_id,
        },
        generation="teacher-integration",
        source="test_low_quality",
    )
    return {
        "sample_id": sample_id,
        "feature_ref": written[0]["feature_ref"].to_dict(),
        "feature_layout_id": contract.feature_layout_id,
        "labels": dict(labels),
        "split_config_id": contract.split_config_id,
        "front_version": contract.front_version,
        "input_image_size": [16, 16],
        "input_tensor_shape": [1, 3, 16, 16],
        "input_resize_mode": "direct_resize",
    }


def test_cache_hit_training_path_does_not_call_teacher_model(tmp_path) -> None:
    fake_teacher = FakeLargeOD()
    learner = CloudContinualLearner(
        _config(tmp_path, async_enabled=True),
        fake_teacher,
    )
    try:
        frame_dir = tmp_path / "frames"
        _write_image(frame_dir, "sample-1")
        requests = learner._build_teacher_annotation_requests_from_frame_dir(
            str(frame_dir),
            ["sample-1"],
            edge_id=1,
            model_id="yolo26n",
            include_empty=True,
            target_model_metadata={},
        )
        learner.teacher_label_cache.write(
            requests[0],
            {"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
            source="test",
        )
        fake_teacher.batch_calls = 0

        annotations = learner._collect_teacher_annotations(
            str(frame_dir),
            ["sample-1"],
            include_empty=True,
            target_model_metadata={},
            edge_id=1,
            model_id="yolo26n",
        )

        assert fake_teacher.batch_calls == 0
        assert annotations["sample-1"]["labels"] == [1]
    finally:
        learner.close()


def test_low_quality_materialized_samples_submit_early(tmp_path) -> None:
    fake_teacher = FakeLargeOD()
    learner = CloudContinualLearner(
        _config(tmp_path, async_enabled=True),
        fake_teacher,
    )
    submitted = {}

    class FakeService:
        def submit_many(self, requests):
            submitted["sample_ids"] = [request.sample_id for request in requests]
            return SimpleNamespace(
                requested_samples=len(requests),
                cache_hits=0,
                cache_misses=len(requests),
                submitted=len(requests),
                duplicate=0,
                failed_count=0,
            )

    try:
        raw_dir = tmp_path / "bundle" / "low_quality_staging" / "raw"
        raw_path = _write_image(raw_dir, "sample-1")
        manifest = {
            "samples": [
                {
                    "sample_id": "sample-1",
                    "quality_bucket": "low_quality",
                    "raw_relpath": str(raw_path).replace(str(tmp_path / "bundle") + "/", ""),
                }
            ],
        }
        requests = learner._build_low_quality_raw_teacher_annotation_requests(
            bundle_cache_path=str(tmp_path / "bundle"),
            manifest=manifest,
            edge_id=1,
            model_id="yolo26n",
            target_model_metadata={},
        )
        learner.teacher_annotation_service = FakeService()
        learner._submit_low_quality_teacher_annotations(requests)

        assert submitted["sample_ids"] == ["sample-1"]
    finally:
        learner.close()


def test_async_disabled_does_not_run_sync_teacher_annotation(tmp_path) -> None:
    fake_teacher = FakeLargeOD()
    learner = CloudContinualLearner(
        _config(tmp_path, async_enabled=False),
        fake_teacher,
    )
    try:
        frame_dir = tmp_path / "frames"
        _write_image(frame_dir, "sample-1")

        annotations = learner._collect_teacher_annotations(
            str(frame_dir),
            ["sample-1"],
            include_empty=True,
            target_model_metadata={},
            edge_id=1,
            model_id="yolo26n",
        )

        assert fake_teacher.batch_calls == 0
        assert annotations == {}
    finally:
        learner.close()


def test_async_unresolved_without_worker_is_explicit(tmp_path) -> None:
    fake_teacher = FakeLargeOD()
    learner = CloudContinualLearner(
        _config(tmp_path, async_enabled=True),
        fake_teacher,
    )
    try:
        learner.teacher_annotation_worker.stop()
        learner.teacher_annotation_service.worker = None
        frame_dir = tmp_path / "frames"
        _write_image(frame_dir, "sample-1")

        annotations = learner._collect_teacher_annotations(
            str(frame_dir),
            ["sample-1"],
            include_empty=True,
            target_model_metadata={},
            edge_id=1,
            model_id="yolo26n",
        )

        assert annotations == {}
    finally:
        learner.close()


def test_teacher_annotation_scope_waits_for_torchlens_forward_guard(tmp_path) -> None:
    learner = CloudContinualLearner(
        _config(tmp_path, async_enabled=True),
        FakeLargeOD(),
    )
    entered_scope = threading.Event()

    def run_scope() -> None:
        with learner._teacher_annotation_scope(
            "teacher annotation guard regression",
            sample_count=1,
        ):
            entered_scope.set()

    try:
        with torchlens_forward_guard():
            thread = threading.Thread(target=run_scope)
            thread.start()
            assert not entered_scope.wait(timeout=0.05)

        assert entered_scope.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        learner.close()
