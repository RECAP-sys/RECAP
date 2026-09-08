from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from cloud.training import FixedSplitProxyEvaluator


class FakeLargeOD:
    def __init__(self) -> None:
        self.model_name = "fake_teacher"
        self.model = SimpleNamespace(label_schema="coco_91", class_names=["person"])

    def large_inference_batch(self, images, threshold=None):
        del threshold
        return [([[1, 2, 3, 4]], [1], [0.9]) for _image in images]

    def large_inference(self, image, threshold=None):
        del image, threshold
        return ([[1, 2, 3, 4]], [1], [0.9])


def _config(tmp_path: Path) -> SimpleNamespace:
    teacher_annotation = SimpleNamespace(
        async_enabled=False,
        cache_enabled=True,
        wait_timeout_sec=0.0,
        worker_batch_size=4,
        worker_max_queue_size=64,
        worker_max_retries=0,
        oom_retry_enabled=True,
        min_worker_batch_size=1,
        cache_root_dir=str(tmp_path / "teacher_cache"),
    )
    feature_cache = SimpleNamespace(
        shard_root_dir=str(tmp_path / "feature_shards"),
        view_root_dir=str(tmp_path / "training_views"),
        storage_format="safetensors_shard",
        accepted_storage_formats=["safetensors_shard", "npy_memmap_shard"],
        materialization_mode="direct_ref",
        view_source="recent_training_window",
    )
    continual_learning = SimpleNamespace(
        num_epoch=1,
        max_concurrent_jobs=1,
        batch_size=2,
        trace_batch_size=2,
        feature_cache_mode="auto",
        feature_cache=feature_cache,
        teacher_annotation_threshold=0.5,
        teacher_batch_size=2,
        teacher_annotation=teacher_annotation,
        proxy_eval_interval_rounds=1,
        proxy_eval_patience=0,
        proxy_eval_min_delta=0.0,
        proxy_eval_max_samples=0,
        proxy_eval_validation_fraction=0.2,
        proxy_eval_max_dets=500,
        proxy_eval_frame_cache_enabled=True,
        training_frame_count_by_model={"yolo26n": 3},
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


def test_cloud_orchestrator_imports_initializes_and_alias_survives(tmp_path) -> None:
    from cloud.orchestrator import CloudContinualLearner, CloudFixedSplitOrchestrator

    learner = CloudFixedSplitOrchestrator(_config(tmp_path), FakeLargeOD())
    alias = CloudContinualLearner(_config(tmp_path / "alias"), FakeLargeOD())
    try:
        assert learner.max_concurrent_jobs == 1
        assert learner.settings.batch_size == 2
        assert learner.training_frame_count == 3
        assert Path(learner.weight_folder) == Path.cwd() / "model_management" / "models"
        assert isinstance(alias, CloudFixedSplitOrchestrator)
    finally:
        learner.close()
        alias.close()


def test_orchestrator_does_not_import_private_proxy_helpers() -> None:
    source = Path("cloud/orchestrator.py").read_text(encoding="utf-8")

    assert "from cloud.training." + "proxy_eval" not in source
    assert "import _" not in source


def test_fixed_split_pipeline_does_not_import_private_proxy_helpers() -> None:
    source = Path("cloud/orchestration/fixed_split_pipeline.py").read_text(
        encoding="utf-8",
    )

    assert "from cloud.training." + "proxy_eval import (" not in source
    assert "_evaluate_detection_proxy_map" not in source
    assert "_snapshot_model_state" not in source
    assert "_proxy_metrics_indicate_dead_detector" not in source
    assert "_fixed_split_proxy_rejection_reason" not in source
    assert "_load" + "_cached_split_batches" not in source


def test_fixed_split_dependencies_does_not_bridge_private_proxy_helpers() -> None:
    source = Path("cloud/orchestration/fixed_split_dependencies.py").read_text(
        encoding="utf-8",
    )

    assert "import cloud.training." + "proxy_eval as " + "_proxy" + "_eval" not in source
    assert "_proxy" + "_eval." not in source
    assert "from cloud.training." + "proxy_eval import " + "_" not in source


def test_full_image_retrain_remains_rejected_public_stub(tmp_path) -> None:
    from cloud.orchestrator import CloudFixedSplitOrchestrator

    learner = CloudFixedSplitOrchestrator(_config(tmp_path), FakeLargeOD())
    try:
        success, model_data, message = learner.get_ground_truth_and_retrain(
            1,
            [1],
            str(tmp_path),
        )
    finally:
        learner.close()

    assert success is False
    assert model_data == ""
    assert "full-frame retrain is unavailable" in message


def test_legacy_low_quality_bundle_manifest_is_rejected(tmp_path) -> None:
    from cloud.orchestrator import CloudFixedSplitOrchestrator

    bundle_dir = tmp_path / "legacy_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps({"protocol_version": "legacy"}),
        encoding="utf-8",
    )
    learner = CloudFixedSplitOrchestrator(_config(tmp_path), FakeLargeOD())
    try:
        success, model_data, message = learner.get_ground_truth_and_fixed_split_retrain(
            1,
            str(bundle_dir),
        )
    finally:
        learner.close()

    assert success is False
    assert model_data == ""
    assert "trigger_manifest.json" in message
    assert "bundle_manifest.json uploads are not supported" in message


def test_proxy_evaluator_formats_map_50_95_summary() -> None:
    evaluator = FixedSplitProxyEvaluator(
        device=torch.device("cpu"),
        default_batch_size=2,
        max_samples=8,
    )

    summary = evaluator.format_summary(
        None,
        {
            "primary_metric": 0.42,
            "map_50": 0.7,
            "map_75": 0.5,
            "mar_500": 0.6,
            "evaluated_samples": 4,
            "skipped_empty_gt": 1,
            "skipped_missing_frame": 0,
        },
    )

    assert summary is not None
    assert "proxy_mAP_50_95 best=0.4200" in summary
    assert "mAP_50=0.7000" in summary


def test_tinynext_evaluation_uses_detection_metric_without_threshold_calibration(
    monkeypatch,
) -> None:
    evaluator = FixedSplitProxyEvaluator(
        device=torch.device("cpu"),
        default_batch_size=2,
        max_samples=30,
    )
    gt_annotations = {
        f"sample-{index:02d}": {"boxes": [[1, 2, 3, 4]], "labels": [1]} for index in range(30)
    }

    calls = {}

    def fake_evaluate_detection(*args, **kwargs):
        del args
        calls["max_samples"] = kwargs["max_samples"]
        return {
            "primary_metric": 0.42,
            "map_50_95": 0.42,
            "evaluated_samples": 30,
        }

    monkeypatch.setattr(evaluator, "evaluate_detection", fake_evaluate_detection)

    metrics = evaluator.evaluate_tinynext(
        torch.nn.Linear(1, 1),
        frame_dir="unused",
        gt_annotations=gt_annotations,
        model_name="tinynext",
        stage_label="baseline",
    )

    assert metrics["primary_metric"] == 0.42
    assert calls["max_samples"] == 30
    assert "full_proxy_evaluation_skipped" not in metrics
