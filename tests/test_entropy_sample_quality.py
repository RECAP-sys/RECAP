from __future__ import annotations

import json
import os

import numpy as np
import torch

from edge.sample_quality import (
    HIGH_QUALITY,
    LOW_QUALITY,
    QUALITY_METHOD,
    EntropyQualityClassifier,
    EntropyQualityStats,
)
from edge.sample_store import EdgeSampleStore
from model_management.payload import boundary_payload_from_tensors
from tools.evaluate_entropy_quality_against_teacher import (
    TEACHER_VERIFIED_HIGH,
    TEACHER_VERIFIED_LOW,
    evaluate_agreement,
)


def _classifier(**overrides) -> EntropyQualityClassifier:
    defaults = {
        "output_warmup_samples": 0,
        "feature_warmup_samples": 0,
        "feature_min_std": 1.0e-4,
    }
    defaults.update(overrides)
    return EntropyQualityClassifier(**defaults)


def _payload(values: list[float] | torch.Tensor):
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(1, -1)
    return boundary_payload_from_tensors(
        {"boundary": tensor},
        split_id="after:test",
        graph_signature="entropy-test",
        batch_size=1,
    )


def _pred(
    output_entropy: float,
    *,
    boxes: list[list[float]] | None = None,
    scores: list[float] | None = None,
) -> dict[str, object]:
    resolved_boxes = [[0.0, 0.0, 2.0, 2.0]] if boxes is None else boxes
    resolved_scores = scores if scores is not None else ([0.99] if resolved_boxes != [] else [])
    return {
        "boxes": resolved_boxes,
        "labels": [1] * len(resolved_boxes),
        "scores": resolved_scores,
        "output_entropy": output_entropy,
    }


def test_low_output_entropy_low_feature_deviation_is_high_quality() -> None:
    quality = _classifier().classify(
        _pred(0.1),
        _payload([1.0, 1.0, 1.0, 1.0]),
        "model-a",
        "split-a",
        "abi-a",
    )

    assert quality.quality == HIGH_QUALITY
    assert quality.output_reliable is True
    assert quality.feature_normal is True


def test_high_output_entropy_low_feature_deviation_is_low_quality() -> None:
    classifier = _classifier()
    classifier.classify(_pred(0.1), _payload([1.0, 1.0, 1.0, 1.0]), "m", "s", "a")

    quality = classifier.classify(
        _pred(0.9),
        _payload([1.0, 1.0, 1.0, 1.0]),
        "m",
        "s",
        "a",
    )

    assert quality.quality == LOW_QUALITY
    assert quality.output_reliable is False
    assert quality.feature_normal is True


def test_low_output_entropy_high_feature_deviation_is_low_quality() -> None:
    classifier = _classifier(feature_deviation_threshold=1.5)
    classifier.classify(_pred(0.2), _payload([1.0, 1.0, 1.0, 1.0]), "m", "s", "a")

    quality = classifier.classify(
        _pred(0.05),
        _payload([100.0, 0.0, 0.0, 0.0]),
        "m",
        "s",
        "a",
    )

    assert quality.quality == LOW_QUALITY
    assert quality.output_reliable is True
    assert quality.feature_normal is False


def test_low_output_entropy_low_confidence_fails_closed() -> None:
    quality = _classifier(output_min_detection_confidence=0.85).classify(
        _pred(0.01, scores=[0.72]),
        _payload([1.0, 1.0, 1.0, 1.0]),
        "m",
        "s",
        "a",
    )

    assert quality.quality == LOW_QUALITY
    assert quality.output_reliable is False
    assert quality.output_confident is False
    assert "output_confidence_low" in quality.reason


def test_high_output_entropy_high_feature_deviation_is_low_quality() -> None:
    classifier = _classifier(feature_deviation_threshold=1.5)
    classifier.classify(_pred(0.2), _payload([1.0, 1.0, 1.0, 1.0]), "m", "s", "a")

    quality = classifier.classify(
        _pred(0.9),
        _payload([100.0, 0.0, 0.0, 0.0]),
        "m",
        "s",
        "a",
    )

    assert quality.quality == LOW_QUALITY
    assert quality.output_reliable is False
    assert quality.feature_normal is False


def test_empty_predictions_are_low_quality() -> None:
    quality = _classifier().classify(
        _pred(0.01, boxes=[]),
        _payload([1.0, 1.0, 1.0, 1.0]),
        "m",
        "s",
        "a",
    )

    assert quality.quality == LOW_QUALITY
    assert "empty_predictions" in quality.reason


def test_feature_entropy_is_deterministic() -> None:
    classifier = _classifier(feature_max_elements=4)
    payload = _payload(torch.arange(16, dtype=torch.float32))

    first = classifier._compute_feature_entropy(payload)
    second = classifier._compute_feature_entropy(payload)

    assert first == second


def test_cpu_feature_entropy_samples_before_full_tensor_abs(monkeypatch) -> None:
    classifier = _classifier(feature_max_elements=32)
    tensor = torch.arange(4096, dtype=torch.float32)
    observed_sizes: list[int] = []
    original_abs = np.abs

    def tracking_abs(values):
        observed_sizes.append(int(values.size))
        return original_abs(values)

    monkeypatch.setattr(np, "abs", tracking_abs)

    first = classifier._tensor_activation_entropy(tensor)
    cached_indices = classifier._feature_sample_indices[tensor.numel()]
    second = classifier._tensor_activation_entropy(tensor)

    assert first == second
    assert observed_sizes == [32, 32]
    assert classifier._feature_sample_indices[tensor.numel()] is cached_indices


def test_output_entropy_threshold_adapts_from_prior_window() -> None:
    classifier = _classifier(output_percentile=25.0)
    for value in [0.10, 0.20, 0.30, 0.40]:
        classifier.classify(_pred(value), _payload([1.0, 1.0, 1.0, 1.0]), "m", "s", "a")

    quality = classifier.classify(
        _pred(0.35),
        _payload([1.0, 1.0, 1.0, 1.0]),
        "m",
        "s",
        "a",
    )

    assert quality.output_entropy_threshold is not None
    assert quality.output_entropy_threshold < 0.35
    assert quality.quality == LOW_QUALITY


def test_output_entropy_warmup_fails_closed() -> None:
    quality = _classifier(output_warmup_samples=2, feature_warmup_samples=0).classify(
        _pred(0.01),
        _payload([1.0, 1.0, 1.0, 1.0]),
        "m",
        "s",
        "a",
    )

    assert quality.quality == LOW_QUALITY
    assert "output_entropy_warmup" in quality.reason


def test_feature_entropy_warmup_fails_closed() -> None:
    quality = _classifier(output_warmup_samples=0, feature_warmup_samples=2).classify(
        _pred(0.01),
        _payload([1.0, 1.0, 1.0, 1.0]),
        "m",
        "s",
        "a",
    )

    assert quality.quality == LOW_QUALITY
    assert "feature_entropy_warmup" in quality.reason


def test_feature_ema_is_isolated_by_model_split_and_abi() -> None:
    classifier = _classifier(feature_warmup_samples=1)
    classifier.classify(_pred(0.2), _payload([1.0, 1.0, 1.0, 1.0]), "m", "s", "abi-a")

    quality = classifier.classify(
        _pred(0.01),
        _payload([100.0, 0.0, 0.0, 0.0]),
        "m",
        "s",
        "abi-b",
    )

    assert quality.feature_entropy_deviation is None
    assert "feature_entropy_warmup" in quality.reason


def test_feature_anomaly_does_not_poison_reference_by_default() -> None:
    classifier = _classifier(
        feature_warmup_samples=1,
        feature_relative_std_floor=0.01,
        feature_update_on_anomaly=False,
    )
    classifier.classify(_pred(0.1), _payload([1.0, 1.0, 1.0, 1.0]), "m", "s", "a")

    first = classifier.classify(
        _pred(0.1), _payload([100.0, 0.0, 0.0, 0.0]), "m", "s", "a"
    )
    second = classifier.classify(
        _pred(0.1), _payload([100.0, 0.0, 0.0, 0.0]), "m", "s", "a"
    )

    assert first.feature_normal is False
    assert second.feature_entropy_deviation == first.feature_entropy_deviation


def test_quality_calibration_can_be_reset_after_model_update() -> None:
    classifier = _classifier()
    classifier.classify(_pred(0.1), _payload([1.0, 1.0]), "m", "s", "a")

    classifier.reset()

    assert classifier._output_windows == {}
    assert classifier._feature_states == {}


def test_sample_store_persists_minimal_quality_metadata_by_default(tmp_path) -> None:
    store = EdgeSampleStore(str(tmp_path / "store"))
    payload = _payload([1.0, 1.0, 1.0, 1.0])

    record = store.store_sample(
        sample_id="sample-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="split-a",
        model_id="model-a",
        model_version="0",
        quality_metadata={"method": QUALITY_METHOD, "quality": HIGH_QUALITY},
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
    )

    metadata_path = tmp_path / "store" / record.metadata_relpath
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["quality"] == {"method": QUALITY_METHOD, "quality": HIGH_QUALITY}
    assert metadata["quality_bucket"] == HIGH_QUALITY
    assert "output_entropy" not in json.dumps(metadata["quality"])
    assert "quality" + "_score" not in metadata
    assert "risk" + "_score" not in metadata


def test_sample_store_persists_debug_stats_only_when_requested(tmp_path) -> None:
    store = EdgeSampleStore(str(tmp_path / "store"))
    payload = _payload([1.0, 1.0, 1.0, 1.0])
    stats = EntropyQualityStats(
        output_entropy=0.1,
        output_entropy_threshold=0.2,
        output_confidence=0.9,
        output_confidence_threshold=0.85,
        output_confident=True,
        feature_entropy=0.9,
        feature_entropy_mean=0.8,
        feature_entropy_std=0.1,
        feature_entropy_deviation=1.0,
        feature_deviation_threshold=1.5,
        output_reliable=True,
        feature_normal=True,
        edge_pseudo_label_trusted=True,
        quality=HIGH_QUALITY,
        reason="trusted_edge_pseudo_label",
    )

    record = store.store_sample(
        sample_id="sample-debug",
        frame_index=1,
        confidence=0.9,
        split_config_id="split-a",
        model_id="model-a",
        model_version="0",
        quality_metadata=stats.quality_metadata(persist_debug_stats=True),
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=payload,
    )

    metadata_path = tmp_path / "store" / record.metadata_relpath
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["quality"]["quality"] == HIGH_QUALITY
    assert metadata["quality"]["debug"]["output_entropy"] == 0.1
    assert metadata["quality"]["debug"]["edge_pseudo_label_trusted"] is True


def test_quality_metadata_drives_protocol_bucket_alias(tmp_path) -> None:
    store = EdgeSampleStore(str(tmp_path / "store"))
    record = store.store_sample(
        sample_id="low-1",
        frame_index=1,
        confidence=0.2,
        split_config_id="split-a",
        model_id="model-a",
        model_version="0",
        quality_metadata={"method": QUALITY_METHOD, "quality": LOW_QUALITY},
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload([1.0, 1.0]),
        raw_frame=np.zeros((4, 4, 3), dtype=np.uint8),
    )

    assert record.quality_bucket == LOW_QUALITY
    assert record.has_raw_sample is True


def test_offline_evaluator_confusion_matrix_and_metrics() -> None:
    report = evaluate_agreement(
        [
            {
                "sample_id": "trusted-ok",
                "predicted_quality": HIGH_QUALITY,
                "edge_prediction": {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
                "teacher_prediction": {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
            },
            {
                "sample_id": "trusted-bad",
                "predicted_quality": HIGH_QUALITY,
                "edge_prediction": {"boxes": [[20, 20, 30, 30]], "labels": [1], "scores": [0.9]},
                "teacher_prediction": {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
            },
            {
                "sample_id": "teacher-needed",
                "predicted_quality": LOW_QUALITY,
                "edge_prediction": {"boxes": [[20, 20, 30, 30]], "labels": [1], "scores": [0.9]},
                "teacher_prediction": {"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]},
            },
        ]
    )

    assert report["confusion_matrix"][HIGH_QUALITY][TEACHER_VERIFIED_HIGH] == 1
    assert report["confusion_matrix"][HIGH_QUALITY][TEACHER_VERIFIED_LOW] == 1
    assert report["confusion_matrix"][LOW_QUALITY][TEACHER_VERIFIED_LOW] == 1
    assert report["high_quality_precision"] == 0.5
    assert report["false_trusted_rate"] == 0.5
    assert report["low_quality_recall"] == 0.5
    assert report["teacher_load"] == 1 / 3


def test_legacy_quality_module_is_absent_from_active_path() -> None:
    assert not os.path.exists("edge/evidence.py")
    active_files = [
        "edge/edge_worker.py",
        "edge/sample_store.py",
        "edge/transmit.py",
        "edge/window_drift_detector.py",
        "edge/resource_aware_trigger.py",
    ]
    banned_terms = [
        "Quality" + "Assessor",
        "quality" + "_assessor",
        "uncovered" + "_evidence",
        "risk" + "_score",
    ]
    for relative in active_files:
        text = open(relative, "r", encoding="utf-8").read()
        for term in banned_terms:
            assert term not in text
