from __future__ import annotations

import pytest

from cloud.training.proxy_eval import _evaluate_detection_proxy_metrics_from_cache


def test_proxy_coco_metrics_perfect_match_scores_one() -> None:
    metrics = _evaluate_detection_proxy_metrics_from_cache(
        {
            "prediction_rows": [
                (
                    [[0, 0, 10, 10]],
                    [1],
                    {
                        "boxes": [[0, 0, 10, 10]],
                        "labels": [1],
                        "scores": [0.9],
                    },
                )
            ],
            "total_gt_samples": 1,
        },
        threshold_high=0.5,
        max_dets=500,
    )

    assert metrics["primary_metric_name"] == "proxy_mAP_50_95"
    assert metrics["primary_metric"] == pytest.approx(1.0)
    assert metrics["map_50_95"] == pytest.approx(1.0)
    assert metrics["map_50"] == pytest.approx(1.0)
    assert metrics["evaluated_samples"] == 1
    assert metrics["nonempty_predictions"] == 1
    assert metrics["total_prediction_boxes"] == 1


def test_proxy_coco_metrics_does_not_drop_low_confidence_true_positive() -> None:
    metrics = _evaluate_detection_proxy_metrics_from_cache(
        {
            "prediction_rows": [
                (
                    [[0, 0, 10, 10]],
                    [1],
                    {
                        "boxes": [[0, 0, 10, 10]],
                        "labels": [1],
                        "scores": [0.1],
                    },
                )
            ],
            "total_gt_samples": 1,
        },
        threshold_high=0.5,
        max_dets=500,
    )

    assert metrics["map_50_95"] == pytest.approx(1.0)
    assert metrics["total_prediction_boxes"] == 1


def test_proxy_coco_metrics_uses_iou_50_95_not_only_iou_50() -> None:
    metrics = _evaluate_detection_proxy_metrics_from_cache(
        {
            "prediction_rows": [
                (
                    [[0, 0, 10, 10]],
                    [1],
                    {
                        "boxes": [[0, 0, 8, 8]],
                        "labels": [1],
                        "scores": [0.9],
                    },
                )
            ],
            "total_gt_samples": 1,
        },
        threshold_high=0.5,
        max_dets=500,
    )

    assert metrics["map_50"] == pytest.approx(1.0)
    assert metrics["map_75"] == pytest.approx(0.0)
    assert metrics["map_50_95"] == pytest.approx(0.3)


def test_proxy_coco_metrics_counts_empty_predictions_as_zero_map() -> None:
    metrics = _evaluate_detection_proxy_metrics_from_cache(
        {
            "prediction_rows": [
                (
                    [[0, 0, 10, 10]],
                    [1],
                    {"boxes": [], "labels": [], "scores": []},
                )
            ],
            "total_gt_samples": 1,
        },
        threshold_high=0.5,
        max_dets=500,
    )

    assert metrics["map_50_95"] == pytest.approx(0.0)
    assert metrics["map_50"] == pytest.approx(0.0)
    assert metrics["nonempty_predictions"] == 0
    assert metrics["total_prediction_boxes"] == 0


def test_proxy_coco_metrics_counts_empty_gt_false_positive() -> None:
    metrics = _evaluate_detection_proxy_metrics_from_cache(
        {
            "prediction_rows": [
                (
                    [[0, 0, 10, 10]],
                    [1],
                    {
                        "boxes": [[0, 0, 10, 10]],
                        "labels": [1],
                        "scores": [0.5],
                    },
                ),
                (
                    [],
                    [],
                    {
                        "boxes": [[20, 20, 30, 30]],
                        "labels": [1],
                        "scores": [0.9],
                    },
                ),
            ],
            "total_gt_samples": 2,
        },
        threshold_high=0.5,
        max_dets=500,
    )

    assert metrics["evaluated_samples"] == 2
    assert metrics["nonempty_predictions"] == 2
    assert metrics["total_prediction_boxes"] == 2
    assert metrics["map_50_95"] == pytest.approx(0.5)


def test_proxy_coco_metrics_labels_mar_with_configured_max_dets() -> None:
    metrics = _evaluate_detection_proxy_metrics_from_cache(
        {
            "prediction_rows": [
                (
                    [[0, 0, 10, 10]],
                    [1],
                    {
                        "boxes": [[0, 0, 10, 10]],
                        "labels": [1],
                        "scores": [0.9],
                    },
                )
            ],
            "total_gt_samples": 1,
        },
        threshold_high=0.5,
        max_dets=100,
    )

    assert metrics["max_dets"] == 100
    assert "mar_100" in metrics
    assert "mar_500" not in metrics
