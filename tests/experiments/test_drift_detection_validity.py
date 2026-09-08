from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from experiments.drift_detection_validity.detection_metrics import (
    box_iou_xyxy,
    detection_f1,
    window_detection_f1,
)
from experiments.drift_detection_validity.drift_signal_extractor import (
    clean_baseline_mask,
    ema_update,
    finalize_signal_records,
)
from experiments.drift_detection_validity.evaluate_real_weather_scenes import _scene_video_rows
from experiments.drift_detection_validity.online_trigger_analysis import (
    extract_harmful_drift_events,
    match_triggers_to_events,
    replay_triggers,
)
from experiments.drift_detection_validity.signal_validity_analysis import (
    average_ranks,
    best_f1_threshold,
    pearson_correlation,
    pr_auc_score,
    roc_auc_score,
    spearman_correlation,
)


def test_real_weather_scene_config_requires_explicit_scene_fields(tmp_path: Path) -> None:
    videos = {}
    for scene_id in ("rainy", "snowy"):
        path = tmp_path / f"{scene_id}.mp4"
        path.write_bytes(b"placeholder")
        videos[scene_id] = path
    with pytest.raises(ValueError, match=r"scene_id is required"):
        _scene_video_rows(
            {
                "data": {
                    "scene_videos": [
                        {
                            "scene_label": "Rainy",
                            "video_path": str(videos["rainy"]),
                        },
                        {
                            "scene_id": "snowy",
                            "scene_label": "Snowy",
                            "video_path": str(videos["snowy"]),
                        },
                    ]
                }
            }
        )
    with pytest.raises(ValueError, match=r"scene_label is required"):
        _scene_video_rows(
            {
                "data": {
                    "scene_videos": [
                        {
                            "scene_id": "rainy",
                            "video_path": str(videos["rainy"]),
                        },
                        {
                            "scene_id": "snowy",
                            "scene_label": "Snowy",
                            "video_path": str(videos["snowy"]),
                        },
                    ]
                }
            }
        )
    with pytest.raises(ValueError, match=r"must be 'rainy'"):
        _scene_video_rows(
            {
                "data": {
                    "scene_videos": [
                        {
                            "scene_id": "snowy",
                            "scene_label": "Snowy",
                            "video_path": str(videos["snowy"]),
                        },
                        {
                            "scene_id": "rainy",
                            "scene_label": "Rainy",
                            "video_path": str(videos["rainy"]),
                        },
                    ]
                }
            }
        )

    rows = _scene_video_rows(
        {
            "data": {
                "scene_videos": [
                    {
                        "scene_id": "rainy",
                        "scene_label": "Rainy",
                        "video_path": str(videos["rainy"]),
                    },
                    {
                        "scene_id": "snowy",
                        "scene_label": "Snowy",
                        "video_path": str(videos["snowy"]),
                    },
                ]
            }
        }
    )
    assert [row["scene_id"] for row in rows] == ["rainy", "snowy"]


def test_iou_matching_detection_f1_and_window_aggregation() -> None:
    boxes1 = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    boxes2 = np.asarray([[0, 0, 10, 10]], dtype=float)
    iou = box_iou_xyxy(boxes1, boxes2)
    assert iou.shape == (2, 1)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[1, 0] == pytest.approx(0.0)

    student = {"boxes": boxes1, "labels": [1, 1], "scores": [0.9, 0.8]}
    teacher = {"boxes": boxes2, "labels": [1], "scores": [0.95]}
    score = detection_f1(student, teacher, iou_threshold=0.5)
    assert score["tp"] == pytest.approx(1.0)
    assert score["fp"] == pytest.approx(1.0)
    assert score["fn"] == pytest.approx(0.0)
    assert score["precision"] == pytest.approx(0.5)
    assert score["recall"] == pytest.approx(1.0)

    student_records = [
        {"global_frame_id": 0, "domain": "rainy", "prediction": student},
        {
            "global_frame_id": 1,
            "domain": "rainy",
            "prediction": {"boxes": [], "labels": [], "scores": []},
        },
    ]
    teacher_records = [
        {"global_frame_id": 0, "domain": "rainy", "prediction": teacher},
        {"global_frame_id": 1, "domain": "rainy", "prediction": teacher},
    ]
    windows = window_detection_f1(student_records, teacher_records, window_size=2, stride=2)
    assert len(windows) == 1
    assert windows[0]["tp"] == 1
    assert windows[0]["fp"] == 1
    assert windows[0]["fn"] == 1


def test_ema_and_clean_baseline_normalization() -> None:
    assert ema_update(None, 2.0, 0.25) == pytest.approx(2.0)
    assert ema_update(2.0, 6.0, 0.25) == pytest.approx(3.0)

    config = {
        "signals": {
            "ema_alpha": 0.5,
            "eps": 1.0e-8,
            "full_score_entropy_weight": 0.5,
            "full_score_feature_weight": 0.5,
        }
    }
    records = [
        {
            "domain": "rainy",
            "domain_index": 0,
            "mean_confidence": 0.9,
            "output_entropy": 0.1,
            "_boundary_feature_vector": np.asarray([0.0, 0.0]),
        },
        {
            "domain": "rainy",
            "domain_index": 0,
            "mean_confidence": 0.8,
            "output_entropy": 0.2,
            "_boundary_feature_vector": np.asarray([0.1, 0.0]),
        },
        {
            "domain": "snowy",
            "domain_index": 1,
            "mean_confidence": 0.4,
            "output_entropy": 0.8,
            "_boundary_feature_vector": np.asarray([1.0, 1.0]),
        },
    ]
    finalized, baseline = finalize_signal_records(records, config, clean_baseline_mask(records))
    assert baseline["mean_confidence"] == pytest.approx(0.85)
    assert finalized[2]["confidence_drop_signal"] == pytest.approx(0.45)
    assert finalized[2]["confidence_drop_z"] > 1.0
    assert finalized[2]["boundary_feature_deviation"] > finalized[0]["boundary_feature_deviation"]
    assert finalized[2]["full_drift_score"] > finalized[0]["full_drift_score"]
    assert "_boundary_feature_vector" not in finalized[0]


def test_local_scalar_metrics_with_ties_and_degenerate_auc(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    ranks = average_ranks([1.0, 1.0, 3.0])
    assert ranks.tolist() == pytest.approx([1.5, 1.5, 3.0])
    assert spearman_correlation([1, 1, 3], [1, 2, 3]) > 0.8
    assert roc_auc_score([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.8]) == pytest.approx(1.0)
    assert pr_auc_score([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.8]) == pytest.approx(1.0)
    best = best_f1_threshold([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.8])
    assert best["f1"] == pytest.approx(1.0)

    caplog.clear()
    assert math.isnan(roc_auc_score([1, 1], [0.2, 0.3]))
    assert "only one class" in caplog.text


def test_harmful_event_extraction_trigger_cooldown_and_matching() -> None:
    rows = [
        {
            "window_start_frame": index * 10,
            "window_end_frame": index * 10 + 9,
            "domain_majority": "rainy" if index < 2 else "snowy",
            "f1_drop": 0.0 if index < 2 else 0.2,
            "mean_full_drift_score_z": 2.0 if index >= 2 else 0.0,
        }
        for index in range(8)
    ]
    events = extract_harmful_drift_events(
        rows,
        harmful_f1_drop_threshold=0.1,
        harmful_consecutive_windows=2,
    )
    assert events == [
        {"frame": 20, "end_frame": 79, "domain": "snowy", "transition_frame": 20}
    ]

    triggers = replay_triggers(
        rows,
        method="recap_full",
        signal_column="mean_full_drift_score_z",
        threshold=1.0,
        trigger_consecutive_windows=1,
        cooldown_windows=3,
        rearm_requires_below_threshold=False,
    )
    assert triggers == [20, 60]

    edge_triggers = replay_triggers(
        rows,
        method="recap_full",
        signal_column="mean_full_drift_score_z",
        threshold=1.0,
        trigger_consecutive_windows=1,
        cooldown_windows=3,
        rearm_requires_below_threshold=True,
    )
    assert edge_triggers == [20]

    matched = match_triggers_to_events(events, triggers, tolerance_frames=15, total_frames=80)
    assert matched["detected"] == 1
    assert matched["false_triggers"] == 1
    assert matched["missed"] == 0
    assert matched["avg_detection_delay_frames"] == pytest.approx(0.0)

    early = match_triggers_to_events(
        [{"frame": 40, "domain": "rainy", "transition_frame": 40}],
        [20],
        tolerance_frames=0,
        early_tolerance_frames=20,
        total_frames=80,
    )
    assert early["detected"] == 1
    assert early["avg_detection_delay_frames"] == pytest.approx(-20.0)

    episode_rows = [
        {
            "window_start_frame": index * 10,
            "window_end_frame": index * 10 + 9,
            "domain_majority": f"domain_{index}",
            "f1_drop": 0.2 if index in {1, 2, 4, 5} else 0.0,
        }
        for index in range(7)
    ]
    merged = extract_harmful_drift_events(
        episode_rows,
        harmful_f1_drop_threshold=0.1,
        harmful_consecutive_windows=1,
        harmful_merge_gap_windows=1,
    )
    assert [event["frame"] for event in merged] == [10]
