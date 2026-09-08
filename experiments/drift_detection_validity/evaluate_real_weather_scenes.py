#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.drift_detection_validity.experiment_io import (  # noqa: E402
    load_config,
    output_dir,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_text,
    resolve_project_path,
)
from experiments.drift_detection_validity.real_weather_backend import (  # noqa: E402
    RealWeatherBackend,
)
from experiments.drift_detection_validity.detection_metrics import (  # noqa: E402
    detection_f1,
    match_detections,
    prediction_to_jsonable,
)
from experiments.drift_detection_validity.drift_signal_extractor import (  # noqa: E402
    DriftSignalExtractor,
    clean_baseline_mask,
    finalize_signal_records,
)
from experiments.drift_detection_validity.online_trigger_analysis import (  # noqa: E402
    analyze_online_triggers,
)
from experiments.drift_detection_validity.signal_validity_analysis import (  # noqa: E402
    analyze_signal_validity,
)


FRAME_FIELDS = [
    "global_frame_id",
    "scene_id",
    "scene_label",
    "video_path",
    "frame_index",
    "time_seconds",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "student_count",
    "teacher_count",
    "student_mean_confidence",
    "teacher_mean_confidence",
]

SUMMARY_FIELDS = [
    "scene_id",
    "scene_label",
    "video_path",
    "sampled_frames",
    "duration_seconds",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "mean_frame_precision",
    "mean_frame_recall",
    "mean_frame_f1",
    "median_frame_f1",
    "total_tp",
    "total_fp",
    "total_fn",
    "mean_student_boxes",
    "mean_teacher_boxes",
    "mean_student_confidence",
    "mean_teacher_confidence",
]

DRIFT_FRAME_FIELDS = [
    "sequence_name",
    "global_frame_id",
    "scene_id",
    "scene_label",
    "source_frame_id",
    "domain",
    "domain_index",
    "precision",
    "recall",
    "f1",
    "student_num_boxes",
    "teacher_num_boxes",
    "student_mean_confidence",
    "teacher_mean_confidence",
    "mean_confidence",
    "confidence_drop_signal",
    "confidence_drop_z",
    "output_entropy",
    "objectness_weighted_entropy",
    "ema_output_entropy",
    "ema_output_entropy_z",
    "boundary_feature_mean",
    "boundary_feature_std",
    "boundary_feature_l2_norm",
    "boundary_feature_deviation",
    "boundary_feature_deviation_z",
    "ema_boundary_feature_deviation",
    "ema_boundary_feature_deviation_z",
    "full_drift_score",
    "full_drift_score_z",
]

WINDOW_METRIC_FIELDS = [
    "sequence_name",
    "window_id",
    "window_start_frame",
    "window_end_frame",
    "domain_majority",
    "scene_label_majority",
    "precision",
    "recall",
    "f1",
    "f1_base",
    "f1_drop",
    "is_harmful_drift_window",
    "mean_confidence_drop_signal",
    "mean_confidence_drop_z",
    "mean_output_entropy",
    "mean_ema_output_entropy",
    "mean_ema_output_entropy_z",
    "mean_boundary_feature_deviation",
    "mean_ema_boundary_feature_deviation",
    "mean_ema_boundary_feature_deviation_z",
    "mean_full_drift_score",
    "mean_full_drift_score_z",
]

REAL_WEATHER_SEQUENCE = "suwon5a_real_weather"
REQUIRED_SCENE_IDS = ("rainy", "snowy")
REQUIRED_SCENE_FIELDS = ("scene_id", "scene_label", "video_path")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _mean_score(prediction: Mapping[str, Any]) -> float:
    scores = [float(value) for value in prediction.get("scores", [])]
    return float(np.mean(scores)) if scores else 0.0


def _scene_video_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    data_cfg = require_mapping(config, "data")
    scenes_value = data_cfg.get("scene_videos")
    if not isinstance(scenes_value, list) or not scenes_value:
        raise ValueError("data.scene_videos must contain at least one scene.")
    if len(scenes_value) != len(REQUIRED_SCENE_IDS):
        raise ValueError(
            "data.scene_videos must contain exactly two scenes: "
            f"{', '.join(REQUIRED_SCENE_IDS)}."
        )
    scenes = list(scenes_value)
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, Mapping):
            raise TypeError(f"Scene entry {index} must be a mapping.")
        context = f"data.scene_videos[{index}]"
        for field in REQUIRED_SCENE_FIELDS:
            require_text(scene, field, context=context)
        scene_id = str(scene["scene_id"]).strip()
        expected_id = REQUIRED_SCENE_IDS[index]
        if scene_id != expected_id:
            raise ValueError(f"{context}.scene_id must be {expected_id!r}.")
        video_path = resolve_project_path(scene["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"Scene video does not exist: {video_path}")
        rows.append(
            {
                "scene_id": scene_id,
                "scene_label": str(scene["scene_label"]).strip(),
                "video_path": video_path,
            }
        )
    return rows


def _video_metadata(video_path: Path) -> dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    duration = float(frames) / fps if fps > 0 else 0.0
    return {
        "frames": float(frames),
        "fps": fps,
        "duration_seconds": duration,
        "width": float(width),
        "height": float(height),
    }


def _sample_frame_indices(
    *,
    frame_count: int,
    fps: float,
    frames_per_scene: int,
    start_margin_seconds: float,
    end_margin_seconds: float,
) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video frame count must be positive.")
    sample_count = max(1, min(int(frames_per_scene), frame_count))
    start = int(round(max(0.0, start_margin_seconds) * max(fps, 1.0)))
    end = frame_count - 1 - int(round(max(0.0, end_margin_seconds) * max(fps, 1.0)))
    if end < start:
        start = 0
        end = frame_count - 1
    if sample_count == 1:
        return [int((start + end) // 2)]
    indices = np.linspace(start, end, num=sample_count)
    rounded = [int(round(value)) for value in indices.tolist()]
    deduped: list[int] = []
    seen: set[int] = set()
    for index in rounded:
        clipped = max(0, min(frame_count - 1, index))
        if clipped not in seen:
            deduped.append(clipped)
            seen.add(clipped)
    return deduped


def _make_backend_config(config: Mapping[str, Any], video_path: Path) -> dict[str, Any]:
    backend_config = dict(config)
    backend_config["data"] = {
        **require_mapping(config, "data"),
        "video_path": str(video_path),
    }
    return backend_config


def _switch_backend_video(backend: RealWeatherBackend, video_path: Path) -> None:
    backend._capture.release()
    backend.video_path = video_path
    backend._capture = backend.cv2.VideoCapture(str(video_path))
    if not backend._capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    backend._last_source_id = None
    backend._last_frame = None


def _draw_prediction(
    image: np.ndarray,
    prediction: Mapping[str, Any],
    *,
    color: tuple[int, int, int],
    prefix: str,
) -> None:
    boxes = list(prediction.get("boxes") or [])
    labels = list(prediction.get("labels") or [])
    scores = list(prediction.get("scores") or [])
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
        x1 = max(0, min(image.shape[1] - 1, x1))
        x2 = max(0, min(image.shape[1] - 1, x2))
        y1 = max(0, min(image.shape[0] - 1, y1))
        y2 = max(0, min(image.shape[0] - 1, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = int(labels[index]) if index < len(labels) else -1
        score = float(scores[index]) if index < len(scores) else 0.0
        text = f"{prefix}{label}:{score:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(18, y1 - 4)
        cv2.rectangle(
            image,
            (x1, y_text - text_size[1] - 4),
            (x1 + text_size[0] + 6, y_text + 3),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (x1 + 3, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 24, 28),
            1,
            cv2.LINE_AA,
        )


def _letterbox(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def _put_header(image: np.ndarray, lines: Sequence[str]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 82), (246, 246, 246), -1)
    y = 24
    for index, line in enumerate(lines):
        scale = 0.62 if index == 0 else 0.48
        thickness = 2 if index == 0 else 1
        cv2.putText(
            image,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (30, 34, 38),
            thickness,
            cv2.LINE_AA,
        )
        y += 25


def _select_example_records(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, num=count)
    return [rows[int(round(value))] for value in indices.tolist()]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_float(row.get(key), math.nan) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def _majority(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _score_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (
        precision + recall
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _build_window_metrics(
    frame_rows: Sequence[Mapping[str, Any]],
    signal_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(frame_rows) != len(signal_rows):
        raise ValueError("frame_rows and signal_rows must have the same length")
    window_cfg = require_mapping(config, "window")
    size = max(1, require_int(window_cfg, "size", context="window"))
    stride = max(1, require_int(window_cfg, "stride", context="window"))
    harmful_threshold = require_float(
        window_cfg,
        "harmful_f1_drop_threshold",
        context="window",
    )
    combined_rows = [
        {**dict(signal_row), **dict(frame_row)}
        for frame_row, signal_row in zip(frame_rows, signal_rows)
    ]
    windows: list[dict[str, Any]] = []
    for window_id, start in enumerate(range(0, max(0, len(combined_rows) - size + 1), stride)):
        end = start + size
        subset = combined_rows[start:end]
        tp = sum(int(row.get("tp", 0)) for row in subset)
        fp = sum(int(row.get("fp", 0)) for row in subset)
        fn = sum(int(row.get("fn", 0)) for row in subset)
        score = _score_from_counts(tp, fp, fn)
        windows.append(
            {
                "sequence_name": REAL_WEATHER_SEQUENCE,
                "window_id": int(window_id),
                "window_start_frame": int(subset[0]["global_frame_id"]),
                "window_end_frame": int(subset[-1]["global_frame_id"]),
                "domain_majority": _majority(subset, "domain"),
                "scene_label_majority": _majority(subset, "scene_label"),
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1"],
                "mean_confidence_drop_signal": _mean(subset, "confidence_drop_signal"),
                "mean_confidence_drop_z": _mean(subset, "confidence_drop_z"),
                "mean_output_entropy": _mean(subset, "output_entropy"),
                "mean_ema_output_entropy": _mean(subset, "ema_output_entropy"),
                "mean_ema_output_entropy_z": _mean(subset, "ema_output_entropy_z"),
                "mean_boundary_feature_deviation": _mean(
                    subset,
                    "boundary_feature_deviation",
                ),
                "mean_ema_boundary_feature_deviation": _mean(
                    subset,
                    "ema_boundary_feature_deviation",
                ),
                "mean_ema_boundary_feature_deviation_z": _mean(
                    subset,
                    "ema_boundary_feature_deviation_z",
                ),
                "mean_full_drift_score": _mean(subset, "full_drift_score"),
                "mean_full_drift_score_z": _mean(subset, "full_drift_score_z"),
            }
        )
    if not signal_rows:
        raise ValueError("At least one signal row is required for window metrics.")
    baseline_domain = str(signal_rows[0]["domain"])
    baseline_f1 = [
        float(row["f1"]) for row in windows if str(row.get("domain_majority")) == baseline_domain
    ]
    f1_base = float(np.mean(baseline_f1)) if baseline_f1 else 0.0
    for row in windows:
        f1_drop = max(0.0, f1_base - float(row["f1"]))
        row["f1_base"] = f1_base
        row["f1_drop"] = f1_drop
        row["is_harmful_drift_window"] = bool(f1_drop >= harmful_threshold)
    return windows


def _save_scene_examples(
    *,
    scene: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    columns: int,
    tile_width: int,
    tile_height: int,
) -> None:
    examples = _select_example_records(rows, count=6)
    if not examples:
        return
    rows_count = int(np.ceil(len(examples) / float(columns)))
    header_height = 82
    canvas = np.full(
        (rows_count * (tile_height + header_height), columns * tile_width, 3),
        255,
        dtype=np.uint8,
    )
    for example_index, row in enumerate(examples):
        tile = np.asarray(row["rendered_frame"], dtype=np.uint8)
        tile = _letterbox(tile, width=tile_width, height=tile_height)
        metric = f"P/R/F1 {float(row['precision']):.2f}/{float(row['recall']):.2f}/{float(row['f1']):.2f}"
        counts = (
            f"TP/FP/FN {int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])}  "
            f"S/T boxes {int(row['student_count'])}/{int(row['teacher_count'])}"
        )
        title = f"{scene['scene_label']}  frame {int(row['frame_index'])}  t={float(row['time_seconds']):.1f}s"
        cell = np.full((tile_height + header_height, tile_width, 3), 255, dtype=np.uint8)
        _put_header(cell, [title, metric, counts])
        cell[header_height:, :] = tile
        y = (example_index // columns) * (tile_height + header_height)
        x = (example_index % columns) * tile_width
        canvas[y : y + cell.shape[0], x : x + cell.shape[1]] = cell
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _save_summary_plot(summary_rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["scene_label"]) for row in summary_rows]
    x = np.arange(len(labels))
    precision = [float(row["micro_precision"]) for row in summary_rows]
    recall = [float(row["micro_recall"]) for row in summary_rows]
    f1 = [float(row["micro_f1"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    width = 0.24
    ax.bar(x - width, precision, width=width, label="Precision", color="#4C78A8")
    ax.bar(x, recall, width=width, label="Recall", color="#F58518")
    ax.bar(x + width, f1, width=width, label="F1", color="#54A24B")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Student vs teacher pseudo-label")
    ax.set_xticks(x, labels)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for xpos, value in zip(x + width, f1):
        ax.text(xpos, min(1.0, value + 0.03), f"{value:.2f}", ha="center", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_drift_detection_plot(
    window_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not window_rows:
        return
    trigger_cfg = require_mapping(config, "trigger")
    threshold_cfg = require_mapping(trigger_cfg, "thresholds", context="trigger")
    threshold = require_float(threshold_cfg, "full_score_z", context="trigger.thresholds")
    window_cfg = require_mapping(config, "window")
    harmful_threshold = require_float(
        window_cfg,
        "harmful_f1_drop_threshold",
        context="window",
    )
    x = np.asarray(
        [
            (float(row["window_start_frame"]) + float(row["window_end_frame"])) / 2.0
            for row in window_rows
        ],
        dtype=np.float64,
    )
    f1_drop = np.asarray([float(row["f1_drop"]) for row in window_rows], dtype=np.float64)
    full_score = np.asarray(
        [float(row["mean_full_drift_score_z"]) for row in window_rows],
        dtype=np.float64,
    )
    labels = [str(row["domain_majority"]) for row in window_rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    colors = ["#B9C6D8" if label == labels[0] else "#E3A55F" for label in labels]
    ax.bar(x, f1_drop, width=5.8, color=colors, label="Teacher-pseudo F1 drop")
    ax.axhline(
        harmful_threshold,
        color="#A23B3B",
        linewidth=1.1,
        linestyle="--",
        label="Harmful-drift threshold",
    )
    ax.set_ylabel("F1 drop")
    ax.set_xlabel("Sampled frame index in real-weather stream")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E4E8", linewidth=0.8)

    score_ax = ax.twinx()
    score_ax.plot(
        x,
        full_score,
        color="#2F6B4F",
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="RECAP full score (z)",
    )
    score_ax.axhline(
        threshold,
        color="#2F6B4F",
        linewidth=1.0,
        linestyle=":",
        label="Trigger threshold",
    )
    score_ax.set_ylabel("Drift score z")
    score_ax.spines["top"].set_visible(False)

    handles, legend_labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = score_ax.get_legend_handles_labels()
    ax.legend(
        handles + extra_handles,
        legend_labels + extra_labels,
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    last_label = None
    for xpos, label in zip(x, labels):
        if label != last_label:
            ax.text(
                xpos,
                max(float(np.max(f1_drop)) * 1.05, harmful_threshold * 1.4),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
            last_label = label
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_html_report(
    *,
    output_path: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    scene_images: Mapping[str, Path],
    summary_plot: Path,
    drift_plot: Path | None = None,
    extra_plots: Sequence[tuple[str, Path]] = (),
) -> None:
    def relative(path: Path) -> str:
        return path.relative_to(output_path.parent).as_posix()

    rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['scene_label']))}</td>"
        f"<td>{int(row['sampled_frames'])}</td>"
        f"<td>{float(row['micro_precision']):.3f}</td>"
        f"<td>{float(row['micro_recall']):.3f}</td>"
        f"<td>{float(row['micro_f1']):.3f}</td>"
        f"<td>{int(row['total_tp'])}/{int(row['total_fp'])}/{int(row['total_fn'])}</td>"
        f"<td>{float(row['mean_student_boxes']):.2f}/{float(row['mean_teacher_boxes']):.2f}</td>"
        "</tr>"
        for row in summary_rows
    )
    figures_html = "\n".join(
        "<section>"
        f"<h2>{html.escape(str(row['scene_label']))}</h2>"
        f"<img src=\"{html.escape(relative(scene_images[str(row['scene_id'])]))}\" alt=\"{html.escape(str(row['scene_label']))}\">"
        "</section>"
        for row in summary_rows
    )
    extra_figures_html = "\n".join(
        "<section>"
        f"<h2>{html.escape(title)}</h2>"
        f"<img src=\"{html.escape(relative(path))}\" alt=\"{html.escape(title)}\">"
        "</section>"
        for title, path in extra_plots
        if path.exists()
    )
    output_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>Suwon real weather scene test</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:24px;color:#20242a;background:#fff}",
                "table{border-collapse:collapse;margin:16px 0 28px 0;font-size:14px}",
                "th,td{border:1px solid #d7dbe0;padding:7px 10px;text-align:right}",
                "th:first-child,td:first-child{text-align:left}",
                "img{max-width:100%;height:auto;border:1px solid #d7dbe0;margin:8px 0 28px 0}",
                "h1{font-size:24px;margin-bottom:6px} h2{font-size:18px;margin-top:24px}",
                ".note{color:#5a626d;font-size:14px;line-height:1.5}",
                "</style></head><body>",
                "<h1>Suwon #5a real weather scene test</h1>",
                "<p class=\"note\">All frames are sampled from real videos. Detection metrics compare the student detector against teacher pseudo-labels at IoU=0.5. Drift metrics use the rainy scene as the clean baseline, label harmful windows by teacher-pseudo-label F1 drop, and evaluate RECAP signals without using teacher predictions at trigger time.</p>",
                f"<img src=\"{html.escape(relative(summary_plot))}\" alt=\"summary plot\">",
                (
                    f"<img src=\"{html.escape(relative(drift_plot))}\" alt=\"drift detection plot\">"
                    if drift_plot is not None and drift_plot.exists()
                    else ""
                ),
                extra_figures_html,
                "<table><thead><tr><th>Scene</th><th>Frames</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP/FP/FN</th><th>Mean S/T boxes</th></tr></thead>",
                f"<tbody>{rows_html}</tbody></table>",
                figures_html,
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )


def write_real_weather_report(config: Mapping[str, Any]) -> Path:
    root = output_dir(config)
    records_dir = root / "records"
    figures_dir = root / "figures"
    summary_rows = _read_csv(records_dir / "real_weather_scene_summary.csv")
    if not summary_rows:
        raise ValueError("No real weather scene summary rows are available for the report.")
    scene_images = {
        str(row["scene_id"]): figures_dir / f"{row['scene_id']}_student_teacher_examples.png"
        for row in summary_rows
    }
    report_path = root / "real_weather_scene_report.html"
    _write_html_report(
        output_path=report_path,
        summary_rows=summary_rows,
        scene_images=scene_images,
        summary_plot=figures_dir / "real_weather_scene_metric_summary.png",
        drift_plot=figures_dir / "real_weather_drift_detection_effectiveness.png",
        extra_plots=(
            ("Signal validity summary", root / "plots" / "figure_signal_validity_summary.png"),
            ("Online trigger summary", root / "plots" / "figure_online_trigger_summary.png"),
        ),
    )
    return report_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def evaluate_real_weather_scenes(config: Mapping[str, Any]) -> Path:
    scenes = _scene_video_rows(config)
    data_cfg = require_mapping(config, "data")
    model_cfg = require_mapping(config, "models")
    evaluation_cfg = require_mapping(config, "evaluation")
    frames_per_scene = require_int(data_cfg, "frames_per_scene", context="data")
    start_margin_seconds = require_float(data_cfg, "start_margin_seconds", context="data")
    end_margin_seconds = require_float(data_cfg, "end_margin_seconds", context="data")
    iou_threshold = require_float(model_cfg, "iou_threshold", context="models")
    class_aware = require_bool(evaluation_cfg, "class_aware", context="evaluation")

    root = output_dir(config)
    records_dir = root / "records"
    figures_dir = root / "figures"
    records_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    backend = RealWeatherBackend(_make_backend_config(config, Path(scenes[0]["video_path"])))
    extractor = DriftSignalExtractor(config, student_model=None)
    frame_rows: list[dict[str, Any]] = []
    raw_signal_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    scene_images: dict[str, Path] = {}
    try:
        for scene_index, scene in enumerate(scenes):
            video_path = Path(scene["video_path"])
            if scene_index > 0:
                _switch_backend_video(backend, video_path)
            metadata = _video_metadata(video_path)
            sampled = _sample_frame_indices(
                frame_count=int(metadata["frames"]),
                fps=float(metadata["fps"]),
                frames_per_scene=frames_per_scene,
                start_margin_seconds=start_margin_seconds,
                end_margin_seconds=end_margin_seconds,
            )

            scene_frame_rows: list[dict[str, Any]] = []
            for frame_index in sampled:
                global_frame_id = len(frame_rows)
                frame = backend.frame(frame_index)
                student_output, teacher_output, boundary_payload = backend.infer(
                    frame
                )
                student_prediction = prediction_to_jsonable(student_output)
                teacher_prediction = prediction_to_jsonable(teacher_output)
                metrics = detection_f1(
                    student_prediction,
                    teacher_prediction,
                    iou_threshold=iou_threshold,
                    class_aware=class_aware,
                )
                matched = match_detections(
                    student_prediction,
                    teacher_prediction,
                    iou_threshold=iou_threshold,
                    class_aware=class_aware,
                )
                rendered = frame.copy()
                _draw_prediction(
                    rendered,
                    teacher_prediction,
                    color=(31, 119, 180),
                    prefix="T",
                )
                _draw_prediction(
                    rendered,
                    student_prediction,
                    color=(255, 191, 0),
                    prefix="S",
                )
                row = {
                    "global_frame_id": global_frame_id,
                    "scene_id": scene["scene_id"],
                    "scene_label": scene["scene_label"],
                    "domain": scene["scene_id"],
                    "domain_index": scene_index,
                    "video_path": str(video_path),
                    "frame_index": int(frame_index),
                    "time_seconds": float(frame_index) / max(float(metadata["fps"]), 1.0),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "student_count": int(metrics["student_count"]),
                    "teacher_count": int(metrics["teacher_count"]),
                    "student_mean_confidence": _mean_score(student_prediction),
                    "teacher_mean_confidence": _mean_score(teacher_prediction),
                    "student_prediction": student_prediction,
                    "teacher_prediction": teacher_prediction,
                    "matches": matched["matches"],
                    "rendered_frame": rendered,
                }
                signal = extractor.extract(frame, student_output, boundary_payload)
                raw_signal_rows.append(
                    {
                        "sequence_name": REAL_WEATHER_SEQUENCE,
                        "global_frame_id": global_frame_id,
                        "scene_id": scene["scene_id"],
                        "scene_label": scene["scene_label"],
                        "source_frame_id": int(frame_index),
                        "domain": scene["scene_id"],
                        "domain_index": scene_index,
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        "student_num_boxes": int(metrics["student_count"]),
                        "teacher_num_boxes": int(metrics["teacher_count"]),
                        "student_mean_confidence": _mean_score(student_prediction),
                        "teacher_mean_confidence": _mean_score(teacher_prediction),
                        **signal,
                    }
                )
                scene_frame_rows.append(row)
                frame_rows.append(row)
                prediction_rows.append(
                    {
                        "scene_id": scene["scene_id"],
                        "scene_label": scene["scene_label"],
                        "video_path": str(video_path),
                        "frame_index": int(frame_index),
                        "time_seconds": row["time_seconds"],
                        "student": student_prediction,
                        "teacher": teacher_prediction,
                        "matches": row["matches"],
                    }
                )

            total_tp = int(sum(int(row["tp"]) for row in scene_frame_rows))
            total_fp = int(sum(int(row["fp"]) for row in scene_frame_rows))
            total_fn = int(sum(int(row["fn"]) for row in scene_frame_rows))
            micro_precision = total_tp / (total_tp + total_fp + 1.0e-8)
            micro_recall = total_tp / (total_tp + total_fn + 1.0e-8)
            micro_f1 = (
                2.0
                * micro_precision
                * micro_recall
                / (micro_precision + micro_recall + 1.0e-8)
            )
            summary = {
                "scene_id": scene["scene_id"],
                "scene_label": scene["scene_label"],
                "video_path": str(video_path),
                "sampled_frames": len(scene_frame_rows),
                "duration_seconds": float(metadata["duration_seconds"]),
                "micro_precision": float(micro_precision),
                "micro_recall": float(micro_recall),
                "micro_f1": float(micro_f1),
                "mean_frame_precision": float(
                    np.mean([float(row["precision"]) for row in scene_frame_rows])
                ),
                "mean_frame_recall": float(
                    np.mean([float(row["recall"]) for row in scene_frame_rows])
                ),
                "mean_frame_f1": float(np.mean([float(row["f1"]) for row in scene_frame_rows])),
                "median_frame_f1": float(
                    np.median([float(row["f1"]) for row in scene_frame_rows])
                ),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "mean_student_boxes": float(
                    np.mean([float(row["student_count"]) for row in scene_frame_rows])
                ),
                "mean_teacher_boxes": float(
                    np.mean([float(row["teacher_count"]) for row in scene_frame_rows])
                ),
                "mean_student_confidence": float(
                    np.mean([float(row["student_mean_confidence"]) for row in scene_frame_rows])
                ),
                "mean_teacher_confidence": float(
                    np.mean([float(row["teacher_mean_confidence"]) for row in scene_frame_rows])
                ),
            }
            summary_rows.append(summary)
            scene_image = figures_dir / f"{scene['scene_id']}_student_teacher_examples.png"
            _save_scene_examples(
                scene=scene,
                rows=scene_frame_rows,
                output_path=scene_image,
                columns=2,
                tile_width=640,
                tile_height=360,
            )
            scene_images[str(scene["scene_id"])] = scene_image
    finally:
        backend.close()

    finalized_signal_rows, _baseline = finalize_signal_records(
        raw_signal_rows,
        config,
        clean_baseline_mask(raw_signal_rows),
    )
    for frame_row, signal_row in zip(frame_rows, finalized_signal_rows):
        for key in DRIFT_FRAME_FIELDS:
            if key in signal_row:
                frame_row[key] = signal_row[key]
    window_rows = _build_window_metrics(frame_rows, finalized_signal_rows, config)

    serializable_frame_rows = [
        {key: value for key, value in row.items() if key not in {"rendered_frame"}}
        for row in frame_rows
    ]
    _write_csv(records_dir / "real_weather_frame_metrics.csv", frame_rows, FRAME_FIELDS)
    _write_csv(records_dir / "frame_signals.csv", finalized_signal_rows, DRIFT_FRAME_FIELDS)
    _write_csv(records_dir / "window_metrics.csv", window_rows, WINDOW_METRIC_FIELDS)
    _write_csv(
        records_dir / "real_weather_window_metrics.csv",
        window_rows,
        WINDOW_METRIC_FIELDS,
    )
    _write_csv(records_dir / "real_weather_scene_summary.csv", summary_rows, SUMMARY_FIELDS)
    (records_dir / "real_weather_predictions.json").write_text(
        json.dumps(prediction_rows, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    (records_dir / "real_weather_frame_metrics.json").write_text(
        json.dumps(serializable_frame_rows, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    summary_plot = figures_dir / "real_weather_scene_metric_summary.png"
    _save_summary_plot(summary_rows, summary_plot)
    drift_plot = figures_dir / "real_weather_drift_detection_effectiveness.png"
    _save_drift_detection_plot(window_rows, config, drift_plot)
    analyze_signal_validity(config)
    analyze_online_triggers(config)
    write_real_weather_report(config)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate real weather videos by scene.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = evaluate_real_weather_scenes(load_config(args.config))
    print(output)


if __name__ == "__main__":
    main()
