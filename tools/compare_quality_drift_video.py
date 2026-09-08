from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.runtime import load_runtime_config
from edge.sample_quality import EntropyQualityClassifier
from edge.window_drift_detector import WindowDriftDetector
from model_management.fixed_split import SplitConstraints, load_or_compute_fixed_split_plan
from model_management.object_detection import Object_Detection
from model_management.universal_model_split import UniversalModelSplitter


def _boxes_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_fit_metrics(
    teacher: dict[str, Any],
    prediction: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    gt_boxes = list(teacher.get("boxes") or [])
    gt_labels = list(teacher.get("labels") or [])
    pred_boxes = list(prediction.get("boxes") or [])
    pred_labels = list(prediction.get("labels") or [])
    pred_scores = list(prediction.get("scores") or [])

    order = sorted(
        range(len(pred_scores)),
        key=lambda index: float(pred_scores[index]),
        reverse=True,
    )
    matched_gt: set[int] = set()
    true_positive = 0

    for pred_index in order:
        best_match = None
        best_iou = 0.0
        for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_index in matched_gt:
                continue
            if int(pred_labels[pred_index]) != int(gt_label):
                continue
            iou = _boxes_iou(np.asarray(pred_boxes[pred_index]), np.asarray(gt_box))
            if iou >= iou_threshold and iou > best_iou:
                best_match = gt_index
                best_iou = iou
        if best_match is not None:
            matched_gt.add(best_match)
            true_positive += 1

    false_positive = max(0, len(pred_boxes) - true_positive)
    false_negative = max(0, len(gt_boxes) - true_positive)
    precision = true_positive / float(max(1, true_positive + false_positive))
    recall = true_positive / float(max(1, true_positive + false_negative))
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "teacher_count": int(len(gt_boxes)),
        "prediction_count": int(len(pred_boxes)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        avg_rank = 0.5 * (index + end) + 1.0
        ranks[order[index : end + 1]] = avg_rank
        index = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return None
    ranked_x = _rankdata(x_arr[mask])
    ranked_y = _rankdata(y_arr[mask])
    if float(ranked_x.std()) <= 0.0 or float(ranked_y.std()) <= 0.0:
        return None
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def _read_video_frames(video_path: Path, *, max_frames: int) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frames: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append({"frame_index": frame_index, "frame": frame})
            frame_index += 1
    finally:
        capture.release()
    return frames


def _collect_teacher_targets(
    frames: list[dict[str, Any]],
    teacher: Object_Detection,
) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for item in frames:
        boxes, labels, scores = teacher.large_inference(item["frame"])
        targets[item["frame_index"]] = {
            "boxes": boxes or [],
            "labels": labels or [],
            "scores": scores or [],
        }
    return targets


def _init_splitter(detector: Object_Detection, cfg, first_frame: np.ndarray, plan_path: Path):
    splitter = UniversalModelSplitter(device="cpu")
    sample_input = detector.prepare_splitter_input(first_frame)
    split_model = detector.get_split_runtime_model()
    splitter.trace(split_model, sample_input)
    constraints = SplitConstraints.from_config(getattr(cfg.split_learning, "fixed_split", None))
    plan = load_or_compute_fixed_split_plan(
        split_model,
        constraints,
        sample_input=sample_input,
        device="cpu",
        model_name=detector.model_name,
        cache_path=str(plan_path),
        splitter=splitter,
    )
    splitter.split(
        candidate_id=plan.candidate_id, boundary_tensor_labels=plan.boundary_tensor_labels
    )
    return splitter, plan


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "fit_f1": _mean(rows, "fit_f1"),
        "confidence": _mean(rows, "confidence"),
        "logit_entropy": _mean(rows, "logit_entropy"),
        "logit_margin": _mean(rows, "logit_margin"),
        "logit_energy": _mean(rows, "logit_energy"),
        "prediction_count": _mean(rows, "prediction_count"),
        "teacher_count": _mean(rows, "teacher_count"),
    }


def _bottom_quartile_threshold(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, 0.25))


def _flag_metrics(
    rows: list[dict[str, Any]],
    *,
    flag_key: str,
    poor_fit_threshold: float,
) -> dict[str, Any]:
    positives = [row for row in rows if bool(row[flag_key])]
    poor_fit = [row for row in rows if float(row["fit_f1"]) <= poor_fit_threshold]
    tp = sum(
        1 for row in rows if bool(row[flag_key]) and float(row["fit_f1"]) <= poor_fit_threshold
    )
    fp = sum(1 for row in rows if bool(row[flag_key]) and float(row["fit_f1"]) > poor_fit_threshold)
    fn = sum(
        1
        for row in rows
        if (not bool(row[flag_key])) and float(row["fit_f1"]) <= poor_fit_threshold
    )
    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    return {
        "flag_count": len(positives),
        "poor_fit_count": len(poor_fit),
        "precision": float(precision),
        "recall": float(recall),
        "mean_fit_flagged": _mean(positives, "fit_f1"),
        "mean_fit_unflagged": _mean([row for row in rows if not bool(row[flag_key])], "fit_f1"),
    }


def _summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fit_scores = [float(row["fit_f1"]) for row in rows]
    poor_fit_threshold = _bottom_quartile_threshold(fit_scores)
    reasons = Counter(reason for row in rows for reason in row.get("quality_reasons", []))

    quality_bucket_summary = {
        bucket: _subset_summary([row for row in rows if row["quality_bucket"] == bucket])
        for bucket in ("high_quality", "low_quality")
    }
    drift_rows = [row for row in rows if row["drift_detected"]]

    return {
        "frame_count": len(rows),
        "overall": _subset_summary(rows),
        "quality_buckets": quality_bucket_summary,
        "drift": {
            "count": len(drift_rows),
            "frames": [int(row["frame_index"]) for row in drift_rows],
            "summary": _subset_summary(drift_rows),
            "non_drift_summary": _subset_summary(
                [row for row in rows if not row["drift_detected"]]
            ),
        },
        "low_quality_alignment": _flag_metrics(
            rows,
            flag_key="is_low_quality",
            poor_fit_threshold=poor_fit_threshold,
        ),
        "drift_alignment": _flag_metrics(
            rows,
            flag_key="drift_detected",
            poor_fit_threshold=poor_fit_threshold,
        ),
        "poor_fit_threshold": poor_fit_threshold,
        "quality_reason_counts": dict(reasons.most_common()),
    }


def run_video_compare(
    *,
    config_path: Path,
    video_path: Path,
    models: list[str],
    max_frames: int,
    report_root: Path,
) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    config = load_runtime_config(config_path)
    frames = _read_video_frames(video_path, max_frames=max_frames)
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    teacher = Object_Detection(config.server, "large inference")
    teacher_targets = _collect_teacher_targets(frames, teacher)

    report: dict[str, Any] = {
        "video_path": str(video_path.resolve()),
        "frame_count": len(frames),
        "models": {},
    }
    frame_cache = {int(item["frame_index"]): item["frame"] for item in frames}

    for model_name in models:
        client_cfg = load_runtime_config(config_path).client
        client_cfg.lightweight = model_name
        detector = Object_Detection(client_cfg, "small inference")
        quality_cfg = getattr(client_cfg, "sample_quality", None)
        quality_classifier = EntropyQualityClassifier.from_config(quality_cfg)
        drift_cfg = getattr(client_cfg, "window_drift", None)
        drift_detector = WindowDriftDetector(
            window_size=int(getattr(drift_cfg, "window_size", 100)),
            min_window_size=int(getattr(drift_cfg, "min_window_size", 30)),
            low_quality_rate_threshold=float(getattr(drift_cfg, "low_quality_rate_threshold", 0.3)),
            persistence_windows=int(getattr(drift_cfg, "persistence_windows", 3)),
        )
        plan_path = report_root / f"{model_name}_fixed_split_plan.json"
        splitter, plan = _init_splitter(detector, client_cfg, frames[0]["frame"], plan_path)
        runtime_contract = dict(getattr(plan, "runtime_contract", {}) or {})
        split_key = str(
            getattr(plan, "canonical_split_key", "")
            or runtime_contract.get("logical_split_id")
            or plan.split_config_id
        )
        feature_abi_id = str(
            runtime_contract.get("feature_abi_id")
            or runtime_contract.get("feature_layout_id")
            or ""
        )

        rows: list[dict[str, Any]] = []
        for item in frames:
            frame_index = int(item["frame_index"])
            frame = frame_cache[frame_index]
            artifacts = detector.infer_sample(frame, splitter=splitter)
            quality = quality_classifier.classify(
                artifacts,
                artifacts.intermediate,
                model_name=model_name,
                split_key=split_key,
                feature_abi_id=feature_abi_id,
            )
            drift_state = drift_detector.update(quality)
            fit = _match_fit_metrics(
                teacher_targets[frame_index],
                artifacts.to_inference_result(),
            )
            rows.append(
                {
                    "frame_index": frame_index,
                    "fit_f1": fit["f1"],
                    "fit_precision": fit["precision"],
                    "fit_recall": fit["recall"],
                    "teacher_count": fit["teacher_count"],
                    "prediction_count": fit["prediction_count"],
                    "confidence": float(artifacts.confidence),
                    "proposal_count": int(artifacts.proposal_count or 0),
                    "retained_count": int(artifacts.retained_count or 0),
                    "feature_spectral_entropy": artifacts.feature_spectral_entropy,
                    "logit_entropy": artifacts.logit_entropy,
                    "logit_margin": artifacts.logit_margin,
                    "logit_energy": artifacts.logit_energy,
                    "quality_bucket": str(quality.quality_bucket),
                    "quality_reasons": [quality.reason],
                    "drift_detected": bool(drift_state.drift_detected),
                    "is_low_quality": str(quality.quality_bucket) == "low_quality",
                }
            )

        report["models"][model_name] = {
            "split_plan": plan.to_dict(),
            "summary": _summarize_model(rows),
            "records": rows,
        }

    with (report_root / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare quality/drift triggers across edge models on a real video."
    )
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--video", type=Path, default=Path("video_data/road.mp4"))
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo26n", "tinynext_s", "rfdetr_nano"],
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("cache/tmp_debug/quality_drift_video_compare"),
    )
    args = parser.parse_args()
    report = run_video_compare(
        config_path=args.config,
        video_path=args.video,
        models=list(args.models),
        max_frames=int(args.max_frames),
        report_root=args.report_root,
    )
    print(
        json.dumps(
            {
                "frame_count": report["frame_count"],
                "models": list(report["models"].keys()),
                "report_path": str((args.report_root / "report.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
