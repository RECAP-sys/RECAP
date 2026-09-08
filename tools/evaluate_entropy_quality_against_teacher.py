from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge.sample_quality import HIGH_QUALITY, LOW_QUALITY
from edge.sample_store import EdgeSampleStore

TEACHER_VERIFIED_HIGH = "teacher_verified_high_quality"
TEACHER_VERIFIED_LOW = "teacher_verified_low_quality"


@dataclass(frozen=True)
class AgreementThresholds:
    iou_threshold: float = 0.5
    match_recall_threshold: float = 0.8
    false_positive_threshold: float = 0.2
    teacher_conf_threshold: float = 0.4
    edge_conf_threshold: float = 0.25


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in list(box_a)[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in list(box_b)[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _filtered_detection_items(
    prediction: Mapping[str, Any],
    *,
    conf_threshold: float,
) -> list[tuple[list[float], int, float]]:
    boxes = list(prediction.get("boxes") or [])
    labels = list(prediction.get("labels") or [])
    scores = list(prediction.get("scores") or [])
    if not scores:
        scores = [1.0] * len(boxes)
    items: list[tuple[list[float], int, float]] = []
    for box, label, score in zip(boxes, labels, scores):
        try:
            parsed_box = [float(value) for value in list(box)[:4]]
            parsed_label = int(label)
            parsed_score = float(score)
        except (TypeError, ValueError):
            continue
        if len(parsed_box) != 4 or parsed_score < float(conf_threshold):
            continue
        items.append((parsed_box, parsed_label, parsed_score))
    items.sort(key=lambda item: item[2], reverse=True)
    return items


def teacher_verified_quality(
    *,
    edge_prediction: Mapping[str, Any],
    teacher_prediction: Mapping[str, Any],
    thresholds: AgreementThresholds = AgreementThresholds(),
) -> tuple[str, dict[str, float | int]]:
    teacher_items = _filtered_detection_items(
        teacher_prediction,
        conf_threshold=thresholds.teacher_conf_threshold,
    )
    edge_items = _filtered_detection_items(
        edge_prediction,
        conf_threshold=thresholds.edge_conf_threshold,
    )
    matched_teacher: set[int] = set()
    matched_edge: set[int] = set()
    for edge_index, (edge_box, edge_label, _edge_score) in enumerate(edge_items):
        best_teacher_index = None
        best_iou = 0.0
        for teacher_index, (teacher_box, teacher_label, _teacher_score) in enumerate(teacher_items):
            if teacher_index in matched_teacher or int(edge_label) != int(teacher_label):
                continue
            iou = box_iou(edge_box, teacher_box)
            if iou >= thresholds.iou_threshold and iou > best_iou:
                best_teacher_index = teacher_index
                best_iou = iou
        if best_teacher_index is not None:
            matched_teacher.add(best_teacher_index)
            matched_edge.add(edge_index)

    matched_teacher_boxes = len(matched_teacher)
    unmatched_edge_boxes = max(0, len(edge_items) - len(matched_edge))
    teacher_recall = matched_teacher_boxes / float(max(len(teacher_items), 1))
    edge_fp_ratio = unmatched_edge_boxes / float(max(len(edge_items), 1))
    verified_high = (
        teacher_recall >= thresholds.match_recall_threshold
        and edge_fp_ratio <= thresholds.false_positive_threshold
    )
    return (
        TEACHER_VERIFIED_HIGH if verified_high else TEACHER_VERIFIED_LOW,
        {
            "teacher_recall": float(teacher_recall),
            "edge_fp_ratio": float(edge_fp_ratio),
            "matched_teacher_boxes": int(matched_teacher_boxes),
            "teacher_box_count": int(len(teacher_items)),
            "edge_box_count": int(len(edge_items)),
            "unmatched_edge_boxes": int(unmatched_edge_boxes),
        },
    )


def evaluate_agreement(
    samples: Sequence[Mapping[str, Any]],
    *,
    thresholds: AgreementThresholds = AgreementThresholds(),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    confusion = {
        HIGH_QUALITY: {TEACHER_VERIFIED_HIGH: 0, TEACHER_VERIFIED_LOW: 0},
        LOW_QUALITY: {TEACHER_VERIFIED_HIGH: 0, TEACHER_VERIFIED_LOW: 0},
    }
    for sample in samples:
        predicted_quality = str(
            sample.get("predicted_quality") or sample.get("quality") or LOW_QUALITY
        )
        if predicted_quality not in {HIGH_QUALITY, LOW_QUALITY}:
            predicted_quality = LOW_QUALITY
        verified_quality, match_stats = teacher_verified_quality(
            edge_prediction=dict(sample.get("edge_prediction") or {}),
            teacher_prediction=dict(sample.get("teacher_prediction") or {}),
            thresholds=thresholds,
        )
        confusion[predicted_quality][verified_quality] += 1
        rows.append(
            {
                "sample_id": str(sample.get("sample_id") or ""),
                "predicted_quality": predicted_quality,
                "teacher_verified_quality": verified_quality,
                **match_stats,
            }
        )

    total = len(rows)
    predicted_high = (
        confusion[HIGH_QUALITY][TEACHER_VERIFIED_HIGH]
        + confusion[HIGH_QUALITY][TEACHER_VERIFIED_LOW]
    )
    predicted_low = (
        confusion[LOW_QUALITY][TEACHER_VERIFIED_HIGH] + confusion[LOW_QUALITY][TEACHER_VERIFIED_LOW]
    )
    teacher_high = (
        confusion[HIGH_QUALITY][TEACHER_VERIFIED_HIGH]
        + confusion[LOW_QUALITY][TEACHER_VERIFIED_HIGH]
    )
    teacher_low = (
        confusion[HIGH_QUALITY][TEACHER_VERIFIED_LOW] + confusion[LOW_QUALITY][TEACHER_VERIFIED_LOW]
    )
    true_high = confusion[HIGH_QUALITY][TEACHER_VERIFIED_HIGH]
    false_trusted = confusion[HIGH_QUALITY][TEACHER_VERIFIED_LOW]
    true_low = confusion[LOW_QUALITY][TEACHER_VERIFIED_LOW]

    return {
        "total_samples": total,
        "predicted_high_quality_count": predicted_high,
        "predicted_low_quality_count": predicted_low,
        "teacher_verified_high_quality_count": teacher_high,
        "teacher_verified_low_quality_count": teacher_low,
        "high_quality_precision": true_high / float(max(predicted_high, 1)),
        "false_trusted_rate": false_trusted / float(max(predicted_high, 1)),
        "low_quality_recall": true_low / float(max(teacher_low, 1)),
        "low_quality_precision": true_low / float(max(predicted_low, 1)),
        "teacher_load": predicted_low / float(max(total, 1)),
        "confusion_matrix": confusion,
        "samples": rows,
    }


def _load_teacher_labels(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "samples" in payload:
        payload = payload["samples"]
    if isinstance(payload, Mapping):
        return {
            str(sample_id): dict(labels)
            for sample_id, labels in payload.items()
            if isinstance(labels, Mapping)
        }
    result: dict[str, dict[str, Any]] = {}
    for item in list(payload or []):
        if not isinstance(item, Mapping):
            continue
        sample_id = str(item.get("sample_id") or "")
        labels = item.get("labels", item)
        if sample_id and isinstance(labels, Mapping):
            result[sample_id] = dict(labels)
    return result


def _quality_from_record(record: object) -> str:
    quality_payload = getattr(record, "quality", None)
    if isinstance(quality_payload, Mapping):
        quality = str(quality_payload.get("quality") or "")
        if quality in {HIGH_QUALITY, LOW_QUALITY}:
            return quality
    quality = str(getattr(record, "quality_bucket", LOW_QUALITY))
    return quality if quality in {HIGH_QUALITY, LOW_QUALITY} else LOW_QUALITY


def load_samples_from_store(
    sample_store_root: Path,
    *,
    teacher_labels: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    store = EdgeSampleStore(str(sample_store_root))
    samples: list[dict[str, Any]] = []
    for record in store.list_records():
        sample_id = str(record.sample_id)
        teacher_prediction = teacher_labels.get(sample_id)
        if teacher_prediction is None:
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "predicted_quality": _quality_from_record(record),
                "edge_prediction": store.load_inference_result(record),
                "teacher_prediction": dict(teacher_prediction),
            }
        )
    return samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate entropy quality buckets against offline teacher labels."
    )
    parser.add_argument("--sample-store", required=True, help="Edge sample store root directory.")
    parser.add_argument(
        "--teacher-labels-json",
        required=True,
        help="JSON mapping sample_id to teacher labels or a list of sample label entries.",
    )
    parser.add_argument("--output-json", default="", help="Optional report JSON path.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--match-recall-threshold", type=float, default=0.8)
    parser.add_argument("--false-positive-threshold", type=float, default=0.2)
    parser.add_argument("--teacher-conf-threshold", type=float, default=0.4)
    parser.add_argument("--edge-conf-threshold", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = AgreementThresholds(
        iou_threshold=float(args.iou_threshold),
        match_recall_threshold=float(args.match_recall_threshold),
        false_positive_threshold=float(args.false_positive_threshold),
        teacher_conf_threshold=float(args.teacher_conf_threshold),
        edge_conf_threshold=float(args.edge_conf_threshold),
    )
    teacher_labels = _load_teacher_labels(Path(args.teacher_labels_json))
    samples = load_samples_from_store(
        Path(args.sample_store),
        teacher_labels=teacher_labels,
    )
    report = evaluate_agreement(samples, thresholds=thresholds)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
