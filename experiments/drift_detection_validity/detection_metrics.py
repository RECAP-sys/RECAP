#!/usr/bin/env python3
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

NormalizedPrediction = dict[str, np.ndarray]


def _to_numpy(value: Any, *, dtype: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _read_prediction_value(prediction: Any, keys: Sequence[str]) -> Any:
    if isinstance(prediction, Mapping):
        for key in keys:
            if key in prediction:
                return prediction.get(key)
        return None
    for key in keys:
        if hasattr(prediction, key):
            return getattr(prediction, key)
    return None


def normalize_prediction(prediction: Any) -> NormalizedPrediction:
    boxes = _to_numpy(
        _read_prediction_value(
            prediction,
            ("boxes", "detection_boxes", "final_detection_boxes", "pred_boxes"),
        ),
        dtype=np.float64,
    )
    if boxes.size == 0:
        boxes = np.zeros((0, 4), dtype=np.float64)
    boxes = np.reshape(boxes, (-1, 4))[:, :4] if boxes.size else np.zeros((0, 4))
    count = int(boxes.shape[0])

    scores = _to_numpy(
        _read_prediction_value(
            prediction,
            (
                "scores",
                "detection_scores",
                "detection_score",
                "final_detection_scores",
                "pred_scores",
            ),
        ),
        dtype=np.float64,
    ).reshape(-1)
    if scores.size < count:
        scores = np.concatenate([scores, np.ones(count - scores.size, dtype=np.float64)])
    scores = scores[:count]

    labels = _to_numpy(
        _read_prediction_value(
            prediction,
            (
                "labels",
                "detection_class",
                "classes",
                "final_detection_labels",
                "final_detection_classes",
                "pred_labels",
            ),
        ),
        dtype=np.int64,
    ).reshape(-1)
    if labels.size < count:
        labels = np.concatenate([labels, np.zeros(count - labels.size, dtype=np.int64)])
    labels = labels[:count]

    return {
        "boxes": boxes.astype(np.float64, copy=False),
        "scores": scores.astype(np.float64, copy=False),
        "labels": labels.astype(np.int64, copy=False),
    }


def prediction_to_jsonable(prediction: Any) -> dict[str, list[Any]]:
    normalized = normalize_prediction(prediction)
    return {
        "boxes": normalized["boxes"].astype(float).tolist(),
        "scores": normalized["scores"].astype(float).tolist(),
        "labels": normalized["labels"].astype(int).tolist(),
    }


def box_iou_xyxy(boxes1: Any, boxes2: Any) -> np.ndarray:
    first = _to_numpy(boxes1, dtype=np.float64)
    second = _to_numpy(boxes2, dtype=np.float64)
    first = first.reshape((-1, 4))[:, :4] if first.size else np.zeros((0, 4))
    second = second.reshape((-1, 4))[:, :4] if second.size else np.zeros((0, 4))
    if first.size == 0 or second.size == 0:
        return np.zeros((first.shape[0], second.shape[0]), dtype=np.float64)

    x1 = np.maximum(first[:, None, 0], second[None, :, 0])
    y1 = np.maximum(first[:, None, 1], second[None, :, 1])
    x2 = np.minimum(first[:, None, 2], second[None, :, 2])
    y2 = np.minimum(first[:, None, 3], second[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area1 = np.maximum(0.0, first[:, 2] - first[:, 0]) * np.maximum(
        0.0, first[:, 3] - first[:, 1]
    )
    area2 = np.maximum(0.0, second[:, 2] - second[:, 0]) * np.maximum(
        0.0, second[:, 3] - second[:, 1]
    )
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def match_detections(
    student_preds: Any,
    teacher_preds: Any,
    iou_threshold: float,
    class_aware: bool = True,
) -> dict[str, Any]:
    student = normalize_prediction(student_preds)
    teacher = normalize_prediction(teacher_preds)
    student_count = int(student["boxes"].shape[0])
    teacher_count = int(teacher["boxes"].shape[0])
    if student_count == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": teacher_count,
            "matches": [],
            "student_count": student_count,
            "teacher_count": teacher_count,
        }

    ious = box_iou_xyxy(student["boxes"], teacher["boxes"])
    matched_teacher: set[int] = set()
    matches: list[dict[str, Any]] = []
    tp = 0
    order = np.argsort(-student["scores"])
    for student_index in order.tolist():
        best_teacher = -1
        best_iou = 0.0
        for teacher_index in range(teacher_count):
            if teacher_index in matched_teacher:
                continue
            if class_aware and int(student["labels"][student_index]) != int(
                teacher["labels"][teacher_index]
            ):
                continue
            iou = float(ious[student_index, teacher_index]) if teacher_count else 0.0
            if iou > best_iou:
                best_iou = iou
                best_teacher = teacher_index
        if best_teacher >= 0 and best_iou >= float(iou_threshold):
            matched_teacher.add(best_teacher)
            tp += 1
            matches.append(
                {
                    "student_index": int(student_index),
                    "teacher_index": int(best_teacher),
                    "iou": float(best_iou),
                }
            )
    fp = student_count - tp
    fn = teacher_count - tp
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "matches": matches,
        "student_count": student_count,
        "teacher_count": teacher_count,
    }


def _score_from_counts(tp: int, fp: int, fn: int, eps: float) -> dict[str, float]:
    precision = float(tp) / (float(tp + fp) + float(eps))
    recall = float(tp) / (float(tp + fn) + float(eps))
    f1 = (2.0 * precision * recall) / (precision + recall + float(eps))
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def detection_f1(
    student_preds: Any,
    teacher_preds: Any,
    iou_threshold: float,
    class_aware: bool = True,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    matched = match_detections(
        student_preds,
        teacher_preds,
        iou_threshold=iou_threshold,
        class_aware=class_aware,
    )
    scores = _score_from_counts(
        int(matched["tp"]),
        int(matched["fp"]),
        int(matched["fn"]),
        eps=float(eps),
    )
    return {
        **scores,
        "tp": float(matched["tp"]),
        "fp": float(matched["fp"]),
        "fn": float(matched["fn"]),
        "student_count": float(matched["student_count"]),
        "teacher_count": float(matched["teacher_count"]),
    }


def _frame_id(record: Mapping[str, Any], default_frame_id: int) -> int:
    for key in ("global_frame_id", "frame_id", "source_frame_id"):
        if key in record:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                pass
    return int(default_frame_id)


def _majority_domain(records: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for record in records:
        domain = str(record.get("domain", ""))
        counts[domain] = counts.get(domain, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def window_detection_f1(
    student_records: Sequence[Mapping[str, Any]],
    teacher_records: Sequence[Mapping[str, Any]],
    window_size: int,
    stride: int,
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
    eps: float = 1.0e-8,
) -> list[dict[str, Any]]:
    if len(student_records) != len(teacher_records):
        raise ValueError("student_records and teacher_records must have the same length")
    size = max(1, int(window_size))
    step = max(1, int(stride))
    windows: list[dict[str, Any]] = []
    window_id = 0
    for start in range(0, max(0, len(student_records) - size + 1), step):
        end = start + size
        tp = fp = fn = 0
        for student, teacher in zip(student_records[start:end], teacher_records[start:end]):
            matched = match_detections(
                student.get("prediction", student),
                teacher.get("prediction", teacher),
                iou_threshold=iou_threshold,
                class_aware=class_aware,
            )
            tp += int(matched["tp"])
            fp += int(matched["fp"])
            fn += int(matched["fn"])
        scores = _score_from_counts(tp, fp, fn, eps=float(eps))
        frame_start = _frame_id(student_records[start], start)
        frame_end = _frame_id(student_records[end - 1], end - 1)
        windows.append(
            {
                "window_id": window_id,
                "window_start_frame": frame_start,
                "window_end_frame": frame_end,
                "domain_majority": _majority_domain(student_records[start:end]),
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
        window_id += 1
    if not windows and student_records:
        # Very small smoke/unit-test inputs still deserve one partial window.
        tp = fp = fn = 0
        for student, teacher in zip(student_records, teacher_records):
            matched = match_detections(
                student.get("prediction", student),
                teacher.get("prediction", teacher),
                iou_threshold=iou_threshold,
                class_aware=class_aware,
            )
            tp += int(matched["tp"])
            fp += int(matched["fp"])
            fn += int(matched["fn"])
        scores = _score_from_counts(tp, fp, fn, eps=float(eps))
        windows.append(
            {
                "window_id": 0,
                "window_start_frame": _frame_id(student_records[0], 0),
                "window_end_frame": _frame_id(student_records[-1], len(student_records) - 1),
                "domain_majority": _majority_domain(student_records),
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    for row in windows:
        for key in ("precision", "recall", "f1"):
            if not math.isfinite(float(row[key])):
                row[key] = 0.0
    return windows


__all__ = [
    "box_iou_xyxy",
    "detection_f1",
    "match_detections",
    "normalize_prediction",
    "prediction_to_jsonable",
    "window_detection_f1",
]
