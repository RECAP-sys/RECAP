from __future__ import annotations

import pandas as pd

from model_management.model_info import annotation_cols


def load_annotations(annotation_path: str) -> pd.DataFrame:
    return pd.read_csv(annotation_path, header=None, names=annotation_cols)


def extract_boxes_and_labels(rows: pd.DataFrame) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for row in rows.itertuples(index=False):
        label = row.target_id
        if label == 0:
            continue
        boxes.append([row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2])
        labels.append(label)
    return boxes, labels


def build_detection_target(rows: pd.DataFrame) -> dict[str, list[object]]:
    boxes, labels = extract_boxes_and_labels(rows)
    return {
        "boxes": boxes,
        "labels": labels,
    }


def load_annotation_targets(annotation_path: str) -> dict[str, dict[str, object]]:
    targets: dict[str, dict[str, object]] = {}
    try:
        annotations = load_annotations(annotation_path)
    except FileNotFoundError:
        return targets

    for frame_index, rows in annotations.groupby("frame_index"):
        target = build_detection_target(rows)
        if target["boxes"] and target["labels"]:
            targets[str(int(frame_index))] = target
    return targets
