from __future__ import annotations

import torch
from torchvision.ops import box_iou

FINAL_DUPLICATE_SUPPRESSION_THRESHOLDS = {
    "tinynext": {
        "same_label": (0.75, 0.9),
        "cross_label": (0.75, 0.9),
    },
    "rfdetr": {
        "same_label": (0.35, 0.75),
        "cross_label": (0.5, 0.8),
    },
}


def resolve_final_dedup_thresholds(
    model_family: str,
    threshold: float,
    *,
    threshold_high: float,
) -> dict[str, tuple[float, float]] | None:
    thresholds = FINAL_DUPLICATE_SUPPRESSION_THRESHOLDS.get(model_family)
    if thresholds is None:
        return None
    if float(threshold) < float(threshold_high) - 1e-6:
        return None
    return {str(key): (float(value[0]), float(value[1])) for key, value in thresholds.items()}


def compute_intersection_over_min_area(
    candidate_box: torch.Tensor,
    reference_boxes: torch.Tensor,
) -> torch.Tensor:
    if reference_boxes.numel() == 0:
        return reference_boxes.new_zeros((0,), dtype=torch.float32)

    inter_x1 = torch.maximum(candidate_box[0], reference_boxes[:, 0])
    inter_y1 = torch.maximum(candidate_box[1], reference_boxes[:, 1])
    inter_x2 = torch.minimum(candidate_box[2], reference_boxes[:, 2])
    inter_y2 = torch.minimum(candidate_box[3], reference_boxes[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
    intersection = inter_w * inter_h

    candidate_area = (candidate_box[2] - candidate_box[0]).clamp_min(0.0) * (
        candidate_box[3] - candidate_box[1]
    ).clamp_min(0.0)
    reference_areas = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp_min(0.0) * (
        reference_boxes[:, 3] - reference_boxes[:, 1]
    ).clamp_min(0.0)
    min_area = torch.minimum(
        reference_areas,
        reference_areas.new_full(reference_areas.shape, float(candidate_area.item())),
    ).clamp_min(1e-6)
    return intersection / min_area


def deduplicate_final_predictions(
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
    *,
    thresholds: dict[str, tuple[float, float]] | None,
) -> tuple[list[list[float]], list[int], list[float]]:
    if thresholds is None or len(scores) <= 1:
        return boxes, labels, scores
    same_label_thresholds = thresholds["same_label"]
    cross_label_thresholds = thresholds["cross_label"]

    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)

    valid_geometry = (boxes_tensor[:, 2] > boxes_tensor[:, 0]) & (
        boxes_tensor[:, 3] > boxes_tensor[:, 1]
    )
    if not torch.any(valid_geometry):
        return [], [], []

    boxes_tensor = boxes_tensor[valid_geometry]
    labels_tensor = labels_tensor[valid_geometry]
    scores_tensor = scores_tensor[valid_geometry]

    score_order = torch.argsort(scores_tensor, descending=True)
    keep_indices: list[int] = []
    for index in score_order.tolist():
        candidate_box = boxes_tensor[index]
        if keep_indices:
            kept_boxes = boxes_tensor[keep_indices]
            kept_labels = labels_tensor[keep_indices]
            candidate_iou = box_iou(candidate_box.unsqueeze(0), kept_boxes).squeeze(0)
            containment = compute_intersection_over_min_area(candidate_box, kept_boxes)
            same_label_mask = kept_labels == labels_tensor[index]
            cross_label_mask = ~same_label_mask
            suppressed_by_iou = False
            suppressed_by_containment = False
            if bool(torch.any(same_label_mask)):
                same_label_iou, same_label_containment = same_label_thresholds
                suppressed_by_iou = suppressed_by_iou or bool(
                    torch.any(candidate_iou[same_label_mask] >= same_label_iou)
                )
                suppressed_by_containment = suppressed_by_containment or bool(
                    torch.any(containment[same_label_mask] >= same_label_containment)
                )
            if bool(torch.any(cross_label_mask)):
                cross_label_iou, cross_label_containment = cross_label_thresholds
                suppressed_by_iou = suppressed_by_iou or bool(
                    torch.any(candidate_iou[cross_label_mask] >= cross_label_iou)
                )
                suppressed_by_containment = suppressed_by_containment or bool(
                    torch.any(containment[cross_label_mask] >= cross_label_containment)
                )
            if suppressed_by_iou or suppressed_by_containment:
                continue
        keep_indices.append(index)

    if not keep_indices:
        return [], [], []

    keep = torch.as_tensor(keep_indices, dtype=torch.int64)
    return (
        boxes_tensor.index_select(0, keep).tolist(),
        labels_tensor.index_select(0, keep).tolist(),
        scores_tensor.index_select(0, keep).tolist(),
    )
