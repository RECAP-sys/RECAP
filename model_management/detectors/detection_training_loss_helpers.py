from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any

import torch

from model_management.detection_box_projection import (
    ORIGINAL_XYXY,
    project_original_xyxy_to_model_input_xyxy,
    require_coordinate_metadata,
)
from model_management.detectors.legacy_model_zoo import COCO_80_TO_91

try:
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors
except Exception:
    build_criterion_and_postprocessors = None


COCO_91_TO_80 = {label: idx for idx, label in enumerate(COCO_80_TO_91)}
_RFDETR_PACKED_AUX_OUTPUTS_MARKER = "__plank_rfdetr_packed_aux_outputs__"


def _extract_detr_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
        return outputs.logits, outputs.pred_boxes
    if isinstance(outputs, dict):
        logits = outputs.get("logits", outputs.get("pred_logits"))
        pred_boxes = outputs.get("pred_boxes")
        if isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor):
            return logits, pred_boxes
    tensors = list(_iter_tensors(outputs))
    logits = next((tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] > 4), None)
    pred_boxes = next(
        (tensor for tensor in tensors if tensor.ndim == 3 and tensor.shape[-1] == 4), None
    )
    if logits is None or pred_boxes is None:
        raise RuntimeError("Unable to extract DETR logits/pred_boxes from model output.")
    return logits, pred_boxes


def _extract_rfdetr_outputs(outputs: Any) -> dict[str, Any]:
    if isinstance(outputs, dict):
        logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        if isinstance(logits, torch.Tensor) and isinstance(pred_boxes, torch.Tensor):
            extracted = {
                "pred_logits": _contiguous_tensor_tree(logits),
                "pred_boxes": _contiguous_tensor_tree(pred_boxes),
            }
            if isinstance(outputs.get("aux_outputs"), (list, tuple, dict)):
                extracted["aux_outputs"] = _unpack_rfdetr_aux_outputs(outputs["aux_outputs"])
            if isinstance(outputs.get("enc_outputs"), dict):
                extracted["enc_outputs"] = _contiguous_tensor_tree(outputs["enc_outputs"])
            return extracted
    logits, pred_boxes = _extract_detr_outputs(outputs)
    return {
        "pred_logits": _contiguous_tensor_tree(logits),
        "pred_boxes": _contiguous_tensor_tree(pred_boxes),
    }


def _is_rfdetr_packed_aux_marker(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        try:
            return bool(value.detach().bool().cpu().item())
        except Exception:
            return False
    return False


def _unpack_rfdetr_aux_outputs(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return _contiguous_tensor_tree(list(value))
    if not isinstance(value, dict):
        return _contiguous_tensor_tree(value)
    if not _is_rfdetr_packed_aux_marker(value.get(_RFDETR_PACKED_AUX_OUTPUTS_MARKER)):
        return _contiguous_tensor_tree(value)
    tensor_items = {
        str(key): tensor
        for key, tensor in value.items()
        if key != _RFDETR_PACKED_AUX_OUTPUTS_MARKER
        and isinstance(tensor, torch.Tensor)
        and tensor.ndim > 0
    }
    if not tensor_items:
        return []
    layer_count = min(int(tensor.shape[0]) for tensor in tensor_items.values())
    if layer_count <= 0:
        return []
    return [
        {key: _contiguous_tensor_tree(tensor[layer_index]) for key, tensor in tensor_items.items()}
        for layer_index in range(layer_count)
    ]


def _extract_rtdetr_loss_outputs(
    outputs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 2 and isinstance(outputs[1], (list, tuple)) and len(outputs[1]) == 5:
            candidate = outputs[1]
            if all(isinstance(item, (torch.Tensor, dict, type(None))) for item in candidate):
                return tuple(candidate)
        if len(outputs) == 5 and all(
            isinstance(item, (torch.Tensor, dict, type(None))) for item in outputs
        ):
            return tuple(outputs)
        for item in outputs:
            if isinstance(item, (list, tuple)):
                try:
                    return _extract_rtdetr_loss_outputs(item)
                except RuntimeError:
                    continue
    raise RuntimeError("Unable to extract RT-DETR decoder outputs from model output.")


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            yield from _iter_tensors(getattr(value, field_info.name))


def _contiguous_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value if value.is_contiguous() else value.contiguous()
    if isinstance(value, dict):
        return {key: _contiguous_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_contiguous_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_contiguous_tensor_tree(item) for item in value]
    return value


def _ensure_ultralytics_loss_args(core_model: torch.nn.Module) -> None:
    defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
    args = getattr(core_model, "args", None)
    if isinstance(args, dict):
        merged = dict(args)
        for key, value in defaults.items():
            merged.setdefault(key, value)
        core_model.args = SimpleNamespace(**merged)
        return
    if args is None:
        core_model.args = SimpleNamespace(**defaults)
        return
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _infer_original_and_model_input_image_sizes(
    targets: Any,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(targets, dict):
        raise RuntimeError("Training targets must be a dict for wrapper-model loss computation.")
    meta = targets.get("_training_meta", {})
    original_image_size, model_input_size, _resize_mode = require_coordinate_metadata(meta)
    return original_image_size, model_input_size


def _infer_resize_mode(targets: Any) -> str:
    if not isinstance(targets, dict):
        raise RuntimeError("Training targets must be a dict for wrapper-model loss computation.")
    _original_image_size, _model_input_size, resize_mode = require_coordinate_metadata(
        targets.get("_training_meta", {})
    )
    return resize_mode


def _assert_original_xyxy_targets(targets: dict[str, Any]) -> None:
    coordinate_space = str(targets.get("label_coordinate_space") or "").strip()
    has_targets = _has_target_values(targets.get("boxes")) or _has_target_values(
        targets.get("labels")
    )
    if coordinate_space != ORIGINAL_XYXY and (coordinate_space or has_targets):
        raise RuntimeError(
            "Training targets must use original_xyxy canonical labels before "
            "model-specific loss conversion."
        )


def _has_target_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return int(value.numel()) > 0
    try:
        return len(value) > 0
    except TypeError:
        return True


def _clamp_xyxy_boxes(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp_(0.0, float(width))
    boxes[..., 1::2] = boxes[..., 1::2].clamp_(0.0, float(height))
    return boxes


def _as_boxes_tensor(boxes: Any, *, device: torch.device) -> torch.Tensor:
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)
    tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4)
    return tensor.reshape(-1, 4)


def _as_labels_tensor(labels: Any, *, device: torch.device) -> torch.Tensor:
    if labels is None:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    tensor = torch.as_tensor(labels, dtype=torch.int64, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0)
    return tensor.reshape(-1)


def _xyxy_to_normalized_cxcywh(
    boxes_xyxy: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    height, width = image_size
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    return torch.stack(
        (
            ((x1 + x2) * 0.5) / float(width),
            ((y1 + y2) * 0.5) / float(height),
            (x2 - x1) / float(width),
            (y2 - y1) / float(height),
        ),
        dim=-1,
    ).clamp_(0.0, 1.0)


def _project_boxes_to_model_input(
    boxes_xyxy: torch.Tensor,
    *,
    original_image_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str = "letterbox",
) -> torch.Tensor:
    projected = project_original_xyxy_to_model_input_xyxy(
        boxes_xyxy,
        original_image_size,
        model_input_size,
        resize_mode,
    )
    if not isinstance(projected, torch.Tensor):
        projected = torch.as_tensor(projected, dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    return projected.to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype).reshape(-1, 4)


def _prepare_coco80_targets(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_classes: int = 80,
    label_schema: str = "coco_91",
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    original_image_size, model_input_size = _infer_original_and_model_input_image_sizes(targets)
    resize_mode = _infer_resize_mode(targets)
    _assert_original_xyxy_targets(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    mapped_labels: list[int] = []
    keep_indices: list[int] = []
    is_zero_based = str(label_schema).strip().lower() == "zero_based"
    max_label = max(int(num_classes), 0)
    for index, label in enumerate(labels.detach().cpu().tolist()):
        value = int(label)
        if is_zero_based:
            if 0 <= value < max_label:
                mapped_labels.append(value)
                keep_indices.append(index)
        elif value in COCO_91_TO_80:
            mapped_labels.append(COCO_91_TO_80[value])
            keep_indices.append(index)
        elif 0 <= value < 80:
            mapped_labels.append(value)
            keep_indices.append(index)
        elif 1 <= value <= 80:
            mapped_labels.append(value - 1)
            keep_indices.append(index)

    if not keep_indices:
        empty_boxes = boxes.new_zeros((0, 4))
        empty_labels = labels.new_zeros((0,), dtype=torch.int64)
        return empty_boxes, empty_labels, model_input_size

    keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=device)
    boxes = _project_boxes_to_model_input(
        boxes.index_select(0, keep_tensor),
        original_image_size=original_image_size,
        model_input_size=model_input_size,
        resize_mode=resize_mode,
    )
    labels = torch.as_tensor(mapped_labels, dtype=torch.int64, device=device)
    valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid_geometry], labels[valid_geometry], model_input_size


def _build_ultralytics_training_batch(
    targets: Any,
    *,
    device: torch.device,
    num_classes: int = 80,
    label_schema: str = "coco_91",
) -> dict[str, torch.Tensor]:
    if isinstance(targets, dict):
        boxes, labels, image_size = _prepare_coco80_targets(
            targets,
            device=device,
            num_classes=num_classes,
            label_schema=label_schema,
        )
        normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
        height, width = image_size
        return {
            "img": torch.zeros((1, 3, height, width), dtype=torch.float32, device=device),
            "batch_idx": torch.zeros((labels.shape[0],), dtype=torch.long, device=device),
            "cls": labels.to(dtype=torch.float32).view(-1, 1),
            "bboxes": normalized_boxes.to(dtype=torch.float32),
        }

    if (
        isinstance(targets, (list, tuple))
        and targets
        and all(isinstance(item, dict) for item in targets)
    ):
        box_pieces: list[torch.Tensor] = []
        label_pieces: list[torch.Tensor] = []
        batch_idx_pieces: list[torch.Tensor] = []
        image_size: tuple[int, int] | None = None
        for batch_index, sample_targets in enumerate(targets):
            boxes, labels, sample_image_size = _prepare_coco80_targets(
                sample_targets,
                device=device,
                num_classes=num_classes,
                label_schema=label_schema,
            )
            if image_size is None:
                image_size = sample_image_size
            elif image_size != sample_image_size:
                raise RuntimeError(
                    "Wrapper-model training expects a consistent model input size "
                    f"within each batch. Got {image_size} and {sample_image_size}."
                )
            normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=sample_image_size)
            box_pieces.append(normalized_boxes.to(dtype=torch.float32))
            label_pieces.append(labels.to(dtype=torch.float32).view(-1, 1))
            if labels.numel() == 0:
                batch_idx_pieces.append(torch.zeros((0,), dtype=torch.long, device=device))
            else:
                batch_idx_pieces.append(
                    torch.full(
                        (labels.shape[0],), int(batch_index), dtype=torch.long, device=device
                    )
                )

        if image_size is None:
            raise RuntimeError("Missing model input size metadata for wrapper-model training.")
        height, width = image_size
        bboxes = (
            torch.cat(box_pieces, dim=0)
            if box_pieces
            else torch.zeros((0, 4), dtype=torch.float32, device=device)
        )
        cls = (
            torch.cat(label_pieces, dim=0)
            if label_pieces
            else torch.zeros((0, 1), dtype=torch.float32, device=device)
        )
        batch_idx = (
            torch.cat(batch_idx_pieces, dim=0)
            if batch_idx_pieces
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        return {
            "img": torch.zeros(
                (len(targets), 3, height, width), dtype=torch.float32, device=device
            ),
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }

    raise RuntimeError(
        "Training targets must be a dict or a non-empty list of dicts for "
        "wrapper-model loss computation."
    )


def _build_detr_training_labels(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_labels: int,
) -> list[dict[str, torch.Tensor]]:
    original_image_size, image_size = _infer_original_and_model_input_image_sizes(targets)
    resize_mode = _infer_resize_mode(targets)
    _assert_original_xyxy_targets(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    if original_image_size != image_size:
        boxes = _project_boxes_to_model_input(
            boxes,
            original_image_size=original_image_size,
            model_input_size=image_size,
            resize_mode=resize_mode,
        )
    boxes = _clamp_xyxy_boxes(boxes, image_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    valid = (labels > 0) & (labels < int(num_labels))
    if boxes.numel():
        valid = valid & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
    return [
        {
            "class_labels": labels.to(dtype=torch.int64),
            "boxes": normalized_boxes.to(dtype=torch.float32),
        }
    ]


def _build_rfdetr_training_labels(
    targets: dict[str, Any],
    *,
    device: torch.device,
    num_classes: int,
    label_schema: str = "coco_91",
) -> list[dict[str, torch.Tensor]]:
    original_image_size, image_size = _infer_original_and_model_input_image_sizes(targets)
    resize_mode = _infer_resize_mode(targets)
    _assert_original_xyxy_targets(targets)
    boxes = _clamp_xyxy_boxes(
        _as_boxes_tensor(targets.get("boxes"), device=device),
        original_image_size,
    )
    if original_image_size != image_size:
        boxes = _project_boxes_to_model_input(
            boxes,
            original_image_size=original_image_size,
            model_input_size=image_size,
            resize_mode=resize_mode,
        )
    boxes = _clamp_xyxy_boxes(boxes, image_size)
    labels = _as_labels_tensor(targets.get("labels"), device=device)
    if boxes.shape[0] != labels.shape[0]:
        count = min(boxes.shape[0], labels.shape[0])
        boxes = boxes[:count]
        labels = labels[:count]

    if str(label_schema or "coco_91").strip().lower() == "zero_based":
        valid = (labels >= 0) & (labels < max(1, int(num_classes) - 1))
    else:
        valid = (labels > 0) & (labels < int(num_classes))
    if boxes.numel():
        valid = valid & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    labels = labels[valid]
    normalized_boxes = _xyxy_to_normalized_cxcywh(boxes, image_size=image_size)
    return [
        {
            "labels": labels.to(dtype=torch.int64),
            "boxes": normalized_boxes.to(dtype=torch.float32),
        }
    ]
