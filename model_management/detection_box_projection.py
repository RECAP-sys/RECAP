from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

ORIGINAL_XYXY = "original_xyxy"
MODEL_INPUT_XYXY = "model_input_xyxy"
SUPPORTED_RESIZE_MODES = {"direct_resize", "letterbox"}


@dataclass(frozen=True)
class CoordinateValidation:
    ok: bool
    reason: str | None = None


def _size_pair_from_value(value: object) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        height = int(value[0])
        width = int(value[1])
        if height > 0 and width > 0:
            return height, width
    return None


def infer_original_image_size(
    metadata: Mapping[str, object] | None,
    raw_frame: Any | None = None,
) -> tuple[int, int] | None:
    if isinstance(metadata, Mapping):
        size = _size_pair_from_value(metadata.get("input_image_size"))
        if size is not None:
            return size
        size = _size_pair_from_value(metadata.get("label_image_size"))
        if size is not None:
            return size
    shape = getattr(raw_frame, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        height = int(shape[0])
        width = int(shape[1])
        if height > 0 and width > 0:
            return height, width
    return None


def infer_model_input_size(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    input_tensor_shape = metadata.get("input_tensor_shape")
    if isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) >= 3:
        height = int(input_tensor_shape[-2])
        width = int(input_tensor_shape[-1])
        if height > 0 and width > 0:
            return height, width
    return None


def _input_image_size_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    return _size_pair_from_value(metadata.get("input_image_size"))


def infer_resize_mode(metadata: Mapping[str, object] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = str(metadata.get("input_resize_mode") or "").strip().lower()
    return value if value in SUPPORTED_RESIZE_MODES else None


def require_coordinate_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    original_size = _input_image_size_from_metadata(metadata)
    model_input_size = infer_model_input_size(metadata)
    resize_mode = infer_resize_mode(metadata)
    missing = []
    if original_size is None:
        missing.append("input_image_size")
    if model_input_size is None:
        missing.append("input_tensor_shape")
    if resize_mode is None:
        missing.append("input_resize_mode")
    if missing:
        raise RuntimeError(
            "Missing coordinate metadata required for split retraining: " + ", ".join(missing)
        )
    return original_size, model_input_size, resize_mode


def _boxes_to_tensor(boxes: Any) -> tuple[torch.Tensor, bool]:
    if isinstance(boxes, torch.Tensor):
        tensor = boxes.to(dtype=torch.float32)
        if tensor.numel() == 0:
            return tensor.reshape(0, 4), True
        return tensor.reshape(-1, 4), True
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32), False
    tensor = torch.as_tensor(boxes, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor.reshape(0, 4), False
    return tensor.reshape(-1, 4), False


def _return_boxes(projected: torch.Tensor, was_tensor: bool) -> torch.Tensor | list[list[float]]:
    if was_tensor:
        return projected
    return projected.detach().cpu().tolist()


def _clamp_xyxy(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp_(0.0, float(width))
    boxes[..., 1::2] = boxes[..., 1::2].clamp_(0.0, float(height))
    return boxes


def project_original_xyxy_to_model_input_xyxy(
    boxes: Any,
    original_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str,
) -> torch.Tensor | list[list[float]]:
    tensor, was_tensor = _boxes_to_tensor(boxes)
    if tensor.numel() == 0:
        return _return_boxes(tensor, was_tensor)

    orig_h, orig_w = original_size
    model_h, model_w = model_input_size
    if min(orig_h, orig_w, model_h, model_w) <= 0:
        raise RuntimeError("Image sizes must be positive for box projection.")

    mode = str(resize_mode or "").strip().lower()
    projected = tensor.clone()
    if mode == "letterbox":
        scale = min(float(model_w) / float(orig_w), float(model_h) / float(orig_h))
        pad_x = (float(model_w) - float(orig_w) * scale) * 0.5
        pad_y = (float(model_h) - float(orig_h) * scale) * 0.5
        projected[..., 0::2] = projected[..., 0::2] * scale + pad_x
        projected[..., 1::2] = projected[..., 1::2] * scale + pad_y
    elif mode == "direct_resize":
        projected[..., 0::2] = projected[..., 0::2] * (float(model_w) / float(orig_w))
        projected[..., 1::2] = projected[..., 1::2] * (float(model_h) / float(orig_h))
    else:
        raise RuntimeError(f"Unsupported resize mode for box projection: {resize_mode!r}")
    return _return_boxes(_clamp_xyxy(projected, model_input_size), was_tensor)


def project_model_input_xyxy_to_original_xyxy(
    boxes: Any,
    original_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str,
) -> torch.Tensor | list[list[float]]:
    tensor, was_tensor = _boxes_to_tensor(boxes)
    if tensor.numel() == 0:
        return _return_boxes(tensor, was_tensor)

    orig_h, orig_w = original_size
    model_h, model_w = model_input_size
    if min(orig_h, orig_w, model_h, model_w) <= 0:
        raise RuntimeError("Image sizes must be positive for box projection.")

    mode = str(resize_mode or "").strip().lower()
    projected = tensor.clone()
    if mode == "letterbox":
        scale = min(float(model_w) / float(orig_w), float(model_h) / float(orig_h))
        pad_x = (float(model_w) - float(orig_w) * scale) * 0.5
        pad_y = (float(model_h) - float(orig_h) * scale) * 0.5
        projected[..., 0::2] = (projected[..., 0::2] - pad_x) / scale
        projected[..., 1::2] = (projected[..., 1::2] - pad_y) / scale
    elif mode == "direct_resize":
        projected[..., 0::2] = projected[..., 0::2] * (float(orig_w) / float(model_w))
        projected[..., 1::2] = projected[..., 1::2] * (float(orig_h) / float(model_h))
    else:
        raise RuntimeError(f"Unsupported resize mode for box projection: {resize_mode!r}")
    return _return_boxes(_clamp_xyxy(projected, original_size), was_tensor)


def _list_or_empty(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value)


def _box_values(box: Any) -> list[float] | None:
    try:
        values = [float(value) for value in list(box)[:4]]
    except (TypeError, ValueError):
        return None
    return values if len(values) == 4 else None


def _normalise_boxes(boxes: Any) -> list[list[float]]:
    normalised: list[list[float]] = []
    for box in _list_or_empty(boxes):
        values = _box_values(box)
        if values is None:
            raise ValueError("Detection boxes must be xyxy values.")
        normalised.append(values)
    return normalised


def _normalise_ints(values: Any) -> list[int]:
    return [int(value) for value in _list_or_empty(values)]


def _normalise_floats(values: Any) -> list[float]:
    return [float(value) for value in _list_or_empty(values)]


def _labels_are_structurally_valid(labels: Mapping[str, Any]) -> bool:
    boxes = _list_or_empty(labels.get("boxes"))
    label_values = _list_or_empty(labels.get("labels"))
    if bool(boxes) != bool(label_values):
        return False
    if boxes and len(boxes) != len(label_values):
        return False
    for box in boxes:
        if _box_values(box) is None:
            return False
    return True


def _boxes_fit_size(labels: Mapping[str, Any], image_size: tuple[int, int]) -> bool:
    height, width = image_size
    epsilon = 1e-3
    for box in _list_or_empty(labels.get("boxes")):
        values = _box_values(box)
        if values is None:
            return False
        if (
            values[0] < -epsilon
            or values[2] < -epsilon
            or values[1] < -epsilon
            or values[3] < -epsilon
            or values[0] > float(width) + epsilon
            or values[2] > float(width) + epsilon
            or values[1] > float(height) + epsilon
            or values[3] > float(height) + epsilon
        ):
            return False
    return True


def validate_box_coordinate_space(
    labels: Mapping[str, Any],
    metadata: Mapping[str, object] | None,
) -> CoordinateValidation:
    if not _labels_are_structurally_valid(labels):
        return CoordinateValidation(False, "label_structure")
    coordinate_space = str(labels.get("label_coordinate_space") or "").strip()
    if not coordinate_space:
        return CoordinateValidation(False, "missing_label_coordinate_space")
    if coordinate_space not in {ORIGINAL_XYXY, MODEL_INPUT_XYXY}:
        return CoordinateValidation(False, "unsupported_label_coordinate_space")

    original_size = _input_image_size_from_metadata(metadata)
    model_input_size = infer_model_input_size(metadata)
    resize_mode = infer_resize_mode(metadata)
    if original_size is None:
        return CoordinateValidation(False, "missing_input_image_size")
    if model_input_size is None:
        return CoordinateValidation(False, "missing_input_tensor_shape")
    if resize_mode is None:
        return CoordinateValidation(False, "missing_input_resize_mode")

    if coordinate_space == ORIGINAL_XYXY:
        label_image_size = _size_pair_from_value(labels.get("label_image_size"))
        if label_image_size is not None and label_image_size != original_size:
            return CoordinateValidation(False, "label_image_size")
        if not _boxes_fit_size(labels, original_size):
            return CoordinateValidation(False, "label_bounds")
        return CoordinateValidation(True)

    label_input_size = _size_pair_from_value(labels.get("label_input_size"))
    if label_input_size is None:
        return CoordinateValidation(False, "missing_label_input_size")
    if label_input_size != model_input_size:
        return CoordinateValidation(False, "label_input_size")
    label_resize_mode = str(labels.get("label_resize_mode") or "").strip().lower()
    if label_resize_mode and label_resize_mode != resize_mode:
        return CoordinateValidation(False, "label_resize_mode")
    if not _boxes_fit_size(labels, model_input_size):
        return CoordinateValidation(False, "label_bounds")
    return CoordinateValidation(True)


def canonicalize_labels_to_original_xyxy(
    labels: Mapping[str, Any],
    metadata: Mapping[str, object],
) -> dict[str, Any]:
    validation = validate_box_coordinate_space(labels, metadata)
    if not validation.ok:
        raise ValueError(f"Invalid detection label coordinate metadata: {validation.reason}")

    original_size, model_input_size, resize_mode = require_coordinate_metadata(metadata)
    canonical = {
        "boxes": _normalise_boxes(labels.get("boxes")),
        "labels": _normalise_ints(labels.get("labels")),
        "label_coordinate_space": ORIGINAL_XYXY,
        "label_image_size": [int(original_size[0]), int(original_size[1])],
        "label_resize_mode": resize_mode,
    }
    if labels.get("scores") is not None:
        canonical["scores"] = _normalise_floats(labels.get("scores"))
    coordinate_space = str(labels.get("label_coordinate_space") or "").strip()
    if coordinate_space == MODEL_INPUT_XYXY:
        canonical["boxes"] = project_model_input_xyxy_to_original_xyxy(
            canonical["boxes"],
            original_size,
            model_input_size,
            resize_mode,
        )
    return canonical
