from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def _gray_float(frame: object, *, max_dimension: int = 640) -> tuple[np.ndarray, float] | None:
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.ndim == 3:
        if array.shape[2] < 3:
            gray = array[..., 0]
        else:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        return None

    gray = np.ascontiguousarray(gray.astype("float32", copy=False))
    height, width = gray.shape[:2]
    if height <= 1 or width <= 1:
        return None
    scale = 1.0
    max_side = max(height, width)
    if max_side > int(max_dimension) > 0:
        scale = float(max_dimension) / float(max_side)
        resized_width = max(2, int(round(width * scale)))
        resized_height = max(2, int(round(height * scale)))
        gray = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return gray, scale


def estimate_frame_translation(
    reference_frame: object,
    current_frame: object,
    *,
    min_response: float = 0.02,
    max_shift_fraction: float = 0.25,
) -> tuple[float, float] | None:
    """Estimate global x/y motion from ``reference_frame`` to ``current_frame``."""
    reference = _gray_float(reference_frame)
    current = _gray_float(current_frame)
    if reference is None or current is None:
        return None
    reference_gray, reference_scale = reference
    current_gray, current_scale = current
    if reference_gray.shape != current_gray.shape or abs(reference_scale - current_scale) > 1e-6:
        return None

    height, width = reference_gray.shape[:2]
    try:
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        (dx_scaled, dy_scaled), response = cv2.phaseCorrelate(
            reference_gray,
            current_gray,
            window,
        )
    except Exception:
        return None

    if not np.isfinite(dx_scaled) or not np.isfinite(dy_scaled) or not np.isfinite(response):
        return None
    if float(response) < float(min_response):
        return None

    scale = max(reference_scale, 1e-6)
    dx = float(dx_scaled) / scale
    dy = float(dy_scaled) / scale
    original_shape = getattr(np.asarray(reference_frame), "shape", ())
    if len(original_shape) < 2:
        return None
    original_height, original_width = int(original_shape[0]), int(original_shape[1])
    if abs(dx) > float(original_width) * float(max_shift_fraction) or abs(dy) > float(
        original_height
    ) * float(max_shift_fraction):
        return None
    return dx, dy


def translate_boxes(
    boxes: Iterable[Iterable[float]],
    *,
    image_shape: tuple[int, ...] | list[int],
    dx: float,
    dy: float,
) -> tuple[list[list[float]], list[int]]:
    height = float(image_shape[0])
    width = float(image_shape[1])
    translated: list[list[float]] = []
    keep_indices: list[int] = []
    for index, box in enumerate(list(boxes or [])):
        try:
            x1, y1, x2, y2 = [float(value) for value in list(box)[:4]]
        except (TypeError, ValueError):
            continue
        x1 = max(0.0, min(width, x1 + float(dx)))
        x2 = max(0.0, min(width, x2 + float(dx)))
        y1 = max(0.0, min(height, y1 + float(dy)))
        y2 = max(0.0, min(height, y2 + float(dy)))
        if x2 <= x1 or y2 <= y1:
            continue
        translated.append([x1, y1, x2, y2])
        keep_indices.append(index)
    return translated, keep_indices


def compensate_boxes_between_frames(
    boxes: Iterable[Iterable[float]],
    reference_frame: object,
    current_frame: object,
) -> tuple[list[list[float]], list[int]]:
    shift = estimate_frame_translation(reference_frame, current_frame)
    if shift is None:
        return [], []
    dx, dy = shift
    shape = getattr(np.asarray(current_frame), "shape", ())
    if len(shape) < 2:
        return [], []
    return translate_boxes(boxes, image_shape=shape, dx=dx, dy=dy)
