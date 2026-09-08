from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def _nan() -> float:
    return float("nan")


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _rgb_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def mse(raw_rgb: np.ndarray, recon_rgb: np.ndarray) -> float:
    raw = _rgb_float(raw_rgb)
    recon = _resize_like(_rgb_float(recon_rgb), raw)
    return float(np.mean((raw - recon) ** 2))


def psnr(raw_rgb: np.ndarray, recon_rgb: np.ndarray, *, eps: float = 1.0e-12) -> float:
    value = mse(raw_rgb, recon_rgb)
    if value <= eps:
        return float("inf")
    return float(10.0 * math.log10(1.0 / value))


def psnr_norm(value: float, *, max_db: float = 40.0) -> float:
    if math.isinf(value):
        return 1.0
    if _is_nan(value):
        return _nan()
    return float(np.clip(float(value) / max(float(max_db), 1.0), 0.0, 1.0))


def ssim(raw_rgb: np.ndarray, recon_rgb: np.ndarray) -> float:
    raw = _rgb_float(raw_rgb)
    recon = _resize_like(_rgb_float(recon_rgb), raw)
    c1 = 0.01**2
    c2 = 0.03**2
    scores: list[float] = []
    for channel in range(3):
        x = raw[..., channel]
        y = recon[..., channel]
        mux = float(x.mean())
        muy = float(y.mean())
        varx = float(((x - mux) ** 2).mean())
        vary = float(((y - muy) ** 2).mean())
        cov = float(((x - mux) * (y - muy)).mean())
        denom = (mux * mux + muy * muy + c1) * (varx + vary + c2)
        if denom <= 0.0:
            scores.append(1.0 if np.allclose(x, y) else 0.0)
        else:
            scores.append(((2 * mux * muy + c1) * (2 * cov + c2)) / denom)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if tuple(image.shape[:2]) == tuple(reference.shape[:2]):
        return image
    import cv2

    height, width = reference.shape[:2]
    return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def _image_size(
    prediction: Mapping[str, Any] | None,
    default_size: tuple[int, int] | list[int] | None,
) -> tuple[float, float] | None:
    candidates: list[Any] = []
    if isinstance(prediction, Mapping):
        candidates.extend(
            prediction.get(key) for key in ("image_size", "image_shape", "raw_image_size")
        )
    candidates.append(default_size)
    for value in candidates:
        if value is None:
            continue
        try:
            size = list(value)
        except TypeError:
            continue
        if len(size) < 2:
            continue
        height = float(size[0])
        width = float(size[1])
        if height > 0.0 and width > 0.0:
            return height, width
    return None


def _normalise_box(box: list[float], image_size: tuple[float, float] | None) -> list[float]:
    if max(abs(value) for value in box) <= 1.5:
        return box
    if image_size is None:
        return box
    height, width = image_size
    if height <= 0.0 or width <= 0.0:
        return box
    return [
        float(np.clip(box[0] / width, 0.0, 1.0)),
        float(np.clip(box[1] / height, 0.0, 1.0)),
        float(np.clip(box[2] / width, 0.0, 1.0)),
        float(np.clip(box[3] / height, 0.0, 1.0)),
    ]


def _prediction_items(
    prediction: Mapping[str, Any] | None,
    *,
    image_size: tuple[float, float] | None = None,
) -> list[tuple[list[float], int, float]]:
    if not isinstance(prediction, Mapping):
        return []
    boxes = _as_list(prediction.get("boxes"))
    labels = _as_list(prediction.get("labels"))
    scores = _as_list(prediction.get("scores"))
    items: list[tuple[list[float], int, float]] = []
    for index, box in enumerate(boxes):
        try:
            coords = [float(value) for value in list(box)[:4]]
        except (TypeError, ValueError):
            continue
        if len(coords) != 4:
            continue
        coords = _normalise_box(coords, image_size)
        label = int(labels[index]) if index < len(labels) else -1
        score = float(scores[index]) if index < len(scores) else 1.0
        items.append((coords, label, score))
    return items


def box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


def object_metrics(
    original_teacher_prediction: Mapping[str, Any] | None,
    recon_teacher_prediction: Mapping[str, Any] | None,
    *,
    iou_threshold: float = 0.5,
    original_image_shape: tuple[int, int] | list[int] | None = None,
    recon_image_shape: tuple[int, int] | list[int] | None = None,
) -> dict[str, float]:
    teacher_size = _image_size(original_teacher_prediction, original_image_shape)
    recon_size = _image_size(recon_teacher_prediction, recon_image_shape)
    teacher_items = _prediction_items(original_teacher_prediction, image_size=teacher_size)
    recon_items = _prediction_items(recon_teacher_prediction, image_size=recon_size)
    if not teacher_items:
        return {"ObjectPrecision": _nan(), "ObjectRecall": _nan(), "ObjectF1": _nan()}
    if not recon_items:
        return {"ObjectPrecision": 0.0, "ObjectRecall": 0.0, "ObjectF1": 0.0}

    matched_teacher: set[int] = set()
    true_positive = 0
    for pred_box, pred_label, pred_score in sorted(
        recon_items, key=lambda item: item[2], reverse=True
    ):
        del pred_score
        best_index = -1
        best_iou = 0.0
        for teacher_index, (teacher_box, teacher_label, _teacher_score) in enumerate(teacher_items):
            if teacher_index in matched_teacher or int(pred_label) != int(teacher_label):
                continue
            iou = box_iou(pred_box, teacher_box)
            if iou > best_iou:
                best_iou = iou
                best_index = teacher_index
        if best_index >= 0 and best_iou >= float(iou_threshold):
            matched_teacher.add(best_index)
            true_positive += 1
    precision = true_positive / float(max(len(recon_items), 1))
    recall = true_positive / float(max(len(teacher_items), 1))
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "ObjectPrecision": float(precision),
        "ObjectRecall": float(recall),
        "ObjectF1": float(f1),
    }


class OptionalLPIPS:
    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.available = False
        self.error: str | None = None
        self.model = None
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").to(self.device).eval()
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            self.error = str(exc)

    def __call__(self, raw_rgb: np.ndarray, recon_rgb: np.ndarray) -> float:
        if not self.available or self.model is None:
            return _nan()
        raw = _to_lpips_tensor(raw_rgb, self.device)
        recon = _to_lpips_tensor(
            _resize_like(_rgb_float(recon_rgb), _rgb_float(raw_rgb)), self.device
        )
        with torch.no_grad():
            value = self.model(raw, recon)
        return float(value.detach().cpu().reshape(-1)[0].item())


def _to_lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = _rgb_float(image)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.mul(2.0).sub(1.0).to(device)


def l_actual(
    metrics: Mapping[str, Any], *, lpips_available: bool, psnr_norm_max: float = 40.0
) -> float:
    object_f1 = float(metrics.get("ObjectF1", _nan()))
    if _is_nan(object_f1):
        return _nan()
    ssim_value = float(metrics.get("SSIM", _nan()))
    psnr_value = psnr_norm(float(metrics.get("PSNR", _nan())), max_db=psnr_norm_max)
    if lpips_available and not _is_nan(metrics.get("LPIPS")):
        lpips_norm_max = max(float(metrics.get("LPIPSNormMax", 1.0) or 1.0), 1.0e-12)
        lpips_norm = float(np.clip(float(metrics["LPIPS"]) / lpips_norm_max, 0.0, 1.0))
        return float(
            0.4 * object_f1 + 0.3 * ssim_value + 0.2 * (1.0 - lpips_norm) + 0.1 * psnr_value
        )
    return float(0.5 * object_f1 + 0.3 * ssim_value + 0.2 * psnr_value)


def evaluate_reconstruction(
    raw_rgb: np.ndarray,
    recon_rgb: np.ndarray,
    *,
    original_teacher_prediction: Mapping[str, Any] | None,
    recon_teacher_prediction: Mapping[str, Any] | None,
    feature_distance_final: float | None,
    lpips_metric: OptionalLPIPS | None = None,
    object_iou_threshold: float = 0.5,
    psnr_norm_max: float = 40.0,
    lpips_norm_max: float = 1.0,
) -> dict[str, Any]:
    lpips_value = _nan()
    lpips_available = bool(lpips_metric is not None and lpips_metric.available)
    if lpips_available and lpips_metric is not None:
        lpips_value = lpips_metric(raw_rgb, recon_rgb)
    result: dict[str, Any] = {
        "MSE": mse(raw_rgb, recon_rgb),
        "PSNR": psnr(raw_rgb, recon_rgb),
        "SSIM": ssim(raw_rgb, recon_rgb),
        "LPIPS": lpips_value,
        "LPIPSNormMax": float(lpips_norm_max),
        "FeatureDistanceFinal": float(feature_distance_final)
        if feature_distance_final is not None
        else _nan(),
    }
    result.update(
        object_metrics(
            original_teacher_prediction,
            recon_teacher_prediction,
            iou_threshold=float(object_iou_threshold),
            original_image_shape=raw_rgb.shape[:2],
            recon_image_shape=recon_rgb.shape[:2],
        )
    )
    result["L_actual"] = l_actual(
        result,
        lpips_available=lpips_available,
        psnr_norm_max=float(psnr_norm_max),
    )
    result["LPIPSAvailable"] = bool(lpips_available)
    result["ObjectF1Valid"] = not _is_nan(result["ObjectF1"])
    return result
