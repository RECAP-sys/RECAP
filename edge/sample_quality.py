from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from model_management.payload import BoundaryPayload

HIGH_QUALITY = "high_quality"
LOW_QUALITY = "low_quality"
QUALITY_METHOD = "output_boundary_entropy"


@dataclass
class _FeatureEntropyState:
    count: int
    mean: float
    variance: float


@dataclass
class EntropyQualityStats:
    output_entropy: float | None
    output_entropy_threshold: float | None
    output_confidence: float | None
    output_confidence_threshold: float
    output_confident: bool
    feature_entropy: float | None
    feature_entropy_mean: float | None
    feature_entropy_std: float | None
    feature_entropy_deviation: float | None
    feature_deviation_threshold: float
    output_reliable: bool
    feature_normal: bool
    edge_pseudo_label_trusted: bool
    quality: str
    reason: str
    window_id: str | None = None
    in_drift_window: bool = False

    @property
    def quality_bucket(self) -> str:
        return self.quality

    def quality_metadata(self, *, persist_debug_stats: bool = False) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "method": QUALITY_METHOD,
            "quality": self.quality,
        }
        if persist_debug_stats:
            metadata["debug"] = {
                "output_entropy": self.output_entropy,
                "output_entropy_threshold": self.output_entropy_threshold,
                "output_confidence": self.output_confidence,
                "output_confidence_threshold": self.output_confidence_threshold,
                "output_confident": self.output_confident,
                "feature_entropy": self.feature_entropy,
                "feature_entropy_mean": self.feature_entropy_mean,
                "feature_entropy_std": self.feature_entropy_std,
                "feature_entropy_deviation": self.feature_entropy_deviation,
                "feature_deviation_threshold": self.feature_deviation_threshold,
                "output_reliable": self.output_reliable,
                "feature_normal": self.feature_normal,
                "edge_pseudo_label_trusted": self.edge_pseudo_label_trusted,
                "reason": self.reason,
            }
        return metadata


def _get_config_value(config: object | None, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _percentile(values: list[float], percentile: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return float(finite[0])
    rank = max(0.0, min(100.0, float(percentile))) / 100.0 * (len(finite) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(finite[lower])
    fraction = rank - lower
    return float(finite[lower] * (1.0 - fraction) + finite[upper] * fraction)


def _as_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value)
    except Exception:
        return None
    return tensor.detach() if isinstance(tensor, torch.Tensor) else None


def _tensor_rows(tensor: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    work = tensor.detach().float()
    if work.ndim == 1:
        return work.reshape(1, -1)
    if work.ndim == 2:
        return work
    if work.ndim == 3:
        if work.shape[-1] > 1 and work.shape[-1] <= 512:
            return work.reshape(-1, work.shape[-1])
        if work.shape[1] > 1 and work.shape[1] <= 512 and work.shape[2] > work.shape[1]:
            return work.permute(0, 2, 1).reshape(-1, work.shape[1])
        return work.reshape(-1, work.shape[-1])
    if work.ndim == 4:
        if work.shape[-1] > 1 and work.shape[-1] <= 512:
            return work.reshape(-1, work.shape[-1])
        if work.shape[1] > 1 and work.shape[1] <= 512:
            return work.permute(0, 2, 3, 1).reshape(-1, work.shape[1])
    return None


def _normalised_entropy_from_probs(probs: torch.Tensor, *, eps: float) -> float | None:
    rows = _tensor_rows(probs)
    if rows is None or rows.numel() == 0 or rows.shape[-1] <= 1:
        return None
    work = rows.detach().float().clamp_min(0.0)
    row_sums = work.sum(dim=-1, keepdim=True)
    valid = row_sums.squeeze(-1) > float(eps)
    if not bool(valid.any()):
        return None
    work = work[valid] / row_sums[valid].clamp_min(float(eps))
    entropy = -(work * torch.log(work.clamp_min(float(eps)))).sum(dim=-1)
    entropy = entropy / max(math.log(max(2, int(work.shape[-1]))), float(eps))
    return float(entropy.mean().item())


def _normalised_entropy_from_logits(
    logits: torch.Tensor,
    *,
    mode: str,
    eps: float,
    max_rows: int = 256,
) -> float | None:
    rows = _tensor_rows(logits)
    if rows is None or rows.numel() == 0 or rows.shape[-1] <= 1:
        return None
    work = rows.detach().float()
    if mode == "softmax_bg_last" and work.shape[-1] > 1:
        work = work[:, :-1]
    if work.shape[-1] <= 1:
        return None
    if work.shape[0] > max_rows:
        priority = torch.softmax(work, dim=-1).max(dim=-1).values
        keep = torch.topk(priority, k=max_rows).indices
        work = work.index_select(0, keep)
    if mode.startswith("sigmoid"):
        probs = torch.sigmoid(work)
        p = probs.max(dim=-1).values.clamp(float(eps), 1.0 - float(eps))
        entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
        entropy = entropy / math.log(2.0)
    else:
        probs = torch.softmax(work, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(float(eps)))).sum(dim=-1)
        entropy = entropy / max(math.log(max(2, int(probs.shape[-1]))), float(eps))
    return float(entropy.mean().item())


class EntropyQualityClassifier:
    def __init__(
        self,
        *,
        enabled: bool = True,
        output_window_size: int = 256,
        output_percentile: float = 25.0,
        output_warmup_samples: int = 20,
        output_min_detection_confidence: float = 0.85,
        feature_max_elements: int = 4096,
        feature_ema_decay: float = 0.95,
        feature_deviation_threshold: float = 1.5,
        feature_min_std: float = 1.0e-4,
        feature_relative_std_floor: float = 0.02,
        feature_update_on_anomaly: bool = False,
        feature_warmup_samples: int = 20,
        eps: float = 1.0e-8,
        persist_debug_stats: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_window_size = max(1, int(output_window_size))
        self.output_percentile = float(output_percentile)
        self.output_warmup_samples = max(0, int(output_warmup_samples))
        self.output_min_detection_confidence = max(
            0.0,
            min(1.0, float(output_min_detection_confidence)),
        )
        self.feature_max_elements = max(1, int(feature_max_elements))
        self.feature_ema_decay = max(0.0, min(0.999999, float(feature_ema_decay)))
        self.feature_deviation_threshold = float(feature_deviation_threshold)
        self.feature_min_std = max(0.0, float(feature_min_std))
        self.feature_relative_std_floor = max(0.0, float(feature_relative_std_floor))
        self.feature_update_on_anomaly = bool(feature_update_on_anomaly)
        self.feature_warmup_samples = max(0, int(feature_warmup_samples))
        self.eps = max(float(eps), 1.0e-12)
        self.persist_debug_stats = bool(persist_debug_stats)
        self._output_windows: dict[tuple[str, str, str], deque[float]] = {}
        self._feature_states: dict[tuple[str, str, str], _FeatureEntropyState] = {}
        self._feature_sample_indices: dict[int, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: object | None) -> "EntropyQualityClassifier":
        output_cfg = _get_config_value(config, "output_entropy", None)
        feature_cfg = _get_config_value(config, "boundary_feature_entropy", None)
        return cls(
            enabled=bool(_get_config_value(config, "enabled", True)),
            output_window_size=int(_get_config_value(output_cfg, "window_size", 256)),
            output_percentile=float(_get_config_value(output_cfg, "percentile", 25.0)),
            output_warmup_samples=int(_get_config_value(output_cfg, "warmup_samples", 20)),
            output_min_detection_confidence=float(
                _get_config_value(output_cfg, "min_detection_confidence", 0.85)
            ),
            feature_max_elements=int(_get_config_value(feature_cfg, "max_elements", 4096)),
            feature_ema_decay=float(_get_config_value(feature_cfg, "ema_decay", 0.95)),
            feature_deviation_threshold=float(
                _get_config_value(feature_cfg, "deviation_threshold", 1.5)
            ),
            feature_min_std=float(_get_config_value(feature_cfg, "min_std", 1.0e-4)),
            feature_relative_std_floor=float(
                _get_config_value(feature_cfg, "relative_std_floor", 0.02)
            ),
            feature_update_on_anomaly=bool(
                _get_config_value(feature_cfg, "update_on_anomaly", False)
            ),
            feature_warmup_samples=int(_get_config_value(feature_cfg, "warmup_samples", 20)),
            eps=float(_get_config_value(config, "eps", 1.0e-8)),
            persist_debug_stats=bool(_get_config_value(config, "persist_debug_stats", False)),
        )

    def reset(self) -> None:
        """Reset model-dependent calibration after a model update."""

        self._output_windows.clear()
        self._feature_states.clear()

    def classify(
        self,
        predictions: object,
        boundary_payload: object,
        model_name: str,
        split_key: str,
        feature_abi_id: str,
    ) -> EntropyQualityStats:
        key = (str(model_name or ""), str(split_key or ""), str(feature_abi_id or ""))
        output_entropy = self._compute_output_entropy(predictions)
        output_threshold, output_entropy_reliable, output_warmup = self._classify_output(
            key,
            output_entropy,
        )
        output_confidence = self._compute_output_confidence(predictions)
        output_confident = self.output_min_detection_confidence <= 0.0 or (
            output_confidence is not None
            and float(output_confidence) >= self.output_min_detection_confidence
        )
        output_reliable = bool(output_entropy_reliable and output_confident)
        feature_entropy = self._compute_feature_entropy(boundary_payload)
        (
            feature_mean,
            feature_std,
            feature_deviation,
            feature_normal,
            feature_warmup,
        ) = self._classify_feature(key, feature_entropy)

        empty_predictions = self._predictions_empty(predictions)
        trusted = (
            self.enabled
            and not empty_predictions
            and bool(output_reliable)
            and bool(feature_normal)
        )
        quality = HIGH_QUALITY if trusted else LOW_QUALITY
        reasons: list[str] = []
        if not self.enabled:
            reasons.append("classifier_disabled")
        if empty_predictions:
            reasons.append("empty_predictions")
        if output_entropy is None:
            reasons.append("output_entropy_unavailable")
        elif output_warmup:
            reasons.append("output_entropy_warmup")
        elif not output_entropy_reliable:
            reasons.append("output_entropy_high")
        if not output_confident:
            if output_confidence is None:
                reasons.append("output_confidence_unavailable")
            else:
                reasons.append("output_confidence_low")
        if feature_entropy is None:
            reasons.append("feature_entropy_unavailable")
        elif feature_warmup:
            reasons.append("feature_entropy_warmup")
        elif not feature_normal:
            reasons.append("feature_entropy_deviation_high")
        if trusted:
            reasons.append("trusted_edge_pseudo_label")

        return EntropyQualityStats(
            output_entropy=output_entropy,
            output_entropy_threshold=output_threshold,
            output_confidence=output_confidence,
            output_confidence_threshold=self.output_min_detection_confidence,
            output_confident=bool(output_confident),
            feature_entropy=feature_entropy,
            feature_entropy_mean=feature_mean,
            feature_entropy_std=feature_std,
            feature_entropy_deviation=feature_deviation,
            feature_deviation_threshold=self.feature_deviation_threshold,
            output_reliable=bool(output_reliable),
            feature_normal=bool(feature_normal),
            edge_pseudo_label_trusted=bool(trusted),
            quality=quality,
            reason=";".join(reasons) if reasons else "low_quality",
        )

    def _classify_output(
        self,
        key: tuple[str, str, str],
        output_entropy: float | None,
    ) -> tuple[float | None, bool, bool]:
        window = self._output_windows.setdefault(
            key,
            deque(maxlen=self.output_window_size),
        )
        warmed = len(window) >= self.output_warmup_samples
        threshold = _percentile(list(window), self.output_percentile)
        if threshold is None and self.output_warmup_samples <= 0:
            threshold = 1.0
            warmed = True
        reliable = (
            output_entropy is not None
            and threshold is not None
            and warmed
            and float(output_entropy) <= float(threshold)
        )
        if output_entropy is not None:
            window.append(float(output_entropy))
        return threshold, bool(reliable), not warmed

    def _classify_feature(
        self,
        key: tuple[str, str, str],
        feature_entropy: float | None,
    ) -> tuple[float | None, float | None, float | None, bool, bool]:
        if feature_entropy is None:
            return None, None, None, False, True

        state = self._feature_states.get(key)
        if state is None:
            if self.feature_warmup_samples <= 0:
                self._feature_states[key] = _FeatureEntropyState(
                    count=1,
                    mean=float(feature_entropy),
                    variance=0.0,
                )
                return float(feature_entropy), self.feature_min_std, 0.0, True, False
            self._feature_states[key] = _FeatureEntropyState(
                count=1,
                mean=float(feature_entropy),
                variance=0.0,
            )
            return float(feature_entropy), self.feature_min_std, None, False, True

        prior_std = max(
            math.sqrt(max(0.0, state.variance)),
            self.feature_min_std,
            abs(float(state.mean)) * self.feature_relative_std_floor,
        )
        deviation = abs(float(feature_entropy) - float(state.mean)) / max(prior_std, self.eps)
        warmed = state.count >= self.feature_warmup_samples
        feature_normal = warmed and deviation <= self.feature_deviation_threshold

        if not warmed or feature_normal or self.feature_update_on_anomaly:
            decay = self.feature_ema_decay
            delta = float(feature_entropy) - float(state.mean)
            new_mean = (decay * float(state.mean)) + ((1.0 - decay) * float(feature_entropy))
            new_variance = decay * (float(state.variance) + ((1.0 - decay) * delta * delta))
            self._feature_states[key] = _FeatureEntropyState(
                count=state.count + 1,
                mean=float(new_mean),
                variance=float(max(0.0, new_variance)),
            )
        return (
            float(state.mean),
            float(prior_std),
            float(deviation),
            bool(feature_normal),
            not warmed,
        )

    def _compute_output_entropy(self, predictions: object) -> float | None:
        direct = _finite_float(_read_prediction_value(predictions, "output_entropy"))
        if direct is not None:
            return max(0.0, direct)
        direct = _finite_float(_read_prediction_value(predictions, "logit_entropy"))
        if direct is not None:
            return max(0.0, direct)

        for key, mode in (
            ("dense_logits", "softmax_bg_last"),
            ("pre_nms_logits", "softmax_bg_last"),
            ("query_logits", "softmax_bg_last"),
            ("pred_logits", "softmax_bg_last"),
            ("logits", "softmax_bg_last"),
            ("cls_logits", "sigmoid"),
        ):
            tensor = _as_tensor(_read_prediction_value(predictions, key))
            if tensor is None:
                continue
            entropy = _normalised_entropy_from_logits(tensor, mode=mode, eps=self.eps)
            if entropy is not None:
                return max(0.0, entropy)

        for key in (
            "class_probabilities",
            "class_probs",
            "probabilities",
            "probs",
            "per_detection_probs",
        ):
            tensor = _as_tensor(_read_prediction_value(predictions, key))
            if tensor is None:
                continue
            entropy = _normalised_entropy_from_probs(tensor, eps=self.eps)
            if entropy is not None:
                return max(0.0, entropy)

        scores = _read_prediction_value(predictions, "final_detection_scores")
        if scores is None:
            scores = _read_prediction_value(predictions, "scores")
        if scores is None:
            scores = _read_prediction_value(predictions, "low_threshold_scores")
        tensor = _as_tensor(scores)
        if tensor is None or tensor.numel() == 0:
            return None
        p = tensor.detach().float().flatten().clamp(float(self.eps), 1.0 - float(self.eps))
        entropy = -((p * torch.log(p)) + ((1.0 - p) * torch.log(1.0 - p)))
        entropy = entropy / math.log(2.0)
        return float(entropy.mean().item())

    def _compute_output_confidence(self, predictions: object) -> float | None:
        direct = _finite_float(_read_prediction_value(predictions, "confidence"))
        if direct is not None:
            return max(0.0, min(1.0, direct))

        scores = _read_prediction_value(predictions, "final_detection_scores")
        if scores is None:
            scores = _read_prediction_value(predictions, "scores")
        tensor = _as_tensor(scores)
        if tensor is None or tensor.numel() == 0:
            return None
        values = tensor.detach().float().flatten()
        if values.numel() == 0:
            return None
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return None
        confidence = float(finite.clamp(0.0, 1.0).mean().item())
        return max(0.0, min(1.0, confidence))

    def _compute_feature_entropy(self, boundary_payload: object) -> float | None:
        tensors = _payload_tensors(boundary_payload)
        weighted_sum = 0.0
        total_weight = 0
        for tensor in tensors:
            entropy, sample_count = self._tensor_activation_entropy(tensor)
            if entropy is None or sample_count <= 0:
                continue
            weighted_sum += float(entropy) * int(sample_count)
            total_weight += int(sample_count)
        if total_weight <= 0:
            return None
        return float(weighted_sum / float(total_weight))

    def _tensor_activation_entropy(self, tensor: torch.Tensor) -> tuple[float | None, int]:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None, 0
        if tensor.device.type == "cpu":
            return self._cpu_tensor_activation_entropy(tensor)
        flat = tensor.detach().float().abs().flatten()
        if flat.numel() > self.feature_max_elements:
            indices = (
                torch.linspace(
                    0,
                    flat.numel() - 1,
                    steps=self.feature_max_elements,
                    dtype=torch.float64,
                    device=flat.device,
                )
                .round()
                .long()
                .unique(sorted=True)
            )
            flat = flat.index_select(0, indices)
        sample_count = int(flat.numel())
        if sample_count <= 1:
            return 0.0, sample_count
        total = float(flat.sum().item())
        if total <= self.eps:
            return 0.0, sample_count
        probs = flat / total
        entropy = -(probs * torch.log(probs.clamp_min(float(self.eps)))).sum()
        entropy = entropy / max(math.log(sample_count), self.eps)
        return float(entropy.item()), sample_count

    def _cpu_tensor_activation_entropy(
        self,
        tensor: torch.Tensor,
    ) -> tuple[float | None, int]:
        detached = tensor.detach()
        try:
            array = detached.numpy()
        except TypeError:
            array = detached.float().numpy()
        value_count = int(array.size)
        if value_count > self.feature_max_elements:
            indices = self._feature_sample_indices.get(value_count)
            if indices is None:
                indices = np.rint(
                    np.linspace(
                        0,
                        value_count - 1,
                        num=self.feature_max_elements,
                        dtype=np.float64,
                    )
                ).astype(np.int64)
                indices.setflags(write=False)
                self._feature_sample_indices[value_count] = indices
            flat = array.flat[indices]
        else:
            flat = array.reshape(-1)
        values = np.abs(flat.astype(np.float32, copy=False))
        sample_count = int(values.size)
        if sample_count <= 1:
            return 0.0, sample_count
        total = float(values.sum())
        if total <= self.eps:
            return 0.0, sample_count
        probabilities = values / total
        log_probabilities = np.log(np.maximum(probabilities, float(self.eps)))
        entropy = -np.dot(probabilities, log_probabilities)
        entropy = float(entropy) / max(math.log(sample_count), self.eps)
        return entropy, sample_count

    @staticmethod
    def _predictions_empty(predictions: object) -> bool:
        sentinel = object()
        boxes = _read_prediction_value(predictions, "final_detection_boxes", default=sentinel)
        if boxes is sentinel:
            boxes = _read_prediction_value(predictions, "boxes", default=sentinel)
        if boxes is sentinel:
            return False
        if boxes is None:
            return True
        if isinstance(boxes, torch.Tensor):
            return boxes.numel() == 0 or (boxes.ndim > 0 and int(boxes.shape[0]) == 0)
        try:
            return len(list(boxes)) == 0
        except TypeError:
            return False


def _read_prediction_value(
    predictions: object,
    name: str,
    *,
    default: object | None = None,
) -> object:
    if isinstance(predictions, Mapping):
        return predictions.get(name, default)
    return getattr(predictions, name, default)


def _payload_tensors(boundary_payload: object) -> list[torch.Tensor]:
    if isinstance(boundary_payload, BoundaryPayload):
        source = getattr(boundary_payload, "tensors", {}) or {}
    elif isinstance(boundary_payload, torch.Tensor):
        source = {"payload": boundary_payload}
    elif isinstance(boundary_payload, Mapping):
        source = boundary_payload.get("tensors", boundary_payload)
    else:
        return []
    return [tensor for tensor in dict(source or {}).values() if isinstance(tensor, torch.Tensor)]


__all__ = [
    "HIGH_QUALITY",
    "LOW_QUALITY",
    "QUALITY_METHOD",
    "EntropyQualityStats",
    "EntropyQualityClassifier",
]
