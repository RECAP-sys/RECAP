#!/usr/bin/env python3
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from experiments.drift_detection_validity.detection_metrics import normalize_prediction


def _cfg(config: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key)
    return default if value is None else value


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _std(values: Sequence[float], eps: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float(eps)
    return float(max(np.std(np.asarray(finite, dtype=np.float64)), eps))


def ema_update(previous: float | None, value: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    if previous is None or not math.isfinite(float(previous)):
        return float(value)
    if not math.isfinite(float(value)):
        return float(previous)
    return float(((1.0 - alpha) * float(previous)) + (alpha * float(value)))


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def _payload_tensors(boundary_payload: Any) -> dict[str, torch.Tensor]:
    if boundary_payload is None:
        return {}
    source: Any
    if isinstance(boundary_payload, torch.Tensor):
        source = {"payload": boundary_payload}
    elif isinstance(boundary_payload, Mapping):
        source = boundary_payload.get("tensors", boundary_payload)
    else:
        source = getattr(boundary_payload, "tensors", None)
    if not isinstance(source, Mapping):
        return {}
    return {
        str(key): value
        for key, value in dict(source).items()
        if isinstance(value, torch.Tensor)
    }


def _channel_axis(array: np.ndarray) -> int | None:
    if array.ndim < 2:
        return None
    if array.ndim >= 4 and 1 < int(array.shape[1]) <= 256:
        return 1
    if array.ndim == 3 and 1 < int(array.shape[-1]) <= 256:
        return 2
    if array.ndim == 3 and 1 < int(array.shape[0]) <= 256:
        return 0
    if array.ndim == 2 and 1 < int(array.shape[-1]) <= 256:
        return 1
    return None


def boundary_feature_vector(boundary_payload: Any) -> np.ndarray:
    """Return compact z_t statistics from boundary tensors.

    Per tensor, this concatenates mean, std, min, max, l2 norm, and feasible
    channel-wise means/stds. Tensor names are processed in sorted order.
    """

    features: list[float] = []
    for name, tensor in sorted(_payload_tensors(boundary_payload).items()):
        del name
        array = _as_numpy(tensor)
        if array is None or array.size == 0:
            continue
        values = array.astype(np.float64, copy=False)
        flat = values.reshape(-1)
        features.extend(
            [
                float(np.mean(flat)),
                float(np.std(flat)),
                float(np.min(flat)),
                float(np.max(flat)),
                float(np.linalg.norm(flat)),
            ]
        )
        axis = _channel_axis(values)
        if axis is None:
            continue
        moved = np.moveaxis(values, axis, -1).reshape(-1, values.shape[axis])
        channel_count = int(moved.shape[-1])
        if channel_count <= 0 or channel_count > 256:
            continue
        features.extend(float(value) for value in np.mean(moved, axis=0).tolist())
        features.extend(float(value) for value in np.std(moved, axis=0).tolist())
    return np.asarray(features, dtype=np.float64)


def _read_output_value(outputs: Any, keys: Sequence[str]) -> Any:
    if isinstance(outputs, Mapping):
        for key in keys:
            if key in outputs:
                return outputs.get(key)
    for key in keys:
        if hasattr(outputs, key):
            return getattr(outputs, key)
    return None


def _tensor_rows(value: Any) -> np.ndarray | None:
    array = _as_numpy(value)
    if array is None or array.size == 0:
        return None
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    return array.reshape(-1, array.shape[-1])


def _entropy_from_logits(
    outputs: Any,
    config: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    eps = float(_cfg(config, "signals", "eps", default=1.0e-8))
    topk = max(1, int(_cfg(config, "signals", "topk_queries", default=100)))
    for key in (
        "query_logits",
        "pred_logits",
        "dense_logits",
        "pre_nms_logits",
        "logits",
        "cls_logits",
    ):
        rows = _tensor_rows(_read_output_value(outputs, (key,)))
        if rows is None or rows.shape[-1] <= 1:
            continue
        work = rows.astype(np.float64, copy=False)
        work = work - np.max(work, axis=1, keepdims=True)
        probabilities = np.exp(work)
        probabilities = probabilities / np.maximum(
            probabilities.sum(axis=1, keepdims=True),
            eps,
        )
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, eps)), axis=1)
        entropy = entropy / max(math.log(max(2, probabilities.shape[1])), eps)
        objectness = 1.0 - probabilities[:, -1]
        if topk < len(entropy):
            keep = np.argsort(-objectness)[:topk]
            entropy = entropy[keep]
            objectness = objectness[keep]
        weighted = float(
            np.sum(np.maximum(objectness, 0.0) * entropy)
            / max(float(np.sum(np.maximum(objectness, 0.0))), eps)
        )
        return float(np.mean(entropy)), weighted
    return None, None


def _entropy_from_scores(outputs: Any, eps: float) -> float | None:
    prediction = normalize_prediction(outputs)
    scores = prediction["scores"]
    if scores.size == 0:
        return 0.0
    p = np.clip(scores.astype(np.float64, copy=False), eps, 1.0 - eps)
    entropy = -((p * np.log(p)) + ((1.0 - p) * np.log(1.0 - p))) / math.log(2.0)
    return float(np.mean(entropy))


def output_entropies(outputs: Any, config: Mapping[str, Any]) -> tuple[float, float]:
    eps = float(_cfg(config, "signals", "eps", default=1.0e-8))
    direct_entropy = _read_output_value(outputs, ("output_entropy", "logit_entropy"))
    direct_weighted = _read_output_value(outputs, ("objectness_weighted_entropy",))
    entropy = None if direct_entropy is None else _finite(direct_entropy, math.nan)
    weighted = None if direct_weighted is None else _finite(direct_weighted, math.nan)
    if entropy is None or not math.isfinite(entropy):
        entropy, weighted_from_logits = _entropy_from_logits(outputs, config)
        if weighted is None:
            weighted = weighted_from_logits
    if entropy is None or not math.isfinite(entropy):
        entropy = _entropy_from_scores(outputs, eps)
    if weighted is None or not math.isfinite(float(weighted)):
        weighted = entropy
    return float(entropy or 0.0), float(weighted or 0.0)


class DriftSignalExtractor:
    def __init__(
        self,
        config: Mapping[str, Any],
        student_model: Any,
        split_runtime: Any | None = None,
    ) -> None:
        self.config = config
        self.student_model = student_model
        self.split_runtime = split_runtime

    @torch.no_grad()
    def extract(
        self,
        frame: np.ndarray,
        student_outputs: Any,
        boundary_payload: Any | None = None,
    ) -> dict[str, Any]:
        del frame
        prediction = normalize_prediction(student_outputs)
        scores = prediction["scores"]
        mean_confidence = float(np.mean(scores)) if scores.size else 0.0
        entropy, weighted_entropy = output_entropies(student_outputs, self.config)
        vector = boundary_feature_vector(boundary_payload)
        boundary_mean = float(np.mean(vector)) if vector.size else 0.0
        boundary_std = float(np.std(vector)) if vector.size else 0.0
        boundary_l2 = float(np.linalg.norm(vector)) if vector.size else 0.0
        return {
            "mean_confidence": mean_confidence,
            "confidence_drop_signal": 0.0,
            "confidence_drop_z": 0.0,
            "output_entropy": float(entropy),
            "objectness_weighted_entropy": float(weighted_entropy),
            "ema_output_entropy": float(entropy),
            "ema_output_entropy_z": 0.0,
            "boundary_feature_mean": boundary_mean,
            "boundary_feature_std": boundary_std,
            "boundary_feature_l2_norm": boundary_l2,
            "boundary_feature_deviation": 0.0,
            "boundary_feature_deviation_z": 0.0,
            "ema_boundary_feature_deviation": 0.0,
            "ema_boundary_feature_deviation_z": 0.0,
            "full_drift_score": 0.0,
            "full_drift_score_z": 0.0,
            "_boundary_feature_vector": vector,
        }


def clean_baseline_mask(records: Sequence[Mapping[str, Any]]) -> list[bool]:
    if not records:
        return []
    first_domain_index = int(records[0].get("domain_index", 0))
    first_domain = str(records[0].get("domain", "clean"))
    return [
        int(record.get("domain_index", -1)) == first_domain_index
        and str(record.get("domain", "")) == first_domain
        for record in records
    ]


def _stack_vectors(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    vectors = [
        np.asarray(record.get("_boundary_feature_vector", []), dtype=np.float64)
        for record in records
    ]
    vectors = [vector for vector in vectors if vector.size]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64)
    width = int(vectors[0].size)
    consistent = [vector for vector in vectors if int(vector.size) == width]
    if not consistent:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(consistent, axis=0)


def compute_clean_baseline(
    records: Sequence[Mapping[str, Any]],
    baseline_mask: Sequence[bool] | None = None,
    *,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    mask = list(baseline_mask) if baseline_mask is not None else clean_baseline_mask(records)
    baseline_records = [record for record, keep in zip(records, mask) if keep]
    if not baseline_records:
        raise ValueError("No clean baseline records were available for drift normalization.")
    confidences = [_finite(record.get("mean_confidence")) for record in baseline_records]
    entropies = [_finite(record.get("output_entropy")) for record in baseline_records]
    vectors = _stack_vectors(baseline_records)
    if vectors.size:
        mu = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - mu[None, :], axis=1)
        feature_sigma = float(max(np.std(distances), eps))
        feature_mean = float(np.mean(distances / feature_sigma))
        feature_std = float(max(np.std(distances / feature_sigma), eps))
    else:
        mu = np.zeros((0,), dtype=np.float64)
        feature_sigma = float(eps)
        feature_mean = 0.0
        feature_std = float(eps)
    clean_conf_mean = float(np.mean(confidences))
    confidence_drops = [clean_conf_mean - value for value in confidences]
    return {
        "mean_confidence": clean_conf_mean,
        "std_confidence": _std(confidences, eps),
        "mean_confidence_drop_signal": float(np.mean(confidence_drops)),
        "std_confidence_drop_signal": _std(confidence_drops, eps),
        "mean_output_entropy": float(np.mean(entropies)),
        "std_output_entropy": _std(entropies, eps),
        "boundary_mu": mu,
        "boundary_sigma": feature_sigma,
        "mean_boundary_feature_deviation": feature_mean,
        "std_boundary_feature_deviation": feature_std,
    }


def _zscore(
    value: float,
    mean: float,
    std: float,
    eps: float,
    *,
    clip: float | None = None,
) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    score = float((float(value) - float(mean)) / (float(std) + float(eps)))
    if clip is not None and math.isfinite(float(clip)) and float(clip) > 0.0:
        limit = float(clip)
        score = max(-limit, min(limit, score))
    return score


def finalize_signal_records(
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
    baseline_mask: Sequence[bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eps = float(_cfg(config, "signals", "eps", default=1.0e-8))
    std_floor = max(
        eps,
        float(_cfg(config, "signals", "zscore_std_floor", default=eps)),
    )
    z_clip_value = _cfg(config, "signals", "zscore_clip", default=None)
    z_clip = None if z_clip_value is None else float(z_clip_value)
    alpha = float(_cfg(config, "signals", "ema_alpha", default=0.05))
    entropy_weight = float(_cfg(config, "signals", "full_score_entropy_weight", default=0.5))
    feature_weight = float(_cfg(config, "signals", "full_score_feature_weight", default=0.5))
    mask = list(baseline_mask) if baseline_mask is not None else clean_baseline_mask(records)
    baseline = compute_clean_baseline(records, mask, eps=std_floor)

    for record in records:
        vector = np.asarray(record.get("_boundary_feature_vector", []), dtype=np.float64)
        mu = np.asarray(baseline["boundary_mu"], dtype=np.float64)
        if vector.size and mu.size and vector.size == mu.size:
            deviation = float(np.linalg.norm(vector - mu) / float(baseline["boundary_sigma"]))
        else:
            deviation = 0.0
        record["boundary_feature_deviation"] = deviation
        record["confidence_drop_signal"] = float(
            baseline["mean_confidence"] - _finite(record.get("mean_confidence"))
        )
        record["confidence_drop_z"] = _zscore(
            record["confidence_drop_signal"],
            baseline["mean_confidence_drop_signal"],
            baseline["std_confidence_drop_signal"],
            eps,
            clip=z_clip,
        )
        record["boundary_feature_deviation_z"] = _zscore(
            record["boundary_feature_deviation"],
            baseline["mean_boundary_feature_deviation"],
            baseline["std_boundary_feature_deviation"],
            eps,
            clip=z_clip,
        )

    ema_entropy: float | None = None
    ema_feature: float | None = None
    for record in records:
        ema_entropy = ema_update(ema_entropy, _finite(record.get("output_entropy")), alpha)
        ema_feature = ema_update(
            ema_feature,
            _finite(record.get("boundary_feature_deviation")),
            alpha,
        )
        record["ema_output_entropy"] = ema_entropy
        record["ema_boundary_feature_deviation"] = ema_feature

    baseline_ema_entropy = [
        float(record["ema_output_entropy"]) for record, keep in zip(records, mask) if keep
    ]
    baseline_ema_feature = [
        float(record["ema_boundary_feature_deviation"])
        for record, keep in zip(records, mask)
        if keep
    ]
    ema_entropy_mean = float(np.mean(baseline_ema_entropy)) if baseline_ema_entropy else 0.0
    ema_feature_mean = float(np.mean(baseline_ema_feature)) if baseline_ema_feature else 0.0
    ema_entropy_std = _std(baseline_ema_entropy, std_floor)
    ema_feature_std = _std(baseline_ema_feature, std_floor)

    for record in records:
        h_norm = _zscore(
            record["ema_output_entropy"],
            ema_entropy_mean,
            ema_entropy_std,
            eps,
            clip=z_clip,
        )
        d_norm = _zscore(
            record["ema_boundary_feature_deviation"],
            ema_feature_mean,
            ema_feature_std,
            eps,
            clip=z_clip,
        )
        record["ema_output_entropy_z"] = h_norm
        record["ema_boundary_feature_deviation_z"] = d_norm
        record["full_drift_score"] = float((entropy_weight * h_norm) + (feature_weight * d_norm))

    clean_scores = [
        float(record["full_drift_score"]) for record, keep in zip(records, mask) if keep
    ]
    full_mean = float(np.mean(clean_scores)) if clean_scores else 0.0
    full_std = _std(clean_scores, std_floor)
    for record in records:
        record["full_drift_score_z"] = _zscore(
            record["full_drift_score"],
            full_mean,
            full_std,
            eps,
            clip=z_clip,
        )
        record.pop("_boundary_feature_vector", None)

    baseline.update(
        {
            "mean_ema_output_entropy": ema_entropy_mean,
            "std_ema_output_entropy": ema_entropy_std,
            "mean_ema_boundary_feature_deviation": ema_feature_mean,
            "std_ema_boundary_feature_deviation": ema_feature_std,
            "mean_full_drift_score": full_mean,
            "std_full_drift_score": full_std,
        }
    )
    return records, baseline


__all__ = [
    "DriftSignalExtractor",
    "boundary_feature_vector",
    "clean_baseline_mask",
    "compute_clean_baseline",
    "ema_update",
    "finalize_signal_records",
    "output_entropies",
]
