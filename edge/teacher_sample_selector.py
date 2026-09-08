from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from edge.sample_quality import LOW_QUALITY, EntropyQualityStats


def _config_value(config: object | None, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class TeacherSampleSelection:
    selected: bool
    critical: bool
    reason: str
    low_quality_index: int


class LowQualityTeacherSampler:
    """Select a temporally sparse, severity-preserving teacher sample stream.

    Quality classification remains conservative and every observation still feeds
    drift detection. Severe anomalies use a denser stream than ordinary uncertain
    samples, while both streams are bounded to avoid uploading adjacent raw frames
    carrying nearly identical evidence.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        initial_stride: int | None = None,
        base_stride: int = 4,
        critical_stride: int = 2,
        critical_confidence: float = 0.40,
        critical_output_entropy_ratio: float = 1.50,
        critical_feature_deviation: float = 6.0,
        retain_empty_predictions: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_stride = max(1, int(base_stride))
        self.initial_stride = (
            self.base_stride
            if initial_stride is None
            else max(1, int(initial_stride))
        )
        self.critical_stride = max(1, int(critical_stride))
        self.critical_confidence = max(0.0, min(1.0, float(critical_confidence)))
        self.critical_output_entropy_ratio = max(
            1.0,
            float(critical_output_entropy_ratio),
        )
        self.critical_feature_deviation = max(0.0, float(critical_feature_deviation))
        self.retain_empty_predictions = bool(retain_empty_predictions)
        self.low_quality_seen = 0
        self.low_quality_selected = 0
        self.critical_seen = 0
        self.model_update_count = 0

    @classmethod
    def from_config(cls, sample_quality_config: object | None) -> "LowQualityTeacherSampler":
        config = _config_value(sample_quality_config, "teacher_sampling", None)
        initial_stride = _config_value(config, "initial_stride", None)
        return cls(
            enabled=bool(_config_value(config, "enabled", True)),
            initial_stride=(None if initial_stride is None else int(initial_stride)),
            base_stride=int(_config_value(config, "base_stride", 4)),
            critical_stride=int(_config_value(config, "critical_stride", 2)),
            critical_confidence=float(
                _config_value(config, "critical_confidence", 0.40)
            ),
            critical_output_entropy_ratio=float(
                _config_value(config, "critical_output_entropy_ratio", 1.50)
            ),
            critical_feature_deviation=float(
                _config_value(config, "critical_feature_deviation", 6.0)
            ),
            retain_empty_predictions=bool(
                _config_value(config, "retain_empty_predictions", False)
            ),
        )

    @property
    def effective_selection_rate(self) -> float:
        if self.low_quality_seen <= 0:
            return 0.0
        return self.low_quality_selected / float(self.low_quality_seen)

    def reset(self) -> None:
        self.low_quality_seen = 0
        self.low_quality_selected = 0
        self.critical_seen = 0

    def on_model_update(self) -> None:
        self.model_update_count += 1
        self.reset()

    @property
    def active_stride(self) -> int:
        return self.initial_stride if self.model_update_count == 0 else self.base_stride

    def select(self, quality: EntropyQualityStats) -> TeacherSampleSelection:
        if quality.quality_bucket != LOW_QUALITY:
            return TeacherSampleSelection(False, False, "trusted_feature_only", 0)

        self.low_quality_seen += 1
        index = int(self.low_quality_seen)
        if not self.enabled:
            self.low_quality_selected += 1
            return TeacherSampleSelection(True, False, "sampling_disabled", index)

        critical_reasons: list[str] = []
        confidence = quality.output_confidence
        if confidence is not None and float(confidence) < self.critical_confidence:
            critical_reasons.append("critical_confidence")

        entropy = quality.output_entropy
        entropy_threshold = quality.output_entropy_threshold
        if (
            entropy is not None
            and entropy_threshold is not None
            and float(entropy_threshold) > 0.0
            and float(entropy) / float(entropy_threshold)
            >= self.critical_output_entropy_ratio
        ):
            critical_reasons.append("critical_output_entropy")

        feature_deviation = quality.feature_entropy_deviation
        if (
            feature_deviation is not None
            and float(feature_deviation) >= self.critical_feature_deviation
        ):
            critical_reasons.append("critical_feature_deviation")

        if self.retain_empty_predictions and "empty_predictions" in quality.reason.split(";"):
            critical_reasons.append("empty_predictions")

        if critical_reasons:
            self.critical_seen += 1
            effective_critical_stride = min(self.critical_stride, self.active_stride)
            selected = (self.critical_seen - 1) % effective_critical_stride == 0
            if selected:
                self.low_quality_selected += 1
            return TeacherSampleSelection(
                selected,
                True,
                "+".join(critical_reasons)
                + ("" if selected else "+critical_stride_skip"),
                index,
            )

        if (index - 1) % self.active_stride == 0:
            self.low_quality_selected += 1
            return TeacherSampleSelection(True, False, "temporal_stride", index)

        return TeacherSampleSelection(False, False, "redundant_low_quality", index)
