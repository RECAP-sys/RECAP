from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from edge.sample_quality import LOW_QUALITY, EntropyQualityStats


@dataclass
class DriftWindowState:
    window_id: str
    drift_detected: bool
    drift_score: float
    low_quality_rate: float
    drift_reasons: list[str]
    severe_anomaly_rate: float = 0.0
    evaluation_performed: bool = False
    drift_started: bool = False
    drift_ended: bool = False
    observation_count: int = 0


class WindowDriftDetector:
    """Windowed drift detector with spaced evaluations and hysteresis.

    ``persistence_windows`` counts evaluation checkpoints, not adjacent frames.
    This avoids treating nearly identical sliding windows as independent drift
    evidence. Once active, a lower exit threshold and recovery persistence keep
    the state from oscillating near the boundary.
    """

    def __init__(
        self,
        *,
        window_size: int = 100,
        min_window_size: int = 30,
        low_quality_rate_threshold: float = 0.3,
        persistence_windows: int = 3,
        evaluation_stride: int = 1,
        recovery_rate_threshold: float | None = None,
        recovery_windows: int = 2,
        severe_anomaly_rate_threshold: float = 0.20,
        recovery_severe_anomaly_rate: float = 0.10,
        critical_confidence: float = 0.40,
        critical_output_entropy_ratio: float = 1.50,
        critical_feature_deviation: float = 6.0,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.min_window_size = max(1, min(int(min_window_size), self.window_size))
        self.low_quality_rate_threshold = max(
            0.0,
            min(1.0, float(low_quality_rate_threshold)),
        )
        self.persistence_windows = max(1, int(persistence_windows))
        self.evaluation_stride = max(1, int(evaluation_stride))
        default_recovery = self.low_quality_rate_threshold * 0.8
        resolved_recovery = (
            default_recovery
            if recovery_rate_threshold is None
            else float(recovery_rate_threshold)
        )
        self.recovery_rate_threshold = max(
            0.0,
            min(
                self.low_quality_rate_threshold,
                resolved_recovery,
            ),
        )
        self.recovery_windows = max(1, int(recovery_windows))
        self.severe_anomaly_rate_threshold = max(
            0.0,
            min(1.0, float(severe_anomaly_rate_threshold)),
        )
        self.recovery_severe_anomaly_rate = max(
            0.0,
            min(self.severe_anomaly_rate_threshold, float(recovery_severe_anomaly_rate)),
        )
        self.critical_confidence = max(0.0, min(1.0, float(critical_confidence)))
        self.critical_output_entropy_ratio = max(
            1.0,
            float(critical_output_entropy_ratio),
        )
        self.critical_feature_deviation = max(0.0, float(critical_feature_deviation))
        self._records: deque[EntropyQualityStats] = deque(maxlen=self.window_size)
        self._abnormal_windows = 0
        self._recovery_windows = 0
        self._step = 0
        self._last_evaluation_step = 0
        self._drift_active = False
        self._last_low_quality_rate = 0.0
        self._last_severe_anomaly_rate = 0.0
        self._last_drift_score = 0.0
        self._last_reasons: list[str] = []

    def reset(self) -> None:
        self._records.clear()
        self._abnormal_windows = 0
        self._recovery_windows = 0
        self._step = 0
        self._last_evaluation_step = 0
        self._drift_active = False
        self._last_low_quality_rate = 0.0
        self._last_severe_anomaly_rate = 0.0
        self._last_drift_score = 0.0
        self._last_reasons = []

    def update(
        self,
        quality_record: EntropyQualityStats,
        feature_stats: dict[str, Any] | None = None,
        teacher_feedback: dict[str, Any] | None = None,
    ) -> DriftWindowState:
        del feature_stats, teacher_feedback
        self._step += 1
        self._records.append(quality_record)
        window_id = f"window-{max(1, self._step - len(self._records) + 1)}-{self._step}"
        evaluation_due = len(self._records) >= self.min_window_size and (
            self._last_evaluation_step == 0
            or self._step - self._last_evaluation_step >= self.evaluation_stride
        )
        state = (
            self._evaluate(window_id)
            if evaluation_due
            else self._current_state(window_id)
        )
        quality_record.window_id = state.window_id
        quality_record.in_drift_window = state.drift_detected
        return state

    def _is_severe_anomaly(self, quality: EntropyQualityStats) -> bool:
        confidence = quality.output_confidence
        if confidence is not None and float(confidence) < self.critical_confidence:
            return True
        entropy = quality.output_entropy
        threshold = quality.output_entropy_threshold
        if (
            entropy is not None
            and threshold is not None
            and float(threshold) > 0.0
            and float(entropy) / float(threshold) >= self.critical_output_entropy_ratio
        ):
            return True
        deviation = quality.feature_entropy_deviation
        return bool(
            deviation is not None
            and float(deviation) >= self.critical_feature_deviation
        )

    def _window_rates(self) -> tuple[float, float]:
        records = list(self._records)
        if not records:
            return 0.0, 0.0
        low_count = sum(1 for item in records if item.quality_bucket == LOW_QUALITY)
        severe_count = sum(1 for item in records if self._is_severe_anomaly(item))
        return low_count / float(len(records)), severe_count / float(len(records))

    def _current_state(self, window_id: str) -> DriftWindowState:
        return DriftWindowState(
            window_id=window_id,
            drift_detected=bool(self._drift_active),
            drift_score=float(self._last_drift_score),
            low_quality_rate=float(self._last_low_quality_rate),
            drift_reasons=list(self._last_reasons) if self._drift_active else [],
            severe_anomaly_rate=float(self._last_severe_anomaly_rate),
            evaluation_performed=False,
            observation_count=int(self._step),
        )

    def _evaluate(self, window_id: str) -> DriftWindowState:
        self._last_evaluation_step = self._step
        low_quality_rate, severe_anomaly_rate = self._window_rates()
        abnormal_low_quality = low_quality_rate >= self.low_quality_rate_threshold
        abnormal_severe = severe_anomaly_rate >= self.severe_anomaly_rate_threshold
        abnormal = abnormal_low_quality or abnormal_severe
        drift_started = False
        drift_ended = False

        if self._drift_active:
            recovered = (
                low_quality_rate <= self.recovery_rate_threshold
                and severe_anomaly_rate <= self.recovery_severe_anomaly_rate
            )
            if recovered:
                self._recovery_windows += 1
            else:
                self._recovery_windows = 0
            if self._recovery_windows >= self.recovery_windows:
                self._drift_active = False
                self._abnormal_windows = 0
                self._recovery_windows = 0
                drift_ended = True
        else:
            if abnormal:
                self._abnormal_windows += 1
            else:
                self._abnormal_windows = 0
            if self._abnormal_windows >= self.persistence_windows:
                self._drift_active = True
                self._recovery_windows = 0
                drift_started = True

        low_quality_score = low_quality_rate / max(self.low_quality_rate_threshold, 1e-6)
        severe_score = severe_anomaly_rate / max(
            self.severe_anomaly_rate_threshold,
            1e-6,
        )
        drift_score = max(low_quality_score, severe_score)
        reasons: list[str] = []
        if self._drift_active and abnormal_low_quality:
            reasons.append("persistent_low_quality_rate")
        if self._drift_active and abnormal_severe:
            reasons.append("persistent_severe_anomaly_rate")
        if self._drift_active and not reasons:
            reasons.append("hysteresis_hold")
        self._last_low_quality_rate = float(low_quality_rate)
        self._last_severe_anomaly_rate = float(severe_anomaly_rate)
        self._last_drift_score = float(max(0.0, min(1.0, drift_score)))
        self._last_reasons = list(reasons)
        return DriftWindowState(
            window_id=window_id,
            drift_detected=bool(self._drift_active),
            drift_score=self._last_drift_score,
            low_quality_rate=float(low_quality_rate),
            drift_reasons=reasons,
            severe_anomaly_rate=float(severe_anomaly_rate),
            evaluation_performed=True,
            drift_started=bool(drift_started),
            drift_ended=bool(drift_ended),
            observation_count=int(self._step),
        )
