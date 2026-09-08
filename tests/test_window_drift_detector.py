from __future__ import annotations

from edge.sample_quality import HIGH_QUALITY, LOW_QUALITY, EntropyQualityStats
from edge.window_drift_detector import WindowDriftDetector


def _quality(bucket: str, *, confidence: float = 0.9) -> EntropyQualityStats:
    trusted = bucket == HIGH_QUALITY
    return EntropyQualityStats(
        output_entropy=0.1,
        output_entropy_threshold=0.2,
        output_confidence=confidence,
        output_confidence_threshold=0.85,
        output_confident=True,
        feature_entropy=0.5,
        feature_entropy_mean=0.5,
        feature_entropy_std=0.1,
        feature_entropy_deviation=0.0,
        feature_deviation_threshold=1.5,
        output_reliable=trusted,
        feature_normal=trusted,
        edge_pseudo_label_trusted=trusted,
        quality=bucket,
        reason="test",
    )


def test_persistence_counts_spaced_evaluations_not_adjacent_frames() -> None:
    detector = WindowDriftDetector(
        window_size=4,
        min_window_size=2,
        low_quality_rate_threshold=0.5,
        persistence_windows=2,
        evaluation_stride=2,
    )

    first = detector.update(_quality(LOW_QUALITY))
    second = detector.update(_quality(LOW_QUALITY))
    third = detector.update(_quality(LOW_QUALITY))
    fourth = detector.update(_quality(LOW_QUALITY))

    assert first.evaluation_performed is False
    assert second.evaluation_performed is True and second.drift_detected is False
    assert third.evaluation_performed is False and third.drift_detected is False
    assert fourth.evaluation_performed is True
    assert fourth.drift_started is True
    assert fourth.drift_detected is True


def test_recovery_uses_lower_threshold_and_persistence() -> None:
    detector = WindowDriftDetector(
        window_size=2,
        min_window_size=2,
        low_quality_rate_threshold=0.5,
        persistence_windows=1,
        evaluation_stride=2,
        recovery_rate_threshold=0.0,
        recovery_windows=2,
    )
    detector.update(_quality(LOW_QUALITY))
    active = detector.update(_quality(LOW_QUALITY))
    detector.update(_quality(HIGH_QUALITY))
    first_recovery = detector.update(_quality(HIGH_QUALITY))
    between = detector.update(_quality(HIGH_QUALITY))
    recovered = detector.update(_quality(HIGH_QUALITY))

    assert active.drift_detected is True
    assert first_recovery.drift_detected is True
    assert between.evaluation_performed is False and between.drift_detected is True
    assert recovered.drift_ended is True
    assert recovered.drift_detected is False


def test_severe_anomaly_channel_detects_precision_risk_below_rate_threshold() -> None:
    detector = WindowDriftDetector(
        window_size=4,
        min_window_size=4,
        low_quality_rate_threshold=0.8,
        persistence_windows=1,
        evaluation_stride=4,
        severe_anomaly_rate_threshold=0.25,
        critical_confidence=0.4,
    )
    detector.update(_quality(HIGH_QUALITY))
    detector.update(_quality(HIGH_QUALITY))
    detector.update(_quality(HIGH_QUALITY))
    state = detector.update(_quality(LOW_QUALITY, confidence=0.1))

    assert state.low_quality_rate == 0.25
    assert state.severe_anomaly_rate == 0.25
    assert state.drift_started is True
    assert "persistent_severe_anomaly_rate" in state.drift_reasons
