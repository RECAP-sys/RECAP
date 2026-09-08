from __future__ import annotations

from edge.sample_quality import HIGH_QUALITY, LOW_QUALITY, EntropyQualityStats
from edge.teacher_sample_selector import LowQualityTeacherSampler


def _quality(
    bucket: str = LOW_QUALITY,
    *,
    confidence: float = 0.8,
    entropy: float = 0.3,
    entropy_threshold: float = 0.25,
    feature_deviation: float = 2.0,
    reason: str = "output_entropy_high",
) -> EntropyQualityStats:
    trusted = bucket == HIGH_QUALITY
    return EntropyQualityStats(
        output_entropy=entropy,
        output_entropy_threshold=entropy_threshold,
        output_confidence=confidence,
        output_confidence_threshold=0.85,
        output_confident=confidence >= 0.85,
        feature_entropy=0.5,
        feature_entropy_mean=0.5,
        feature_entropy_std=0.1,
        feature_entropy_deviation=feature_deviation,
        feature_deviation_threshold=1.5,
        output_reliable=trusted,
        feature_normal=trusted,
        edge_pseudo_label_trusted=trusted,
        quality=bucket,
        reason=reason,
    )


def test_sampler_temporally_thins_ordinary_low_quality_samples() -> None:
    sampler = LowQualityTeacherSampler(
        base_stride=4,
        critical_confidence=0.4,
        critical_output_entropy_ratio=2.0,
        critical_feature_deviation=6.0,
    )

    decisions = [sampler.select(_quality()) for _ in range(8)]

    assert [item.selected for item in decisions] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert sampler.effective_selection_rate == 0.25


def test_sampler_keeps_severe_anomalies_more_densely() -> None:
    sampler = LowQualityTeacherSampler(base_stride=100, critical_stride=2)

    first = sampler.select(_quality())
    critical_confidence = sampler.select(_quality(confidence=0.1))
    critical_entropy = sampler.select(_quality(entropy=0.5, entropy_threshold=0.25))
    critical_feature = sampler.select(_quality(feature_deviation=8.0))

    assert first.selected is True
    assert critical_confidence.selected is True and critical_confidence.critical is True
    assert critical_entropy.selected is False and critical_entropy.critical is True
    assert critical_feature.selected is True and critical_feature.critical is True


def test_sampler_never_requests_raw_for_trusted_samples() -> None:
    sampler = LowQualityTeacherSampler()

    decision = sampler.select(_quality(HIGH_QUALITY, confidence=0.99))

    assert decision.selected is False
    assert sampler.low_quality_seen == 0


def test_sampler_switches_from_bootstrap_to_sparse_stride_after_model_update() -> None:
    sampler = LowQualityTeacherSampler(
        initial_stride=1,
        base_stride=4,
        critical_output_entropy_ratio=2.0,
    )

    bootstrap = [sampler.select(_quality()) for _ in range(4)]
    sampler.on_model_update()
    steady_state = [sampler.select(_quality()) for _ in range(8)]

    assert [item.selected for item in bootstrap] == [True, True, True, True]
    assert [item.selected for item in steady_state] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert sampler.model_update_count == 1
    assert sampler.effective_selection_rate == 0.25


def test_sampler_does_not_thin_critical_samples_during_dense_bootstrap() -> None:
    sampler = LowQualityTeacherSampler(
        initial_stride=1,
        base_stride=4,
        critical_stride=2,
    )

    decisions = [sampler.select(_quality(confidence=0.1)) for _ in range(4)]

    assert all(item.selected and item.critical for item in decisions)
