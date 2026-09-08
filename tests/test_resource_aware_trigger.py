from __future__ import annotations

from types import SimpleNamespace

import pytest

from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    create_resource_aware_trigger,
)


def test_model_specific_sample_limits_override_defaults() -> None:
    config = SimpleNamespace(
        lightweight="rfdetr_nano",
        retrain=SimpleNamespace(collect_num=8),
        resource_aware_trigger=SimpleNamespace(
            min_training_samples=128,
            min_training_samples_by_model={"rfdetr_nano": 64},
            max_training_samples=128,
            max_training_samples_by_model={"rfdetr_nano": 64},
        ),
    )

    trigger = create_resource_aware_trigger(config)

    assert trigger.min_training_samples == 64
    assert trigger.max_training_samples == 64


def _cloud(pressure: float) -> CloudResourceState:
    return CloudResourceState(
        cpu_utilization=pressure,
        gpu_utilization=pressure,
        memory_utilization=pressure,
        train_queue_size=0,
        max_queue_size=1,
    )


def _stats(
    *,
    total_samples: int,
    low_quality_count: int,
    low_quality_rate: float,
    high_quality_feature_bytes: int = 0,
    low_quality_feature_bytes: int = 0,
    low_quality_raw_bytes: int = 0,
) -> PendingTrainingStats:
    return PendingTrainingStats(
        total_samples=total_samples,
        high_quality_count=max(0, total_samples - low_quality_count),
        low_quality_count=low_quality_count,
        low_quality_rate=low_quality_rate,
        drift_detected=False,
        high_quality_feature_bytes=high_quality_feature_bytes,
        low_quality_feature_bytes=low_quality_feature_bytes,
        low_quality_raw_bytes=low_quality_raw_bytes,
    )


def test_urgency_matches_documented_level_best_and_derivative_terms() -> None:
    trigger = ResourceAwareCLTrigger(
        K_p=0.6,
        K_d=0.2,
        min_training_samples=10,
        drift_bonus=0.2,
    )

    first = trigger._urgency(
        True,
        _stats(total_samples=10, low_quality_count=5, low_quality_rate=0.5),
    )
    second = trigger._urgency(
        True,
        _stats(total_samples=10, low_quality_count=10, low_quality_rate=1.0),
    )

    assert first == pytest.approx(0.84)
    assert second == pytest.approx(1.60)


def test_action_scores_equal_dpp_queue_terms_plus_bounded_regularizers() -> None:
    trigger = ResourceAwareCLTrigger(
        V=100.0,
        K_p=0.0,
        K_d=0.0,
        lambda_cloud=0.0,
        lambda_bw=0.0,
        w_cloud=1.0,
        w_bw=1.0,
        feature_cloud_cost_factor=0.5,
        min_training_samples=1,
        upload_time_budget_sec=1.0,
    )
    trigger.Q_cloud = 2.0
    trigger.Q_bw = 3.0

    decision = trigger.decide(
        drift_detected=False,
        cloud_state=_cloud(0.8),
        bandwidth_mbps=8.0,
        sample_stats=_stats(
            total_samples=10,
            low_quality_count=5,
            low_quality_rate=0.5,
            high_quality_feature_bytes=500_000,
            low_quality_feature_bytes=500_000,
        ),
    )

    assert decision.action_scores["train_raw_only"] == pytest.approx(5.69)
    assert decision.action_scores["train_raw_plus_feature"] == pytest.approx(7.92)
    assert decision.train_now is True
    assert decision.send_low_conf_features is False


def test_raw_plus_feature_uses_discounted_cloud_cost_in_queue_update() -> None:
    trigger = ResourceAwareCLTrigger(
        V=100.0,
        K_p=0.0,
        K_d=0.0,
        lambda_cloud=0.0,
        lambda_bw=0.0,
        w_cloud=1.0,
        w_bw=1.0,
        feature_cloud_cost_factor=0.5,
        min_training_samples=1,
        upload_time_budget_sec=1.0,
    )
    trigger.Q_cloud = 10.0

    decision = trigger.decide(
        drift_detected=False,
        cloud_state=_cloud(1.0),
        bandwidth_mbps=1_000.0,
        sample_stats=_stats(
            total_samples=1,
            low_quality_count=1,
            low_quality_rate=1.0,
            high_quality_feature_bytes=100,
            low_quality_feature_bytes=1,
        ),
    )
    trigger.confirm_trigger(decision)

    assert decision.train_now is True
    assert decision.send_low_conf_features is True
    assert trigger.Q_cloud == pytest.approx(10.5)


def test_training_minimum_counts_teacher_needed_samples_not_total_samples() -> None:
    trigger = ResourceAwareCLTrigger(V=100.0, min_training_samples=5)

    decision = trigger.decide(
        drift_detected=True,
        cloud_state=_cloud(0.0),
        bandwidth_mbps=1_000.0,
        sample_stats=_stats(
            total_samples=100,
            low_quality_count=4,
            low_quality_rate=0.04,
        ),
    )

    assert decision.train_now is False
    assert "teacher-needed samples" in decision.reason


def test_post_trigger_decision_cooldown_suppresses_immediate_retraining() -> None:
    trigger = ResourceAwareCLTrigger(
        V=100.0,
        K_p=0.0,
        K_d=0.0,
        min_training_samples=1,
        cooldown_decisions=2,
    )
    stats = _stats(total_samples=1, low_quality_count=1, low_quality_rate=1.0)

    first = trigger.decide(
        drift_detected=True,
        cloud_state=_cloud(0.0),
        bandwidth_mbps=1_000.0,
        sample_stats=stats,
    )
    trigger.confirm_trigger(first)
    second = trigger.decide(
        drift_detected=True,
        cloud_state=_cloud(0.0),
        bandwidth_mbps=1_000.0,
        sample_stats=stats,
    )
    third = trigger.decide(
        drift_detected=True,
        cloud_state=_cloud(0.0),
        bandwidth_mbps=1_000.0,
        sample_stats=stats,
    )

    assert first.train_now is True
    assert second.train_now is False
    assert second.cooldown_remaining_decisions == 1
    assert third.train_now is True


def test_cancelled_trigger_does_not_arm_cooldown_or_charge_resources() -> None:
    trigger = ResourceAwareCLTrigger(
        V=100.0,
        K_p=0.0,
        K_d=0.0,
        lambda_cloud=0.0,
        lambda_bw=0.0,
        min_training_samples=1,
        cooldown_decisions=10,
        min_training_interval_sec=60.0,
    )
    stats = _stats(total_samples=1, low_quality_count=1, low_quality_rate=1.0)

    failed = trigger.decide(
        drift_detected=True,
        cloud_state=_cloud(0.5),
        bandwidth_mbps=1_000.0,
        sample_stats=stats,
    )
    trigger.cancel_trigger(failed)
    retry = trigger.decide(
        drift_detected=True,
        cloud_state=_cloud(0.5),
        bandwidth_mbps=1_000.0,
        sample_stats=stats,
    )

    assert retry.train_now is True
    assert retry.cooldown_remaining_decisions == 0
    assert retry.cooldown_remaining_sec == 0.0
    assert trigger.trigger_count == 0
    assert trigger.Q_cloud == 0.0
    assert trigger.Q_bw == 0.0
