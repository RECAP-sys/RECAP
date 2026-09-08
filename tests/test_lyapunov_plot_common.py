from types import SimpleNamespace

import pytest

from scripts.lyapunov_plot_common import budget_line_specs, compute_action_scores


def test_plot_scores_match_documented_drift_plus_penalty_equations() -> None:
    trigger = SimpleNamespace(
        V=100.0,
        w_cloud=1.0,
        w_bw=1.0,
        feature_cloud_cost_factor=0.5,
    )

    scores = compute_action_scores(
        trigger=trigger,
        urgency=0.5,
        compute_pressure=0.8,
        raw_only_bw_pressure=0.5,
        raw_plus_feature_bw_pressure=1.0,
        feature_ratio=0.5,
        q_cloud=2.0,
        q_bw=3.0,
    )

    assert scores["skip_training"] == pytest.approx(50.0)
    assert scores["train_raw_only"] == pytest.approx(5.69)
    assert scores["train_raw_plus_feature"] == pytest.approx(7.92)


def test_budget_line_specs_preserve_independent_resource_budgets() -> None:
    trigger = SimpleNamespace(lambda_cloud=0.4, lambda_bw=0.7)

    assert budget_line_specs(trigger) == (
        (0.4, r"Cloud budget $\lambda_c$"),
        (0.7, r"Bandwidth budget $\lambda_b$"),
    )


def test_budget_line_specs_merge_equal_resource_budgets() -> None:
    trigger = SimpleNamespace(lambda_cloud=0.5, lambda_bw=0.5)

    assert budget_line_specs(trigger) == ((0.5, r"Budgets $\lambda_c=\lambda_b$"),)
