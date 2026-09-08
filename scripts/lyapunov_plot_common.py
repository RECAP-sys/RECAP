from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_runtime_config
from edge.resource_aware_trigger import create_resource_aware_trigger

FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

ACTION_NAMES = ("skip_training", "train_raw_only", "train_raw_plus_feature")
ACTION_TICK_LABELS = (r"$a_0$", r"$a_1$", r"$a_2$")
ACTION_LEGEND_LABELS = (
    "skip training",
    "train raw-only",
    "train raw+feature",
)


def budget_line_specs(trigger) -> tuple[tuple[float, str], ...]:
    cloud_budget = float(trigger.lambda_cloud)
    bandwidth_budget = float(trigger.lambda_bw)
    if np.isclose(cloud_budget, bandwidth_budget):
        return ((cloud_budget, r"Budgets $\lambda_c=\lambda_b$"),)
    return (
        (cloud_budget, r"Cloud budget $\lambda_c$"),
        (bandwidth_budget, r"Bandwidth budget $\lambda_b$"),
    )


def build_trigger(config_path: Path = CONFIG_PATH):
    return create_resource_aware_trigger(load_runtime_config(config_path).client)


def representative_payloads() -> dict[str, float]:
    # Normalized payload units for paper figures:
    # always_sent = high-quality features + low-quality raw samples.
    # low_quality_feature is the optional payload controlled by the trigger.
    return {
        "always_sent_bytes": 20.0,
        "low_quality_feature_bytes": 6.0,
    }


def low_conf_feature_ratio(payloads: dict[str, float]) -> float:
    raw_plus_feature = float(payloads["always_sent_bytes"]) + float(
        payloads["low_quality_feature_bytes"]
    )
    return float(payloads["low_quality_feature_bytes"]) / max(raw_plus_feature, 1.0)


def raw_plus_feature_pressure(
    raw_only_pressure: Any,
    payloads: dict[str, float],
) -> Any:
    raw_only_payload = max(float(payloads["always_sent_bytes"]), 1.0)
    raw_plus_feature_payload = raw_only_payload + float(payloads["low_quality_feature_bytes"])
    return np.minimum(1.0, (raw_plus_feature_payload / raw_only_payload) * raw_only_pressure)


def compute_action_scores(
    *,
    trigger,
    urgency: Any,
    compute_pressure: Any,
    raw_only_bw_pressure: Any,
    raw_plus_feature_bw_pressure: Any,
    feature_ratio: float,
    q_cloud: float | Any = 0.0,
    q_bw: float | Any = 0.0,
    training_enabled: bool = True,
) -> dict[str, Any]:
    cloud_factor = trigger.feature_cloud_cost_factor
    skip_score = trigger.V * urgency
    raw_only_score = (
        trigger.w_cloud * q_cloud * compute_pressure
        + trigger.w_bw * q_bw * raw_only_bw_pressure
        + trigger.w_cloud * compute_pressure * (1.0 + compute_pressure)
        + trigger.w_bw * raw_only_bw_pressure * (1.0 + raw_only_bw_pressure)
        + compute_pressure * feature_ratio
    )
    raw_plus_feature_score = (
        trigger.w_cloud * q_cloud * cloud_factor * compute_pressure
        + trigger.w_bw * q_bw * raw_plus_feature_bw_pressure
        + trigger.w_cloud
        * compute_pressure
        * (1.0 + cloud_factor * compute_pressure)
        + trigger.w_bw
        * raw_plus_feature_bw_pressure
        * (1.0 + raw_plus_feature_bw_pressure)
        + (1.0 + raw_plus_feature_bw_pressure) * feature_ratio
    )
    if not training_enabled:
        raw_only_score = np.full_like(skip_score, np.inf, dtype=float)
        raw_plus_feature_score = np.full_like(skip_score, np.inf, dtype=float)

    return {
        "skip_training": skip_score,
        "train_raw_only": raw_only_score,
        "train_raw_plus_feature": raw_plus_feature_score,
    }


def decision_region(
    *,
    trigger,
    urgency: Any,
    compute_pressure: Any,
    raw_only_bw_pressure: Any,
    payloads: dict[str, float] | None = None,
    q_cloud: float | Any = 0.0,
    q_bw: float | Any = 0.0,
    training_enabled: bool = True,
) -> Any:
    active_payloads = payloads or representative_payloads()
    scores = compute_action_scores(
        trigger=trigger,
        urgency=urgency,
        compute_pressure=compute_pressure,
        raw_only_bw_pressure=raw_only_bw_pressure,
        raw_plus_feature_bw_pressure=raw_plus_feature_pressure(
            raw_only_bw_pressure,
            active_payloads,
        ),
        feature_ratio=low_conf_feature_ratio(active_payloads),
        q_cloud=q_cloud,
        q_bw=q_bw,
        training_enabled=training_enabled,
    )
    stacked = np.stack([scores[name] for name in ACTION_NAMES], axis=0)
    return np.argmin(stacked, axis=0)


def figure_note(trigger, payloads: dict[str, float]) -> str:
    payload_scale = (
        payloads["always_sent_bytes"] + payloads["low_quality_feature_bytes"]
    ) / payloads["always_sent_bytes"]
    return (
        f"$V={trigger.V:g}$, $\\lambda_c={trigger.lambda_cloud:g}$, "
        f"$\\lambda_b={trigger.lambda_bw:g}$, feature ratio="
        f"{low_conf_feature_ratio(payloads):.2f}, payload scale="
        f"{payload_scale:.2f}"
    )
