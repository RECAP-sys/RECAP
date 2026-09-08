#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import transforms  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.drift_detection_validity.experiment_io import (  # noqa: E402
    load_config,
    output_dir,
    require_float,
    require_mapping,
)
from experiments.drift_detection_validity.figure_style import (  # noqa: E402
    PALETTE,
    SIGNAL_LABELS,
    add_panel_label,
    apply_publication_style,
    display_values,
    polish_axis,
    save_figure,
)

REAL_WEATHER_SEQUENCE = "suwon5a_real_weather"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _domain_label(value: str) -> str:
    return value.replace("_", " ")


def _domain_spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    current_domain = None
    start = 0
    end = 0
    for row in rows:
        domain = str(row.get("domain_majority", ""))
        frame_start = int(float(row.get("window_start_frame", 0)))
        frame_end = int(float(row.get("window_end_frame", frame_start)))
        if current_domain is None:
            current_domain = domain
            start = frame_start
        elif domain != current_domain:
            spans.append((start, end, current_domain))
            current_domain = domain
            start = frame_start
        end = frame_end
    if current_domain is not None:
        spans.append((start, end, current_domain))
    return spans


def _real_weather_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if str(row.get("sequence_name")) == REAL_WEATHER_SEQUENCE
    ]
    if len(selected) != len(rows):
        raise ValueError(f"Signal plots only support sequence {REAL_WEATHER_SEQUENCE!r}.")
    selected.sort(key=lambda row: _float(row.get("window_start_frame")))
    return selected


def _draw_domain_spans(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = [
        PALETTE["lilac"],
        PALETTE["sand"],
        PALETTE["green_soft"],
        PALETTE["red_soft"],
        "#E8E2F5",
        PALETTE["aqua"],
        PALETTE["grey_band"],
    ]
    text_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for index, (start, end, domain) in enumerate(_domain_spans(rows)):
        ax.axvspan(start, end, color=colors[index % len(colors)], alpha=0.42, lw=0)
        ax.text(
            (start + end) / 2,
            0.98,
            _domain_label(domain),
            transform=text_transform,
            ha="center",
            va="top",
            fontsize=5.6,
            color=PALETTE["ink"],
        )


def _draw_timeline(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    panel_label: str | None = None,
) -> Any:
    x = np.asarray([_float(row.get("window_start_frame")) for row in rows], dtype=float)
    f1_drop = np.asarray([_float(row.get("f1_drop")) for row in rows], dtype=float)
    full_score = np.asarray(
        display_values([_float(row.get("mean_full_drift_score_z")) for row in rows], config),
        dtype=float,
    )
    harmful = np.asarray([_bool(row.get("is_harmful_drift_window")) for row in rows])
    trigger_cfg = require_mapping(config, "trigger")
    threshold_cfg = require_mapping(trigger_cfg, "thresholds", context="trigger")
    threshold = require_float(threshold_cfg, "full_score_z", context="trigger.thresholds")

    _draw_domain_spans(ax, rows)
    ax.fill_between(x, 0, f1_drop, color=PALETTE["red_soft"], alpha=0.62, linewidth=0)
    ax.plot(x, f1_drop, color=PALETTE["red"], lw=1.35, label="F1 drop")
    if np.any(harmful):
        ax.scatter(
            x[harmful],
            f1_drop[harmful],
            s=18,
            color=PALETTE["red"],
            edgecolor="white",
            linewidth=0.45,
            zorder=6,
            label="Harmful window",
        )
    ax.set_ylim(-0.04, max(0.75, float(np.nanmax(f1_drop)) * 1.18 if f1_drop.size else 0.75))
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Teacher-pseudo-label F1 drop", color=PALETTE["red"])
    ax.tick_params(axis="y", labelcolor=PALETTE["red"])
    polish_axis(ax)

    ax_score = ax.twinx()
    ax_score.plot(x, full_score, color=PALETTE["blue"], lw=1.5, label="Full drift score z")
    ax_score.axhline(
        threshold,
        color=PALETTE["muted"],
        lw=0.9,
        linestyle=(0, (3, 2)),
    )
    y_min = min(0.0, float(np.nanmin(full_score)) if full_score.size else 0.0)
    y_max = max(threshold * 1.25, float(np.nanmax(full_score)) * 1.08 if full_score.size else 1.0)
    ax_score.set_ylim(y_min - max(1.0, 0.04 * (y_max - y_min)), y_max)
    ax_score.set_ylabel("Unlabeled drift score z", color=PALETTE["blue"])
    ax_score.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax_score.spines["top"].set_visible(False)
    if x.size:
        ax_score.text(
            x[-1],
            threshold,
            "threshold",
            ha="right",
            va="bottom",
            fontsize=5.8,
            color=PALETTE["muted"],
        )
    if panel_label:
        add_panel_label(ax, panel_label)
    return ax_score


def _draw_scatter(ax: Any, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    x = np.asarray(
        display_values([_float(row.get("mean_full_drift_score_z")) for row in rows], config),
        dtype=float,
    )
    y = np.asarray([_float(row.get("f1_drop")) for row in rows], dtype=float)
    harmful = np.asarray([_bool(row.get("is_harmful_drift_window")) for row in rows])
    ax.scatter(
        x[~harmful],
        y[~harmful],
        s=20,
        color=PALETTE["blue_soft"],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
        label="Non-harmful",
    )
    ax.scatter(
        x[harmful],
        y[harmful],
        s=24,
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.95,
        label="Harmful",
    )
    if x.size >= 2 and np.std(x) > 0.0:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 120)
        ax.plot(xs, coef[0] * xs + coef[1], color=PALETTE["ink"], lw=0.9, alpha=0.72)
    ax.axhline(
        require_float(
            require_mapping(config, "window"),
            "harmful_f1_drop_threshold",
            context="window",
        ),
        color=PALETTE["muted"],
        lw=0.8,
        linestyle=(0, (3, 2)),
    )
    ax.set_xlabel("Full drift score z")
    ax.set_ylabel("F1 drop")
    polish_axis(ax)
    ax.legend(loc="upper left", handletextpad=0.3)


def _draw_auc(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: _float(row.get("pr_auc"), math.nan))
    labels = [SIGNAL_LABELS.get(str(row.get("signal")), str(row.get("signal"))) for row in ordered]
    y = np.arange(len(ordered))
    height = 0.34
    roc = [_float(row.get("roc_auc"), math.nan) for row in ordered]
    pr = [_float(row.get("pr_auc"), math.nan) for row in ordered]
    ax.barh(y - height / 2, roc, height, color=PALETTE["blue_soft"], label="ROC-AUC")
    ax.barh(y + height / 2, pr, height, color=PALETTE["blue"], label="PR-AUC")
    for ypos, value in zip(y + height / 2, pr):
        if math.isfinite(value):
            ax.text(
                min(1.015, value + 0.012),
                ypos,
                f"{value:.2f}",
                ha="left",
                va="center",
                fontsize=6,
                color=PALETTE["ink"],
            )
    ax.axvline(0.5, color=PALETTE["muted"], lw=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.45, 1.04)
    ax.set_xlabel("Window-level discrimination")
    polish_axis(ax, grid_axis="x")
    ax.legend(loc="lower right", handlelength=1.5)


def plot_timeseries(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    _draw_timeline(ax, _real_weather_rows(rows), config)
    ax.set_title("Real weather stream: unlabeled drift score tracks harmful F1 drop", loc="left", pad=6)
    save_figure(fig, path, config)


def plot_scatter(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    _draw_scatter(ax, rows, config)
    ax.set_title("Score-to-degradation alignment", loc="left")
    save_figure(fig, path, config)


def plot_auc_bar(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.9), constrained_layout=True)
    _draw_auc(ax, rows)
    ax.set_title("Signal validity across windows", loc="left")
    save_figure(fig, path, config)


def plot_composite(
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    _draw_timeline(ax_a, _real_weather_rows(records), config, panel_label="a")
    _draw_auc(ax_b, summary)
    _draw_scatter(ax_c, records, config)
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    ax_a.set_title("Unlabeled signal rises with harmful detection degradation", loc="left", pad=6)
    ax_b.set_title("Validity", loc="left")
    ax_c.set_title("Dose-response", loc="left")
    save_figure(fig, path, config)


def plot_signal_validity(config: Mapping[str, Any]) -> Path:
    apply_publication_style()
    root = output_dir(config)
    records = _read_csv(root / "records" / "window_metrics.csv")
    summary = _read_csv(root / "analysis" / "signal_validity_summary.csv")
    plots_dir = root / "plots"
    plot_timeseries(config, records, plots_dir / "exp1_timeseries_real_weather.png")
    plot_scatter(config, records, plots_dir / "exp1_scatter_full_score_vs_f1_drop.png")
    plot_auc_bar(config, summary, plots_dir / "exp1_signal_auc_bar.png")
    plot_composite(config, records, summary, plots_dir / "figure_signal_validity_summary.png")
    return plots_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot drift signal validity results.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plot_signal_validity(load_config(args.config))


if __name__ == "__main__":
    main()
