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
    require_int,
    require_mapping,
)
from experiments.drift_detection_validity.figure_style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    PALETTE,
    add_panel_label,
    apply_publication_style,
    display_values,
    polish_axis,
    save_figure,
)
from experiments.drift_detection_validity.online_trigger_analysis import (  # noqa: E402
    extract_harmful_drift_events,
)

REAL_WEATHER_SEQUENCE = "suwon5a_real_weather"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _real_weather_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if str(row.get("sequence_name")) == REAL_WEATHER_SEQUENCE
    ]
    if len(selected) != len(rows):
        raise ValueError(f"Online trigger plots only support sequence {REAL_WEATHER_SEQUENCE!r}.")
    selected.sort(key=lambda row: _float(row.get("window_start_frame")))
    return selected


def _method_label(method: Any) -> str:
    value = str(method)
    return METHOD_LABELS.get(value, value.replace("_", " "))


def _event_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    window_cfg = require_mapping(config, "window")
    return extract_harmful_drift_events(
        rows,
        harmful_f1_drop_threshold=require_float(
            window_cfg,
            "harmful_f1_drop_threshold",
            context="window",
        ),
        harmful_consecutive_windows=require_int(
            window_cfg,
            "harmful_consecutive_windows",
            context="window",
        ),
        harmful_merge_gap_windows=require_int(
            window_cfg,
            "harmful_merge_gap_windows",
            context="window",
        ),
    )


def _draw_domain_transitions(ax: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    last_domain = None
    for row in rows:
        domain = str(row.get("domain_majority"))
        frame = _float(row.get("window_start_frame"))
        if last_domain is not None and domain != last_domain:
            ax.axvline(frame, color=PALETTE["grid"], linewidth=0.7, alpha=0.9, zorder=0)
        last_domain = domain


def _draw_timeline(
    ax: Any,
    config: Mapping[str, Any],
    window_rows: Sequence[Mapping[str, Any]],
    trigger_rows: Sequence[Mapping[str, Any]],
    *,
    panel_label: str | None = None,
) -> Any:
    rows = _real_weather_rows(window_rows)
    x = np.asarray([_float(row.get("window_start_frame")) for row in rows], dtype=float)
    f1_drop = np.asarray([_float(row.get("f1_drop")) for row in rows], dtype=float)
    score = np.asarray(
        display_values([_float(row.get("mean_full_drift_score_z")) for row in rows], config),
        dtype=float,
    )
    trigger_cfg = require_mapping(config, "trigger")
    threshold_cfg = require_mapping(trigger_cfg, "thresholds", context="trigger")
    threshold = require_float(threshold_cfg, "full_score_z", context="trigger.thresholds")

    _draw_domain_transitions(ax, rows)
    ax.fill_between(x, 0, f1_drop, color=PALETTE["red_soft"], alpha=0.55, linewidth=0)
    ax.plot(x, f1_drop, color=PALETTE["red"], lw=1.25, label="F1 drop")
    ax.set_ylim(-0.04, max(0.75, float(np.nanmax(f1_drop)) * 1.18 if f1_drop.size else 0.75))
    ax.set_xlabel("Frame index")
    ax.set_ylabel("F1 drop", color=PALETTE["red"])
    ax.tick_params(axis="y", labelcolor=PALETTE["red"])
    polish_axis(ax)

    text_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for event in _event_rows(rows, config):
        start = _float(event.get("frame"))
        end = _float(event.get("end_frame"), start)
        ax.axvspan(start, end, color=PALETTE["red_soft"], alpha=0.28, lw=0)
        ax.axvline(start, color=PALETTE["red"], lw=1.0, linestyle=(0, (3, 2)), zorder=5)
        ax.text(
            start,
            0.93,
            "harmful episode",
            transform=text_transform,
            ha="left",
            va="top",
            fontsize=5.8,
            color=PALETTE["red"],
        )

    ax_score = ax.twinx()
    ax_score.plot(x, score, color=PALETTE["blue"], lw=1.45, label="Full score z")
    ax_score.axhline(
        threshold,
        color=PALETTE["muted"],
        lw=0.9,
        linestyle=(0, (3, 2)),
    )
    triggers = [
        _float(row.get("frame"))
        for row in trigger_rows
        if str(row.get("sequence_name")) == REAL_WEATHER_SEQUENCE
        and str(row.get("method")) == "recap_full"
        and str(row.get("kind")) != "missed"
    ]
    for trigger in triggers:
        ax_score.axvline(trigger, color=PALETTE["green"], lw=1.0, zorder=6)
        ax_score.text(
            trigger,
            0.84,
            "trigger",
            transform=text_transform,
            ha="left",
            va="top",
            fontsize=5.8,
            color=PALETTE["green"],
        )
    y_min = min(0.0, float(np.nanmin(score)) if score.size else 0.0)
    y_max = max(threshold * 1.25, float(np.nanmax(score)) * 1.08 if score.size else 1.0)
    ax_score.set_ylim(y_min - max(1.0, 0.04 * (y_max - y_min)), y_max)
    ax_score.set_ylabel("Full drift score z", color=PALETTE["blue"])
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
    ax.set_title(f"Online trigger replay: {REAL_WEATHER_SEQUENCE}", loc="left", pad=6)
    if panel_label:
        add_panel_label(ax, panel_label)
    return ax_score


def _draw_trigger_metric_bars(ax: Any, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    rows = sorted(summary_rows, key=lambda row: str(row.get("method")) == "recap_full")
    methods = [_method_label(row.get("method")) for row in rows]
    y = np.arange(len(rows))
    height = 0.22
    precision = [_float(row.get("precision")) for row in rows]
    recall = [_float(row.get("recall")) for row in rows]
    trigger_f1 = [_float(row.get("trigger_f1")) for row in rows]
    ax.barh(y - height, precision, height, color=PALETTE["blue_soft"], label="Precision")
    ax.barh(y, recall, height, color=PALETTE["teal"], label="Recall")
    ax.barh(y + height, trigger_f1, height, color=PALETTE["blue"], label="Trigger F1")
    for ypos, value in zip(y + height, trigger_f1):
        ax.text(min(1.02, value + 0.018), ypos, f"{value:.2f}", ha="left", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Episode-level score")
    polish_axis(ax, grid_axis="x")
    ax.legend(loc="lower right", ncol=1, handlelength=1.4)


def _draw_event_count_bars(ax: Any, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(summary_rows)
    methods = [_method_label(row.get("method")) for row in rows]
    x = np.arange(len(rows))
    detected = np.asarray([_float(row.get("detected")) for row in rows], dtype=float)
    missed = np.asarray([_float(row.get("missed")) for row in rows], dtype=float)
    false_triggers = np.asarray([_float(row.get("false_triggers")) for row in rows], dtype=float)
    width = 0.62
    ax.bar(x, detected, width, color=PALETTE["blue"], label="Detected")
    ax.bar(x, missed, width, bottom=detected, color=PALETTE["red_soft"], label="Missed")
    for xi, det, miss, false in zip(x, detected, missed, false_triggers):
        if false > 0:
            ax.scatter(
                [xi],
                [det + miss + false + 0.12],
                marker="x",
                s=32,
                color=PALETTE["red"],
                linewidth=1.1,
                zorder=6,
            )
            text_color = PALETTE["ink"]
        else:
            text_color = PALETTE["muted"]
        ax.text(
            xi,
            det + miss + 0.18,
            f"FP {int(false)}",
            ha="center",
            va="bottom",
            fontsize=6,
            color=text_color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Episode count")
    polish_axis(ax)


def _draw_delay_bars(ax: Any, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(summary_rows)
    methods = [_method_label(row.get("method")) for row in rows]
    delays = np.asarray([_float(row.get("avg_detection_delay_frames"), math.nan) for row in rows])
    x = np.arange(len(rows))
    colors = [PALETTE["green"] if value <= 0 else PALETTE["orange"] for value in delays]
    ax.bar(x, delays, color=colors, width=0.62)
    ax.axhline(0, color=PALETTE["ink"], lw=0.8)
    for xi, value in zip(x, delays):
        if math.isfinite(float(value)):
            va = "top" if value < 0 else "bottom"
            offset = -1.2 if value < 0 else 1.2
            ax.text(xi, value + offset, f"{value:.0f}", ha="center", va=va, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Mean delay (frames)")
    polish_axis(ax)


def plot_timeline(
    config: Mapping[str, Any],
    window_rows: Sequence[Mapping[str, Any]],
    trigger_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    _draw_timeline(ax, config, window_rows, trigger_rows)
    fig.subplots_adjust(left=0.10, right=0.88, top=0.86, bottom=0.20)
    save_figure(fig, path, config)


def plot_detected_bar(
    config: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    _draw_event_count_bars(ax, summary_rows)
    ax.set_title("Detected episodes and false triggers", loc="left")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.30)
    save_figure(fig, path, config)


def plot_delay_bar(
    config: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    _draw_delay_bars(ax, summary_rows)
    ax.set_title("Detection delay", loc="left")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.30)
    save_figure(fig, path, config)


def plot_composite(
    config: Mapping[str, Any],
    window_rows: Sequence[Mapping[str, Any]],
    trigger_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(7.6, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    _draw_timeline(ax_a, config, window_rows, trigger_rows, panel_label="a")
    _draw_trigger_metric_bars(ax_b, summary_rows)
    _draw_event_count_bars(ax_c, summary_rows)
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    ax_b.set_title("Episode-level trigger quality", loc="left")
    ax_c.set_title("Error profile", loc="left")
    fig.subplots_adjust(left=0.08, right=0.94, top=0.92, bottom=0.13, hspace=0.48, wspace=0.26)
    save_figure(fig, path, config)


def plot_online_trigger(config: Mapping[str, Any]) -> Path:
    apply_publication_style()
    root = output_dir(config)
    window_rows = _read_csv(root / "records" / "window_metrics.csv")
    trigger_rows = _read_csv(root / "analysis" / "online_trigger_events.csv")
    summary_rows = _read_csv(root / "analysis" / "online_trigger_method_summary.csv")
    plots_dir = root / "plots"
    plot_timeline(
        config,
        window_rows,
        trigger_rows,
        plots_dir / "exp2_trigger_timeline_real_weather.png",
    )
    plot_detected_bar(config, summary_rows, plots_dir / "exp2_detected_missed_false_bar.png")
    plot_delay_bar(config, summary_rows, plots_dir / "exp2_delay_bar.png")
    plot_composite(
        config,
        window_rows,
        trigger_rows,
        summary_rows,
        plots_dir / "figure_online_trigger_summary.png",
    )
    return plots_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot online drift trigger analysis results.")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    plot_online_trigger(load_config(args.config))


if __name__ == "__main__":
    main()
