from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PALETTE = {
    "ink": "#272727",
    "muted": "#767676",
    "grid": "#D8D8D8",
    "blue": "#0F4D92",
    "blue_soft": "#B4C0E4",
    "teal": "#42949E",
    "orange": "#E28E2C",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "green": "#2E9E44",
    "green_soft": "#DDF3DE",
    "violet": "#9A4D8E",
    "lilac": "#E0E0F0",
    "aqua": "#E0F0F0",
    "peach": "#F0E0D0",
    "sand": "#F3E7C9",
    "grey_band": "#F3F3F3",
}

METHOD_LABELS = {
    "confidence_only": "Confidence",
    "ema_entropy": "EMA entropy",
    "ema_feature_deviation": "Feature EMA",
    "recap_full": "RECAP full",
}

METHOD_COLORS = {
    "confidence_only": "#B4C0E4",
    "ema_entropy": "#7884B4",
    "ema_feature_deviation": "#42949E",
    "recap_full": "#0F4D92",
}

SIGNAL_LABELS = {
    "confidence_drop_z": "Confidence drop",
    "confidence_drop_signal": "Confidence drop",
    "output_entropy": "Output entropy",
    "ema_output_entropy_z": "EMA entropy",
    "ema_output_entropy": "EMA entropy",
    "boundary_feature_deviation": "Boundary feature",
    "ema_boundary_feature_deviation_z": "Boundary EMA",
    "ema_boundary_feature_deviation": "Boundary EMA",
    "full_drift_score_z": "Full drift score",
    "full_drift_score": "Full drift score",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def display_values(values: Sequence[float], config: Mapping[str, Any]) -> list[float]:
    clip_value = dict(config.get("plots") or {}).get("score_display_clip")
    if clip_value is None:
        return [float(value) for value in values]
    clip = float(clip_value)
    if not math.isfinite(clip) or clip <= 0.0:
        return [float(value) for value in values]
    return [max(-clip, min(clip, float(value))) for value in values]


def add_panel_label(ax: Any, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def polish_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.7)


def save_figure(fig: Any, path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = int(dict(config.get("plots") or {}).get("dpi", 300))
    stem = path.with_suffix("")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
