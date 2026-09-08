"""Redraw Fig. 2 as a compact, single-column manuscript figure.

The source data are the five repetitions recorded by the tail-training
motivation experiment.  The figure is deliberately exported at the size at
which it is placed in the TMC manuscript so that labels and box outlines stay
legible after PDF embedding.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "tail_training_motivation" / "summary.csv"
OUT = ROOT / "Chencang" / "tmc" / "figs" / "fig2_split_tail_cost"

BUCKETS = ["Early25%", "Middle50%", "Late75%"]
BUCKET_LABELS = ["Early 25%", "Middle 50%", "Late 75%"]
MODES = ["raw_freeze", "freeze", "split_rebuild", "split_cached"]
# Keys match the `mode` column of the source CSV; labels use the manuscript's
# name for the training mode (Partition train), while "split" stays reserved
# for the boundary position.
MODE_LABELS = {
    "raw_freeze": "Raw freeze",
    "freeze": "TorchLens freeze",
    "split_rebuild": "Partition rebuild",
    "split_cached": "Partition cached",
}
FACES = {
    "raw_freeze": "#F2C1BE",
    "freeze": "#C6D0EE",
    "split_rebuild": "#F0D4B1",
    "split_cached": "#B9DEBA",
}
EDGES = {
    "raw_freeze": "#A83D3A",
    "freeze": "#46527D",
    "split_rebuild": "#80612C",
    "split_cached": "#2D7544",
}


def main() -> None:
    df = pd.read_csv(DATA)
    df["training_time"] = df["suffix_train_time_sec"].astype(float)
    rebuild = df["feature_rebuild_time_sec"].fillna(0).astype(float)
    df.loc[df["mode"].eq("split_rebuild"), "training_time"] += rebuild[
        df["mode"].eq("split_rebuild")
    ]
    df["metric_percent"] = df["metric_after"].astype(float) * 100.0

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.8,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 600,
        }
    )

    # 3.45 in is the target single-column width used by the manuscript.
    fig, (ax_time, ax_map) = plt.subplots(
        2,
        1,
        figsize=(3.45, 2.55),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.02], "hspace": 0.08},
    )

    positions = {bucket: i + 1 for i, bucket in enumerate(BUCKETS)}
    offsets = [-0.285, -0.095, 0.095, 0.285]
    width = 0.145

    def draw_boxes(ax, column: str, scale: float = 1.0) -> None:
        for bucket in BUCKETS:
            for mode, offset in zip(MODES, offsets):
                values = df.loc[
                    df["split_bucket"].eq(bucket) & df["mode"].eq(mode), column
                ].dropna().astype(float).tolist()
                if not values:
                    continue
                values = [value * scale for value in values]
                bp = ax.boxplot(
                    values,
                    positions=[positions[bucket] + offset],
                    widths=width,
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False,
                    showmeans=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(FACES[mode])
                    patch.set_edgecolor(EDGES[mode])
                    patch.set_linewidth(0.8)
                    patch.set_alpha(0.95)
                for key in ("whiskers", "caps"):
                    for line in bp[key]:
                        line.set_color(EDGES[mode])
                        line.set_linewidth(0.7)
                for line in bp["medians"]:
                    line.set_color("#202020")
                    line.set_linewidth(1.0)
                for flier in bp["fliers"]:
                    flier.set_marker("o")
                    flier.set_markerfacecolor(FACES[mode])
                    flier.set_markeredgecolor(EDGES[mode])
                    flier.set_markersize(2.0)
                    flier.set_alpha(0.65)

    draw_boxes(ax_time, "training_time")
    draw_boxes(ax_map, "metric_percent")

    for ax in (ax_time, ax_map):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.4, alpha=0.7)
        ax.set_xlim(0.48, 3.52)
        ax.set_xticks([1, 2, 3])
        ax.tick_params(axis="both", colors="#4D4D4D", length=2.2, width=0.6)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#4D4D4D")
            ax.spines[spine].set_linewidth(0.65)

    ax_time.set_ylabel("Training time (s)", labelpad=2)
    ax_time.set_ylim(0, 45)
    ax_time.set_yticks([0, 15, 30, 45])
    ax_time.set_xticklabels([])

    ax_map.set_ylabel("Proxy mAP (%)", labelpad=2)
    ax_map.set_xlabel("Split position", labelpad=2)
    ax_map.set_xticklabels(BUCKET_LABELS)
    ax_map.set_ylim(15, 55)
    ax_map.set_yticks([20, 30, 40, 50])

    handles = [
        Patch(
            facecolor=FACES[mode],
            edgecolor=EDGES[mode],
            linewidth=0.75,
            label=MODE_LABELS[mode],
        )
        for mode in MODES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        columnspacing=0.55,
        handlelength=0.95,
        handleheight=0.7,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.17, top=0.91)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(OUT.with_suffix(suffix), **kwargs)
    plt.close(fig)
    print(f"Saved {OUT}.{{svg,pdf,tiff,png}}")


if __name__ == "__main__":
    main()
