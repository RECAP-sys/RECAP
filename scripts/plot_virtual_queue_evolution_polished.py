"""Candidate polished version of the Lyapunov trigger trace (Fig. 6).

The data-generating simulation is reused unchanged. The revision only improves
the visual hierarchy: compact journal typography, legends outside the data
traces, and consistent action/queue/cost color semantics.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.fontsize": 6.0,
        "legend.frameon": False,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "Chencang" / "tmc" / "figs"
DATA_PATH = FIGURES_DIR / "fig6_lyapunov_data.csv"


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.01,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main():
    source = pd.read_csv(DATA_PATH)
    epochs = source["epoch"].to_numpy(dtype=float)
    queue_epochs = np.arange(len(source) + 1, dtype=float)
    data = {
        "actions": source["action"].to_numpy(dtype=int),
        "score_skip": source["score_skip"].to_numpy(dtype=float),
        "score_raw": source["score_raw"].to_numpy(dtype=float),
        "score_raw_plus_feature": source["score_raw_plus_feature"].to_numpy(dtype=float),
        "Q_cloud": np.r_[0.0, source["Q_cloud"].to_numpy(dtype=float)],
        "Q_bw": np.r_[0.0, source["Q_bw"].to_numpy(dtype=float)],
        "mean_cloud_cost": source["mean_cloud_cost"].to_numpy(dtype=float),
        "mean_bw_cost": source["mean_bw_cost"].to_numpy(dtype=float),
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.50, 3.95),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0, 1.0], "hspace": 0.25},
    )
    fig.patch.set_facecolor("white")

    cloud_color = "#155A99"
    bandwidth_color = "#D97706"
    feature_color = "#87529A"
    skip_color = "#A8A8A8"

    # (a) Action score and the selected minimum.
    ax0 = axes[0]
    score_specs = [
        ("score_skip", r"$J_t(a_0)$", skip_color),
        ("score_raw", r"$J_t(a_1)$", cloud_color),
        ("score_raw_plus_feature", r"$J_t(a_2)$", feature_color),
    ]
    for action_index, (key, label, color) in enumerate(score_specs):
        ax0.plot(epochs, data[key], color=color, linewidth=1.0, label=label, zorder=2)
        selected = data["actions"] == action_index
        ax0.scatter(
            epochs[selected],
            data[key][selected],
            s=8,
            color=color,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
    ax0.set_ylabel("Action score")
    ax0.set_ylim(-0.05, 4.45)
    ax0.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.005),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.35,
        borderaxespad=0,
    )
    add_panel_label(ax0, "a")

    # (b) Virtual queues.
    ax1 = axes[1]
    ax1.plot(
        queue_epochs,
        data["Q_cloud"],
        color=cloud_color,
        linewidth=1.15,
        label=r"Cloud queue $Q_c$",
    )
    ax1.plot(
        queue_epochs,
        data["Q_bw"],
        color=bandwidth_color,
        linewidth=1.15,
        label=r"Bandwidth queue $Q_b$",
    )
    ax1.set_ylabel("Virtual queue")
    ax1.set_ylim(-0.02, 0.82)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.005),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.45,
        borderaxespad=0,
    )
    add_panel_label(ax1, "b")

    # (c) Running mean cost against the common budget.
    ax2 = axes[2]
    ax2.plot(
        epochs,
        data["mean_cloud_cost"],
        color=cloud_color,
        linewidth=1.15,
        label="Mean cloud cost",
    )
    ax2.plot(
        epochs,
        data["mean_bw_cost"],
        color=bandwidth_color,
        linewidth=1.15,
        label="Mean bandwidth cost",
    )
    ax2.axhline(
        0.5,
        color="#4D4D4D",
        linewidth=0.85,
        linestyle=(0, (2, 1.5)),
        label=r"Budgets $\lambda_c=\lambda_b$",
    )
    ax2.set_xlabel(r"Decision epoch $t$")
    ax2.set_ylabel("Running mean cost")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.005),
        ncol=3,
        columnspacing=0.7,
        handlelength=1.45,
        borderaxespad=0,
    )
    add_panel_label(ax2, "c")

    for ax in axes:
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.45)
        ax.set_xlim(-1, 120)
        ax.tick_params(width=0.65, length=2.5)
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[-1].set_xticks(np.arange(0, len(source) + 1, 20))
    fig.subplots_adjust(left=0.205, right=0.985, top=0.94, bottom=0.105)

    output_dir = Path(FIGURES_DIR)
    stem = output_dir / "fig6_lyapunov"
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
