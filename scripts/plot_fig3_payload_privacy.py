"""Create a compact line-based version of the payload/privacy figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "split_tradeoff" / "rfdetr_nano" / "candidate_records.json"
OUT = ROOT / "Chencang" / "tmc" / "figs" / "fig3_payload_privacy"
RAW_INPUT_MB = 5.9326171875


def main() -> None:
    records = json.loads(DATA.read_text(encoding="utf-8"))
    ordered = sorted(
        enumerate(records),
        key=lambda item: (
            item[1].get("legacy_layer_index")
            if item[1].get("legacy_layer_index") is not None
            else item[0],
            item[1].get("candidate_id", ""),
        ),
    )
    ordered_records = [record for _index, record in ordered]
    payload = [float(record["payload_mb"]) for record in ordered_records]
    privacy = [float(record["privacy_leakage_score"]) for record in ordered_records]
    x = list(range(len(ordered_records)))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.8,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 600,
        }
    )

    # The claim is a contrast: payload is non-monotonic, whereas leakage
    # falls with depth. Lines preserve every candidate without the visual
    # density of 569 individual bars.
    fig, (ax_payload, ax_privacy) = plt.subplots(
        2,
        1,
        figsize=(3.45, 2.55),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.10},
    )

    payload_color = "#5F6B9D"
    payload_edge = "#3D486F"
    privacy_color = "#B64D4B"

    ax_payload.step(
        x,
        payload,
        where="mid",
        color=payload_color,
        linewidth=0.72,
        alpha=0.95,
        zorder=3,
    )
    ax_payload.axhline(
        RAW_INPUT_MB,
        color=privacy_color,
        linewidth=0.8,
        linestyle=(0, (1.2, 2.0)),
        alpha=0.9,
        zorder=2,
    )
    ax_privacy.plot(
        x,
        privacy,
        color=privacy_color,
        linewidth=0.85,
        alpha=0.95,
        zorder=3,
    )

    for ax in (ax_payload, ax_privacy):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.4, alpha=0.7)
        ax.set_xlim(0, len(x) - 1)
        ax.tick_params(axis="both", colors="#4D4D4D", length=2.2, width=0.6)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#4D4D4D")
            ax.spines[spine].set_linewidth(0.65)

    ax_payload.set_ylabel("Payload (MB)", labelpad=2)
    ax_payload.set_ylim(0, 8.5)
    ax_payload.set_yticks([0, 2, 4, 6, 8])
    fig.legend(
        handles=[
            Line2D([0], [0], color=payload_edge, linewidth=1.2, label="Feature payload"),
            Line2D(
                [0],
                [0],
                color=privacy_color,
                linewidth=1.0,
                linestyle=(0, (1.2, 2.0)),
                label="Raw input (5.93 MB)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.98),
        ncol=2,
        handlelength=1.15,
        columnspacing=0.8,
        borderaxespad=0,
    )

    ax_privacy.set_ylabel("Leakage score", labelpad=2)
    ax_privacy.set_xlabel("Candidate index (depth order)", labelpad=2)
    ax_privacy.set_ylim(0, 1.05)
    ax_privacy.set_yticks([0.0, 0.5, 1.0])
    ax_privacy.set_xticks([0, 160, 320, 480, len(x) - 1])
    ax_privacy.set_xticklabels(["0", "160", "320", "480", str(len(x) - 1)])

    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.18, top=0.91)
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
