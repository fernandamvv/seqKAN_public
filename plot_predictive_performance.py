"""Generate IEEE paper Figure 2 without overlapping point labels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "seqkan_ieee_assets" / "figs" / "fig2_predictive_performance.png"

HIDDEN_DIMENSIONS = np.array([8, 24, 352])
SEQKAN_MSE = np.array([0.173, 0.159, 0.169])
SEQKAN_SE = np.array([0.004, 0.003, 0.004])
GRU_MSE = np.array([0.178, 0.163, 0.146])
GRU_SE = np.array([0.004, 0.003, 0.003])


def add_labels(ax, x_values, y_values, offsets, color):
    for x, y, offset in zip(x_values, y_values, offsets):
        ax.annotate(
            f"{y:.3f}",
            xy=(x, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=9,
            fontweight="semibold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    seq_color = "#1976C9"
    gru_color = "#D55E00"
    ax.errorbar(
        HIDDEN_DIMENSIONS,
        SEQKAN_MSE,
        yerr=SEQKAN_SE,
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=3,
        color=seq_color,
        label="seqKAN",
        zorder=3,
    )
    ax.errorbar(
        HIDDEN_DIMENSIONS,
        GRU_MSE,
        yerr=GRU_SE,
        marker="s",
        markersize=6,
        linewidth=2,
        capsize=3,
        color=gru_color,
        label="GRU",
        zorder=3,
    )

    # Alternating vertical positions keep the close H=8 and H=24 values legible.
    add_labels(ax, HIDDEN_DIMENSIONS, SEQKAN_MSE, (-13, -13, 11), seq_color)
    add_labels(ax, HIDDEN_DIMENSIONS, GRU_MSE, (11, 11, -13), gru_color)

    ax.set_xscale("log")
    ax.set_xticks(HIDDEN_DIMENSIONS, [str(value) for value in HIDDEN_DIMENSIONS])
    ax.set_xlabel("Hidden dimension")
    ax.set_ylabel("Test MSE")
    ax.set_ylim(0.137, 0.188)
    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.grid(False, which="minor")
    ax.legend(loc="lower center", ncols=2, frameon=True)
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
