from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).with_name("seqKAN_results")
OUTPUT_PATH = RESULTS_DIR / "fig_learned_spline_paths_english.png"

PATHS = [
    {
        "feature": 8,
        "hidden": 15,
        "output": 2,
        "weight": 0.7006396055221558,
        "color": "#0072B2",
    },
    {
        "feature": 6,
        "hidden": 17,
        "output": 2,
        "weight": -0.42732521891593933,
        "color": "#D55E00",
    },
    {
        "feature": 4,
        "hidden": 9,
        "output": 4,
        "weight": 0.43483516573905945,
        "color": "#009E73",
    },
]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.0), sharex=True)

    for row, path in enumerate(PATHS):
        feature = path["feature"]
        hidden = path["hidden"]
        output = path["output"]
        weight = path["weight"]
        color = path["color"]
        csv_path = RESULTS_DIR / (
            f"kan_function_i{feature}_j{hidden}_m{output}_k3.csv"
        )
        data = pd.read_csv(csv_path)

        x = data["x"]
        spline = data["f_ij"]
        contribution = data["contrib_m"]
        delta = float(contribution.iloc[-1] - contribution.iloc[0])

        ax_spline, ax_contribution = axes[row]
        ax_spline.plot(x, spline, color=color, linewidth=2.2)
        ax_spline.axhline(0, color="#555555", linewidth=0.8, alpha=0.55)
        ax_spline.set_ylabel(r"Edge function $\phi_{ij}(x_i)$")
        ax_spline.set_title(
            f"Feature {feature} to hidden unit {hidden}", loc="left", weight="bold"
        )

        ax_contribution.plot(x, contribution, color=color, linewidth=2.2)
        ax_contribution.axhline(0, color="#555555", linewidth=0.8, alpha=0.55)
        ax_contribution.scatter(
            [x.iloc[0], x.iloc[-1]],
            [contribution.iloc[0], contribution.iloc[-1]],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=42,
            zorder=3,
        )
        ax_contribution.set_ylabel(r"Direct contribution $w_{mj}\phi_{ij}(x_i)$")
        ax_contribution.set_title(
            f"Hidden unit {hidden} to output {output}  "
            rf"($w_{{mj}}={weight:+.3f}$)",
            loc="left",
            weight="bold",
        )
        ax_contribution.text(
            0.97,
            0.08,
            rf"$\Delta c=c(3)-c(-3)={delta:+.3f}$",
            transform=ax_contribution.transAxes,
            ha="right",
            va="bottom",
            color=color,
            weight="bold",
            bbox={
                "boxstyle": "square,pad=0.3",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
                "alpha": 0.92,
            },
        )

        for ax in (ax_spline, ax_contribution):
            ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
            ax.set_xlim(-3, 3)
            ax.margins(y=0.12)

    axes[-1, 0].set_xlabel("Robust-scaled feature value")
    axes[-1, 1].set_xlabel("Robust-scaled feature value")
    fig.suptitle(
        "Learned seqKAN edge functions and direct output contributions",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.008,
        "Direct path components from the final 24-unit model; values are not causal effects.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=2.0, w_pad=1.6)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
