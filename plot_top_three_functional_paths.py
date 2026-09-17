from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).with_name("seqKAN_results")
SOURCE_FIGURE = RESULTS_DIR / "outputseqKANpath.png"
OUTPUT_PATH = RESULTS_DIR / "fig_top_three_functional_paths_english.png"

PATHS = [
    (8, 15, 2, 0.7006396055221558, (-0.60, 2.52), (40, 330)),
    (6, 17, 2, -0.42732521891593933, (2.35, 3.03), (390, 680)),
    (4, 9, 4, 0.43483516573905945, (-0.95, 1.90), (730, 1020)),
]


def digitize_partial_dependence(y_limits, row_bounds):
    """Recover the green PD curve from the original model-generated figure."""
    image = mpimg.imread(SOURCE_FIGURE)
    rgb = image[:, :, :3]
    green = (
        (rgb[:, :, 1] > 0.45)
        & (rgb[:, :, 1] > 1.35 * rgb[:, :, 0])
        & (rgb[:, :, 1] > 1.20 * rgb[:, :, 2])
    )
    lo, hi = row_bounds
    rows, cols = np.where(green[lo:hi])
    rows = rows + lo

    x_pixels = np.arange(cols.min(), cols.max() + 1)
    y_pixels = np.array(
        [np.median(rows[cols == x]) if np.any(cols == x) else np.nan for x in x_pixels]
    )
    y_pixels = pd.Series(y_pixels).interpolate(limit_direction="both").to_numpy()

    x = np.linspace(-3.0, 3.0, len(x_pixels))
    # Pixel bounds of the Matplotlib axes in the preserved 1189 x 1063 figure.
    axes_top = {40: 58, 390: 402, 730: 746}[lo]
    axes_bottom = {40: 323, 390: 667, 730: 1011}[lo]
    y_min, y_max = y_limits
    y = y_max - (y_pixels - axes_top) * (y_max - y_min) / (axes_bottom - axes_top)
    return x, y


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.2))

    for row, (feature, hidden, output, weight, y_limits, row_bounds) in enumerate(PATHS):
        csv_path = RESULTS_DIR / f"kan_function_i{feature}_j{hidden}_m{output}_k3.csv"
        data = pd.read_csv(csv_path)
        x_pd, y_pd = digitize_partial_dependence(y_limits, row_bounds)

        left, right = axes[row]
        left.plot(
            data["x"], data["f_ij"], color="#1976C9", linewidth=1.8,
            label=rf"$\phi_{{{feature}\to {hidden}}}(x)$",
        )
        left.plot(
            data["x"], data["contrib_m"], color="#FF7F0E", linewidth=1.8,
            linestyle="--", label=rf"$w_{{{output},{hidden}}}\phi(x)$",
        )
        left.set_title(
            f"Feature {feature} to hidden unit {hidden} (output {output})"
        )
        left.set_xlabel(f"Feature {feature} (robust-scaled)")
        left.set_ylabel("Function value")
        left.set_xlim(-3.2, 3.2)
        left.legend(loc="upper left", bbox_to_anchor=(0.63, 1.0), fontsize=8)

        right.plot(x_pd, y_pd, color="#2CA02C", linewidth=1.8)
        right.set_title(f"Partial dependence for output {output}")
        right.set_xlabel(f"Feature {feature} (robust-scaled)")
        right.set_ylabel(r"Mean predicted output at the last step")
        right.set_xlim(-3.2, 3.2)
        right.set_ylim(*y_limits)

        for ax in (left, right):
            ax.grid(True, alpha=0.28)

    fig.suptitle(
        "Top three functional paths: learned spline and contribution to the output",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.0, w_pad=2.0)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
