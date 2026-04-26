"""
figures/fig_sensitivity.py
===========================
Figure 2: Sensitivity analysis grid.

Rows    = sample size n ∈ {500, 1000, 5000, 50000}
Columns = algorithm
X-axis  = bias coefficient β
Color   = detection rate (0→1)
Cell annotation = mean SHD ± std

Reads: results/sensitivity_results.pkl
Output: figures/output/fig_sensitivity.pdf
"""

import os, sys, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from figures.fig_utils import set_paper_style, save_pdf, FIGURES_DIR

set_paper_style()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")

ALGORITHMS = ["PC", "FCI", "GES", "GRaSP", "ICA-LiNGAM", "DirectLiNGAM"]
NS         = [500, 1_000, 5_000, 50_000]
BETAS      = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]


def load_results() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "sensitivity_results.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Run synthetic/sensitivity_analysis.py first.\n"
            f"Expected: {path}"
        )
    return pd.read_pickle(path)


def build_grid(df: pd.DataFrame, alg: str, n: int, metric: str) -> np.ndarray:
    """Return 1D array over BETAS for given algorithm/n/metric."""
    sub = df[(df["algorithm"] == alg) & (df["n"] == n)]
    vals = []
    for beta in BETAS:
        cell = sub[sub["beta"] == beta]
        if len(cell) == 0 or metric not in cell.columns:
            vals.append(float("nan"))
        elif metric == "detection_rate":
            # For biased conditions (beta > 0): fraction detected
            # For beta == 0: fraction with NO false positive
            if beta == 0.0:
                vals.append(cell["correct_unbiased"].mean()
                            if "correct_unbiased" in cell else float("nan"))
            else:
                vals.append(cell["detected_biased"].mean()
                            if "detected_biased" in cell else float("nan"))
        elif metric == "shd_mean":
            vals.append(cell["shd_biased"].mean()
                        if "shd_biased" in cell else float("nan"))
        elif metric == "shd_std":
            vals.append(cell["shd_biased"].std()
                        if "shd_biased" in cell else float("nan"))
    return np.array(vals)


def make_figure(df: pd.DataFrame):
    n_rows = len(NS)
    n_cols = len(ALGORITHMS)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(10, 6),
                              sharex=True, sharey=False)
    fig.suptitle("Sensitivity Analysis: Bias Detection Rate and SHD\n"
                 "by Algorithm, Sample Size (n), and Bias Coefficient (β)",
                 fontsize=10, fontweight="bold", y=1.01)

    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for row_i, n in enumerate(NS):
        for col_i, alg in enumerate(ALGORITHMS):
            ax = axes[row_i, col_i]

            det  = build_grid(df, alg, n, "detection_rate")
            shd_m = build_grid(df, alg, n, "shd_mean")
            shd_s = build_grid(df, alg, n, "shd_std")

            # Color bar chart by detection rate
            colors = [cmap(norm(v)) if not np.isnan(v) else "#cccccc"
                      for v in det]
            bars = ax.bar(range(len(BETAS)), det, color=colors,
                          width=0.7, edgecolor="white", linewidth=0.4)

            # Annotate each bar with SHD mean±std
            for bi, (d, sm, ss) in enumerate(zip(det, shd_m, shd_s)):
                if not np.isnan(sm):
                    ax.text(bi, min(d + 0.05, 0.95) if not np.isnan(d) else 0.5,
                            f"{sm:.0f}±{ss:.0f}",
                            ha="center", va="bottom", fontsize=5.5,
                            color="#222222")

            ax.set_ylim(0, 1.12)
            ax.set_xticks(range(len(BETAS)))
            ax.axhline(1.0, color="#aaaaaa", lw=0.5, ls="--")
            ax.axhline(0.5, color="#cccccc", lw=0.4, ls=":")

            # Column header (algorithm name) on top row
            if row_i == 0:
                ax.set_title(alg, fontsize=8, fontweight="bold", pad=4)

            # Row label (n) on leftmost column
            if col_i == 0:
                ax.set_ylabel(f"n={n:,}", fontsize=8, labelpad=4)
            else:
                ax.set_ylabel("")

            # X-axis labels on bottom row only
            if row_i == n_rows - 1:
                ax.set_xticklabels([str(b) for b in BETAS],
                                    fontsize=6.5, rotation=45)
                if col_i == n_cols // 2:
                    ax.set_xlabel("Bias coefficient β", fontsize=8, labelpad=6)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=6.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                         fraction=0.015, pad=0.02)
    cbar.set_label("Detection rate\n(green=1.0 correct)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Caption note
    fig.text(0.5, -0.02,
             "Bar height = detection rate (biased: fraction with Race→Loan detected;"
             " β=0: fraction with NO false positive).\n"
             "Annotation = mean SHD ± std over 20 seeds. GRaSP shows near-zero"
             " detection regardless of β or n (structural failure).",
             ha="center", fontsize=7, style="italic")

    plt.tight_layout()
    save_pdf(fig, "fig_sensitivity.pdf")
    plt.close(fig)
    print("Figure 2 (sensitivity) complete.")


def main():
    df = load_results()
    print(f"Loaded sensitivity results: {len(df)} records, "
          f"algorithms: {df['algorithm'].unique()}")
    make_figure(df)


if __name__ == "__main__":
    main()
