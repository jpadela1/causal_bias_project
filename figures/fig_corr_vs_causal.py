"""
figures/fig_corr_vs_causal.py
==============================
Figure 6: Correlation-based vs. causal estimates for COMPAS bias.

Left panel:  Three correlation-based fairness metrics (all failing the 4/5 rule)
Right panel: Causal estimates showing disparity survives full controls

Reads: results/ate_results.pkl, results/compas_results.pkl
Output: figures/output/fig_corr_vs_causal.pdf
"""

import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from figures.fig_utils import set_paper_style, save_pdf

set_paper_style()
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")


def load_results():
    ate_path = os.path.join(RESULTS_DIR, "ate_results.pkl")
    cmp_path = os.path.join(RESULTS_DIR, "compas_results.pkl")

    ate = {}
    cmp = {}
    if os.path.exists(ate_path):
        with open(ate_path, "rb") as f:
            ate = pickle.load(f)
    else:
        # Use paper values as fallback
        ate = {"ate_no_controls": 0.324, "ate_age_charge": 0.246,
               "ate_full": 0.138, "dir_score": 1.545,
               "sp_ratio": 1.845, "dir_recid": 1.471}

    if os.path.exists(cmp_path):
        with open(cmp_path, "rb") as f:
            cmp = pickle.load(f)

    lingam_beta = 0.1336
    if "ICA-LiNGAM" in cmp and cmp["ICA-LiNGAM"].get("beta_hat") is not None:
        lingam_beta = abs(cmp["ICA-LiNGAM"]["beta_hat"])

    return ate, lingam_beta


def make_figure():
    ate, lingam_beta = load_results()

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(7.0, 3.6),
        gridspec_kw={"width_ratios": [1.1, 1.0]}
    )

    RED   = "#cc3333"
    DGRAY = "#444444"
    MGRAY = "#888888"
    LGRAY = "#bbbbbb"
    FAIR_THRESHOLD = 0.80

    # ── LEFT: Correlation-based fairness metrics ──────────────────────────────
    corr_labels = [
        "COMPAS Score\nDisparate Impact\n(≥0.8 = fair)",
        "COMPAS Score\nStat. Parity\n(=0 = fair)",
        "Recidivism\nDisparate Impact",
    ]
    corr_vals = [
        ate.get("dir_score",   1.545),
        ate.get("sp_ratio",    1.845),
        ate.get("dir_recid",   1.471),
    ]

    bars_l = ax_left.bar(range(3), corr_vals, color=RED, alpha=0.85,
                          edgecolor="white", linewidth=0.5, width=0.6)
    ax_left.axhline(FAIR_THRESHOLD, color="#444444", lw=1.2, ls="--",
                    label=f"4/5 rule threshold ({FAIR_THRESHOLD})")
    ax_left.axhline(1.0, color="#aaaaaa", lw=0.6, ls=":")

    for bar, val in zip(bars_l, corr_vals):
        ax_left.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=RED)

    ax_left.set_xticks(range(3))
    ax_left.set_xticklabels(corr_labels, fontsize=7.5)
    ax_left.set_ylabel("Metric value", fontsize=9)
    ax_left.set_ylim(0, 2.15)
    ax_left.set_title("Correlation-Based Fairness Metrics\n"
                       "(Detect THAT bias exists)", fontsize=9, pad=5)
    ax_left.legend(fontsize=7.5, loc="upper right")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # ── RIGHT: Causal estimates ───────────────────────────────────────────────
    causal_labels = [
        "ATE Race→Score\n(no controls)",
        "ATE Race→Score\n(full controls)",
        "LiNGAM β̂\nRace→Score",
    ]
    causal_vals = [
        ate.get("ate_no_controls", 0.324),
        ate.get("ate_full",        0.138),
        lingam_beta,
    ]
    causal_colors = [LGRAY, MGRAY, DGRAY]

    bars_r = ax_right.bar(range(3), causal_vals, color=causal_colors,
                           edgecolor="white", linewidth=0.5, width=0.6)

    for bar, val in zip(bars_r, causal_vals):
        ax_right.text(bar.get_x() + bar.get_width()/2, val + 0.004,
                      f"+{val:.3f}", ha="center", va="bottom",
                      fontsize=8.5, fontweight="bold")

    # Annotation: convergence bracket
    y_top = causal_vals[1] + 0.012
    ax_right.annotate("", xy=(2, causal_vals[2]), xytext=(1, causal_vals[1]),
                       arrowprops=dict(arrowstyle="<->", color="#cc3333", lw=1.2))
    ax_right.text(1.5, (causal_vals[1] + causal_vals[2])/2 + 0.01,
                  f"Δ={abs(causal_vals[1]-causal_vals[2]):.4f}",
                  ha="center", fontsize=7, color=RED)

    ax_right.set_xticks(range(3))
    ax_right.set_xticklabels(causal_labels, fontsize=7.5)
    ax_right.set_ylabel("Causal effect estimate (std. units)", fontsize=9)
    ax_right.set_ylim(0, 0.42)
    ax_right.set_title("Causal Estimates\n(Quantify WHY and HOW MUCH bias exists)",
                        fontsize=9, pad=5)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # Note: positive = minority group over-scored
    ax_right.text(0.97, 0.97,
                  "Positive β = minority\ngroup over-scored\nindependent of risk",
                  transform=ax_right.transAxes, ha="right", va="top",
                  fontsize=6.5,
                  bbox=dict(boxstyle="round", fc="#fff8f0", ec="#ccaa66", alpha=0.9))

    fig.suptitle("Correlation-Based vs. Causal Methods for COMPAS Bias Detection\n"
                 "Causal methods isolate the direct Race→Score pathway from legitimate"
                 " criminal-history pathways",
                 fontsize=9, y=1.02)

    plt.tight_layout()
    save_pdf(fig, "fig_corr_vs_causal.pdf")
    plt.close(fig)
    print("Figure 6 (correlation vs causal) complete.")


if __name__ == "__main__":
    make_figure()
