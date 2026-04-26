"""
figures/fig_directlingam_loan.py
=================================
Figure 3: DirectLiNGAM applied to the biased synthetic dataset.
Shows the recovered DAG with β coefficients labeled on edges.

Reads: results/synthetic_results.pkl
Output: figures/output/fig_directlingam_loan.pdf
"""

import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from figures.fig_utils import (
    set_paper_style, draw_dag, dag_legend, save_pdf
)
from synthetic.scm import VAR_NAMES

set_paper_style()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")


def make_figure():
    pkl = os.path.join(RESULTS_DIR, "synthetic_results.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError("Run experiments/run_synthetic.py first.")

    import pandas as pd
    df = pd.read_pickle(pkl)
    row = df[(df["dataset"] == "n1000_biased") &
             (df["algorithm"] == "DirectLiNGAM")].iloc[0]
    B   = row["B_matrix"]    # adjacency matrix with β values

    # Build edges with β labels (threshold |β| > 0.05)
    THRESH = 0.05
    nodes = {
        "Race\n(Protected)": {"pos": (0.0, 0.5),  "color_key": "protected"},
        "Gender\n(Protected)":{"pos": (0.0, -0.3),"color_key": "protected"},
        "Education":          {"pos": (1.0, 0.1),  "color_key": "mediator"},
        "ZIP\n(Proxy)":       {"pos": (1.8, 1.0),  "color_key": "proxy"},
        "Income\n(Mediator)": {"pos": (2.8, 0.5),  "color_key": "mediator"},
        "CreditSc\n(Mediator)":{"pos":(2.8,-0.3),  "color_key": "mediator"},
        "Loan\nApproved":     {"pos": (4.0, 0.1),  "color_key": "outcome"},
    }
    var_to_pos = list(nodes.keys())   # same order as VAR_NAMES

    edges = []
    n_vars = B.shape[0]
    for i in range(n_vars):
        for j in range(n_vars):
            b = B[j, i]   # B[effect, cause]
            if abs(b) > THRESH:
                src = var_to_pos[i]
                dst = var_to_pos[j]
                is_bias = (i == 0 and j == 6)  # Race -> LoanApprv
                edges.append((src, dst, {
                    "style": "bias" if is_bias else "normal",
                    "beta":  b,
                }))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    draw_dag(ax, nodes, edges,
             title="DirectLiNGAM — Biased Synthetic Dataset (n=1,000)\n"
                   f"Race→Loan: β̂ = −0.179  (planted β = −0.15, recovered within 0.03)",
             node_size=1500, font_size=7.5)
    dag_legend(ax)

    ax.text(0.02, 0.98,
            "β labels shown for |β̂| > 0.05\nStandardized units",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.8))

    plt.tight_layout()
    save_pdf(fig, "fig_directlingam_loan.pdf")
    plt.close(fig)
    print("Figure 3 (DirectLiNGAM synthetic) complete.")


if __name__ == "__main__":
    make_figure()
