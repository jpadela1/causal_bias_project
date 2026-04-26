"""
figures/fig_gt_biased.py
========================
Figure 1: Ground-truth causal DAG for the biased synthetic dataset.

Output: figures/output/fig_gt_biased.pdf
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from figures.fig_utils import (
    set_paper_style, draw_dag, dag_legend, save_pdf, FIGURES_DIR
)

set_paper_style()


def make_figure():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Node layout: left-to-right causal flow
    nodes = {
        "Race\n(Protected)": {"pos": (0.0, 0.5),  "color_key": "protected"},
        "Gender\n(Protected)":{"pos": (0.0, -0.3), "color_key": "protected"},
        "Education":          {"pos": (1.0,  0.1), "color_key": "mediator"},
        "ZIP Code\n(Proxy)":  {"pos": (1.8,  1.0), "color_key": "proxy"},
        "Income\n(Mediator)": {"pos": (2.8,  0.5), "color_key": "mediator"},
        "CreditSc\n(Mediator)":{"pos": (2.8, -0.3),"color_key": "mediator"},
        "Loan\nApproved":     {"pos": (4.0,  0.1), "color_key": "outcome"},
    }

    edges = [
        ("Race\n(Protected)",  "ZIP Code\n(Proxy)",    {"style": "normal"}),
        ("Race\n(Protected)",  "Income\n(Mediator)",   {"style": "normal"}),
        ("Race\n(Protected)",  "Loan\nApproved",
         {"style": "bias", "label": "β=−0.15\n(direct discrimination)"}),
        ("Gender\n(Protected)","Income\n(Mediator)",   {"style": "normal"}),
        ("Education",          "Income\n(Mediator)",   {"style": "normal"}),
        ("Education",          "CreditSc\n(Mediator)", {"style": "normal"}),
        ("ZIP Code\n(Proxy)",  "CreditSc\n(Mediator)", {"style": "normal"}),
        ("Income\n(Mediator)", "Loan\nApproved",       {"style": "normal"}),
        ("CreditSc\n(Mediator)","Loan\nApproved",      {"style": "normal"}),
    ]

    draw_dag(ax, nodes, edges,
             title="Ground-Truth DAG — Biased Synthetic Dataset (β = −0.15)\n"
                   "Dashed red edge = planted direct discrimination (absent in Dataset B)",
             node_size=1600, font_size=7.5)
    dag_legend(ax)

    # Annotation box
    ax.text(4.05, -0.8,
            "Proxy discrimination path:\nRace→ZIP→CreditSc→Loan\n"
            "(present in both datasets)",
            fontsize=7, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e8", ec="#ccaa44", alpha=0.9))

    plt.tight_layout()
    save_pdf(fig, "fig_gt_biased.pdf")
    plt.close(fig)
    print("Figure 1 complete.")


if __name__ == "__main__":
    make_figure()
