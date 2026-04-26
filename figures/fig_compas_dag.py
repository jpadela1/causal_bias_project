"""
figures/fig_compas_dag.py
==========================
Figure 4: Domain-knowledge causal DAG for COMPAS (hypothesized ground truth).
Based on ProPublica analysis and prior causal fairness literature.

Output: figures/output/fig_compas_dag.pdf
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from figures.fig_utils import (
    set_paper_style, draw_dag, dag_legend, save_pdf
)

set_paper_style()


def make_figure():
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    nodes = {
        "Race\n(Protected)":  {"pos": (0.0,  0.6),  "color_key": "protected"},
        "Sex\n(Protected)":   {"pos": (0.0, -0.4),  "color_key": "protected"},
        "Age":                {"pos": (0.0, -1.4),  "color_key": "covariate"},
        "Juv\nFelony":        {"pos": (1.5,  1.2),  "color_key": "mediator"},
        "Juv\nMisd":          {"pos": (1.5,  0.2),  "color_key": "mediator"},
        "Prior\nConvictions": {"pos": (2.8,  0.6),  "color_key": "proxy"},
        "Charge\nDegree":     {"pos": (1.5, -1.0),  "color_key": "covariate"},
        "COMPAS\nScore":      {"pos": (4.2,  0.1),  "color_key": "outcome"},
        "Recidivism\n(Actual)":{"pos":(5.5,  0.1),  "color_key": "outcome"},
    }

    edges = [
        # Bias hypotheses (dashed red)
        ("Race\n(Protected)",  "Prior\nConvictions",
         {"style": "bias", "label": "proxy discrimination\n(policing disparities)"}),
        ("Race\n(Protected)",  "COMPAS\nScore",
         {"style": "bias", "label": "direct discrimination\nhypothesis"}),
        # Legitimate structural edges
        ("Race\n(Protected)",  "Juv\nFelony",    {"style": "normal"}),
        ("Race\n(Protected)",  "Juv\nMisd",      {"style": "normal"}),
        ("Sex\n(Protected)",   "COMPAS\nScore",   {"style": "normal"}),
        ("Age",               "COMPAS\nScore",   {"style": "normal"}),
        ("Age",               "Prior\nConvictions",{"style":"normal"}),
        ("Juv\nFelony",       "Prior\nConvictions",{"style":"normal"}),
        ("Juv\nMisd",         "Prior\nConvictions",{"style":"normal"}),
        ("Prior\nConvictions","COMPAS\nScore",    {"style": "normal"}),
        ("Charge\nDegree",    "COMPAS\nScore",    {"style": "normal"}),
        ("COMPAS\nScore",     "Recidivism\n(Actual)",{"style":"normal"}),
    ]

    draw_dag(ax, nodes, edges,
             title="Domain-Knowledge Causal Graph — COMPAS Dataset (Hypothesized Ground Truth)\n"
                   "Based on ProPublica analysis & prior causal fairness literature",
             node_size=1700, font_size=7.0, arrow_size=18)
    dag_legend(ax)

    # Annotation
    ax.text(0.5, -0.12,
            "Dashed red edges are the two bias hypotheses tested by causal discovery algorithms.\n"
            "Direct path Race→Score = score inflation. Proxy path Race→Priors→Score = policing disparities.",
            transform=ax.transAxes, ha="center", fontsize=7, style="italic",
            bbox=dict(boxstyle="round", fc="#f8f8f8", ec="#cccccc", alpha=0.9))

    plt.tight_layout()
    save_pdf(fig, "fig_compas_dag.pdf")
    plt.close(fig)
    print("Figure 4 (COMPAS domain DAG) complete.")


if __name__ == "__main__":
    make_figure()
