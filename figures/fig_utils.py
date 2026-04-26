"""
figures/fig_utils.py
====================
Shared plotting utilities for all paper figures.

All figures are saved as vector PDFs for full reviewer zoomability.
Color palette, font sizes, and node styles are defined here once
and imported by every figure script.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── Backend: use non-interactive Agg for Colab / server rendering ─────────────
matplotlib.use("Agg")

# ── Output directory ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "figures", "output")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Paper-quality matplotlib settings ────────────────────────────────────────
def set_paper_style():
    """Apply IEEE-paper-appropriate matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    1.5,
        "axes.linewidth":     0.8,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.4,
    })


# ── Node color palette (consistent across all DAG figures) ────────────────────
NODE_COLORS = {
    "protected":  "#8ab4d4",   # blue — Race, Sex/Gender
    "proxy":      "#f0a868",   # orange — ZIP, Priors
    "mediator":   "#90c88c",   # green — Income, CreditSc, Juv_Fel, Juv_Misd
    "outcome":    "#e8a0a0",   # pink — LoanApprv, COMPAS_Score, Recidivism
    "covariate":  "#c8c8c8",   # grey — Age, Charge, Education
}
BIAS_EDGE_COLOR  = "#cc2222"   # red dashed — flagged bias path
LATENT_EDGE_COLOR = "#7060b0"  # purple — FCI bidirected / latent confounder
NORMAL_EDGE_COLOR = "#333333"  # dark grey — standard edges


# ── DAG drawing helper ─────────────────────────────────────────────────────────

def draw_dag(
    ax,
    nodes: dict,          # {name: {"pos": (x,y), "color_key": str}}
    edges: list,          # [(src, dst, {"style": "normal"|"bias"|"latent", "label": str})]
    title: str = "",
    node_size: int = 1800,
    font_size: int = 8,
    arrow_size: int = 20,
):
    """
    Draw a DAG on the given matplotlib Axes using networkx + matplotlib patches.

    Parameters
    ----------
    nodes : dict mapping node name -> {"pos": (x, y), "color_key": str}
    edges : list of (src, dst, attr_dict)
            attr_dict keys: "style" in {"normal","bias","latent"}, "label" str
    """
    G = nx.DiGraph()
    pos = {}
    node_colors = []

    for name, attrs in nodes.items():
        G.add_node(name)
        pos[name] = attrs["pos"]
        node_colors.append(NODE_COLORS.get(attrs.get("color_key", "covariate"),
                                           "#cccccc"))

    for src, dst, attrs in edges:
        G.add_edge(src, dst)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_size, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                            font_weight="bold")

    # Draw edges by style
    for src, dst, attrs in edges:
        style   = attrs.get("style", "normal")
        label   = attrs.get("label", "")
        beta    = attrs.get("beta", None)

        if style == "bias":
            color    = BIAS_EDGE_COLOR
            ls       = "dashed"
            width    = 2.0
            alpha    = 0.9
        elif style == "latent":
            color    = LATENT_EDGE_COLOR
            ls       = "dashed"
            width    = 1.5
            alpha    = 0.8
        else:
            color    = NORMAL_EDGE_COLOR
            ls       = "solid"
            width    = 1.2
            alpha    = 0.75

        nx.draw_networkx_edges(
            G, pos, edgelist=[(src, dst)], ax=ax,
            edge_color=color, style=ls, width=width, alpha=alpha,
            arrows=True, arrowsize=arrow_size,
            connectionstyle="arc3,rad=0.08",
            min_source_margin=25, min_target_margin=25,
        )

        # Edge label (β coefficient)
        if beta is not None:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"β={beta:+.3f}", fontsize=6,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.7))

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.axis("off")


# ── Legend builder ─────────────────────────────────────────────────────────────

def dag_legend(ax, include_latent=False):
    """Add a standard node/edge type legend to the given axes."""
    patches = [
        mpatches.Patch(color=NODE_COLORS["protected"], label="Protected attribute"),
        mpatches.Patch(color=NODE_COLORS["proxy"],     label="Proxy / high-leverage"),
        mpatches.Patch(color=NODE_COLORS["mediator"],  label="Mediator"),
        mpatches.Patch(color=NODE_COLORS["outcome"],   label="Outcome / score"),
        mpatches.Patch(color=NODE_COLORS["covariate"], label="Covariate"),
    ]
    line_patches = [
        matplotlib.lines.Line2D([0],[0], color=BIAS_EDGE_COLOR,
                                 linestyle="--", lw=2, label="Flagged bias path"),
        matplotlib.lines.Line2D([0],[0], color=NORMAL_EDGE_COLOR,
                                 linestyle="-",  lw=1.5, label="Causal edge"),
    ]
    if include_latent:
        line_patches.append(
            matplotlib.lines.Line2D([0],[0], color=LATENT_EDGE_COLOR,
                                     linestyle="--", lw=1.5, label="Latent confounder (↔)")
        )
    ax.legend(handles=patches + line_patches,
              loc="lower left", fontsize=7, framealpha=0.85,
              ncol=1, handlelength=1.5)


# ── Save helper ───────────────────────────────────────────────────────────────

def save_pdf(fig, filename: str, also_png: bool = True):
    """
    Save figure as vector PDF (for paper) and optionally PNG (for preview).
    The PDF is fully zoomable in any PDF viewer.
    """
    pdf_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"  ✓ Saved PDF: {pdf_path}")

    if also_png:
        png_path = pdf_path.replace(".pdf", ".png")
        fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
        print(f"  ✓ Saved PNG: {png_path}")

    return pdf_path
