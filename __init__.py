"""
experiments/run_synthetic.py
============================
Runs all six causal-discovery algorithms on the four canonical synthetic
datasets and saves results to results/synthetic_results.pkl.

Output columns
--------------
  dataset      : str   — e.g. "n1000_biased"
  algorithm    : str
  adj_matrix   : np.ndarray (7x7)
  shd          : int
  detected     : bool  — Race->LoanApprv in recovered graph
  beta_hat     : float — LiNGAM only, else NaN
  edges        : list of (i, j) tuples

Run
---
  python experiments/run_synthetic.py

Colab: notebook 01_synthetic_study.ipynb
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic.scm import (
    generate_loan_data, standardize, ground_truth_adj,
    CANONICAL_N, VALIDATION_N, CANONICAL_SEED,
    BETA_BIASED, BETA_UNBIASED, VAR_NAMES
)

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")
RACE_IDX = 0
LOAN_IDX = 6


# ── Shared adjacency extractor (same as sensitivity_analysis.py) ──────────────

def _adj_from_causallearn(result, n_vars=7):
    if hasattr(result, "G"):
        G = result.G
        arr = np.array(G.graph if hasattr(G, "graph") else G)
        adj = np.zeros((n_vars, n_vars), dtype=int)
        for i in range(n_vars):
            for j in range(n_vars):
                if arr[i, j] == 1 and arr[j, i] == -1:
                    adj[i, j] = 1
        return adj
    elif hasattr(result, "adjacency_matrix_"):
        return (np.abs(result.adjacency_matrix_) > 0.05).astype(int)
    raise ValueError(f"Unknown result type: {type(result)}")


def _shd(est, gt):
    return int(np.sum(est != gt))


def _edges(adj):
    return [(VAR_NAMES[i], VAR_NAMES[j])
            for i in range(adj.shape[0])
            for j in range(adj.shape[1]) if adj[i, j] == 1]


def run_all_algorithms(X_raw, X_std, label, biased):
    """Run all 6 algorithms and return list of result dicts."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.PermutationBased.GRaSP import grasp
    from causallearn.search.FCMBased import lingam

    gt = ground_truth_adj(biased=biased)
    n_vars = X_raw.shape[1]
    records = []

    def add(alg, adj, beta_hat=float("nan"), extra=None):
        r = {
            "dataset":    label,
            "algorithm":  alg,
            "adj_matrix": adj,
            "shd":        _shd(adj, gt),
            "detected":   bool(adj[RACE_IDX, LOAN_IDX]),
            "beta_hat":   beta_hat,
            "edges":      _edges(adj),
        }
        if extra:
            r.update(extra)
        records.append(r)
        print(f"  {alg:20s}  SHD={r['shd']:2d}  "
              f"Race->Loan={'YES' if r['detected'] else 'no ':3s}  "
              f"β={beta_hat:+.4f}" if not np.isnan(beta_hat) else
              f"  {alg:20s}  SHD={r['shd']:2d}  "
              f"Race->Loan={'YES' if r['detected'] else 'no '}")

    print(f"\n{'─'*60}")
    print(f"Dataset: {label}")
    print(f"{'─'*60}")

    # PC
    try:
        res = pc(X_std, alpha=0.05, indep_test="fisherz", show_progress=False)
        add("PC", _adj_from_causallearn(res, n_vars))
    except Exception as e:
        print(f"  PC  ERROR: {e}")

    # FCI
    try:
        res, edges = fci(X_std, independence_test_method="fisherz",
                         alpha=0.05, show_progress=False)
        adj = _adj_from_causallearn(res, n_vars)
        # Count bidirected edges (latent confounders)
        if hasattr(res, "graph"):
            arr = np.array(res.graph)
            n_bidir = int(np.sum((arr == 1) & (arr.T == 1)) // 2)
        else:
            n_bidir = 0
        add("FCI", adj, extra={"n_bidirected": n_bidir})
    except Exception as e:
        print(f"  FCI ERROR: {e}")

    # GES
    try:
        res = ges(X_std)
        add("GES", _adj_from_causallearn(res, n_vars))
    except Exception as e:
        print(f"  GES ERROR: {e}")

    # GRaSP
    try:
        res = grasp(X_std, score_func="local_score_BIC")
        add("GRaSP", _adj_from_causallearn(res, n_vars))
    except Exception as e:
        print(f"  GRaSP ERROR: {e}")

    # ICA-LiNGAM (raw data, not standardized)
    try:
        model = lingam.ICALiNGAM(random_state=CANONICAL_SEED, max_iter=2000)
        model.fit(X_raw)
        adj = (np.abs(model.adjacency_matrix_) > 0.05).astype(int)
        beta_hat = float(model.adjacency_matrix_[LOAN_IDX, RACE_IDX])
        add("ICA-LiNGAM", adj, beta_hat=beta_hat,
            extra={"B_matrix": model.adjacency_matrix_,
                   "causal_order": model.causal_order_})
    except Exception as e:
        print(f"  ICA-LiNGAM ERROR: {e}")

    # DirectLiNGAM (raw data)
    try:
        model = lingam.DirectLiNGAM()
        model.fit(X_raw)
        adj = (np.abs(model.adjacency_matrix_) > 0.05).astype(int)
        beta_hat = float(model.adjacency_matrix_[LOAN_IDX, RACE_IDX])
        add("DirectLiNGAM", adj, beta_hat=beta_hat,
            extra={"B_matrix": model.adjacency_matrix_,
                   "causal_order": model.causal_order_})
    except Exception as e:
        print(f"  DirectLiNGAM ERROR: {e}")

    return records


def main():
    datasets = [
        ("n1000_biased",    CANONICAL_N,  BETA_BIASED,   True),
        ("n1000_unbiased",  CANONICAL_N,  BETA_UNBIASED,  False),
        ("n50000_biased",   VALIDATION_N, BETA_BIASED,   True),
        ("n50000_unbiased", VALIDATION_N, BETA_UNBIASED,  False),
    ]

    all_records = []
    for label, n, beta, biased in datasets:
        X_raw = generate_loan_data(n, beta, seed=CANONICAL_SEED)
        X_std = standardize(X_raw)
        all_records.extend(run_all_algorithms(X_raw, X_std, label, biased))

    df = pd.DataFrame(all_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "synthetic_results.pkl")
    df.to_pickle(out)
    print(f"\n✓ Saved {len(df)} records to {out}")
    return df


if __name__ == "__main__":
    main()
