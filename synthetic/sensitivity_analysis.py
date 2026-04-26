"""
synthetic/sensitivity_analysis.py
==================================
Grid sweep: beta × n × random seed × algorithm.

Grid
----
  beta  : [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
  n     : [500, 1000, 5000, 50000]
  seeds : 20 per cell  (total: 6 × 4 × 20 = 480 runs per algorithm)

For each (beta, n, seed) cell and each algorithm we record:
  - detected_biased   : bool  — Race->Loan present in recovered graph
  - detected_unbiased : bool  — Race->Loan absent in recovered graph (True = correct)
  - shd_biased        : int
  - shd_unbiased      : int
  - beta_hat_biased   : float or NaN (LiNGAM only)
  - beta_hat_unbiased : float or NaN (LiNGAM only)

Output
------
  results/sensitivity_results.pkl   — pd.DataFrame, all records

Run
---
  python synthetic/sensitivity_analysis.py [--jobs 4]

  --jobs  : number of parallel workers (default = 1; set -1 for all cores)

Colab note: run notebook 02_sensitivity_analysis.ipynb which sets --jobs=-1
and shows a live tqdm progress bar.
"""

import os, sys, argparse, pickle, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic.scm import (
    generate_loan_data, standardize,
    ground_truth_adj, VAR_NAMES
)

warnings.filterwarnings("ignore")

# ── Grid parameters ───────────────────────────────────────────────────────────
BETAS  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
NS     = [500, 1_000, 5_000, 50_000]
N_REPS = 20       # random seeds per cell
RACE_IDX = 0
LOAN_IDX = 6

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")


# ── Causal-learn helpers ──────────────────────────────────────────────────────

def _adj_from_causallearn(result, n_vars: int) -> np.ndarray:
    """Extract adjacency matrix from causal-learn result objects."""
    if hasattr(result, "G"):          # PC, FCI, GES, GRaSP
        G = result.G
        if hasattr(G, "graph"):
            arr = np.array(G.graph)
        else:
            arr = np.array(G)
        # causal-learn encodes: arr[i,j]=1 & arr[j,i]=-1 means i->j
        adj = np.zeros((n_vars, n_vars), dtype=int)
        for i in range(n_vars):
            for j in range(n_vars):
                if arr[i, j] == 1 and arr[j, i] == -1:
                    adj[i, j] = 1
        return adj
    elif hasattr(result, "adjacency_matrix_"):  # LiNGAM
        B = result.adjacency_matrix_
        return (np.abs(B) > 0.05).astype(int)
    else:
        raise ValueError(f"Unknown result type: {type(result)}")


def _compute_shd(est: np.ndarray, gt: np.ndarray) -> int:
    """Structural Hamming Distance between two adjacency matrices."""
    return int(np.sum(est != gt))


def _lingam_beta(result, cause_idx: int, effect_idx: int) -> float:
    """Extract β coefficient from LiNGAM result."""
    try:
        B = result.adjacency_matrix_
        return float(B[effect_idx, cause_idx])
    except Exception:
        return float("nan")


# ── Single-cell runner ────────────────────────────────────────────────────────

def run_one_cell(beta: float, n: int, seed: int) -> list:
    """
    Run all algorithms on one (beta, n, seed) cell.
    Returns a list of dicts, one per algorithm.
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.PermutationBased.GRaSP import grasp
    from causallearn.search.FCMBased import lingam

    # Per-cell seed: deterministic but independent across cells
    cell_seed = abs(hash((beta, n, seed))) % (2**31)

    X_biased   = generate_loan_data(n, beta,   seed=cell_seed)
    X_unbiased = generate_loan_data(n, 0.0,    seed=cell_seed + 1)
    Xs_biased   = standardize(X_biased)
    Xs_unbiased = standardize(X_unbiased)

    gt_b = ground_truth_adj(biased=True)
    gt_u = ground_truth_adj(biased=False)
    n_vars = X_biased.shape[1]
    records = []

    def record(alg_name, adj_b, adj_u, beta_hat_b=float("nan"), beta_hat_u=float("nan")):
        detected_b = bool(adj_b[RACE_IDX, LOAN_IDX])
        detected_u = bool(adj_u[RACE_IDX, LOAN_IDX])
        records.append({
            "algorithm":        alg_name,
            "beta":             beta,
            "n":                n,
            "seed":             seed,
            "detected_biased":  detected_b,
            "correct_unbiased": not detected_u,   # True = no false positive
            "shd_biased":       _compute_shd(adj_b, gt_b),
            "shd_unbiased":     _compute_shd(adj_u, gt_u),
            "beta_hat_biased":  beta_hat_b,
            "beta_hat_unbiased":beta_hat_u,
        })

    # ── PC ────────────────────────────────────────────────────────────────────
    try:
        res_b = pc(Xs_biased,   alpha=0.05, indep_test="fisherz")
        res_u = pc(Xs_unbiased, alpha=0.05, indep_test="fisherz")
        record("PC",
               _adj_from_causallearn(res_b, n_vars),
               _adj_from_causallearn(res_u, n_vars))
    except Exception as e:
        records.append({"algorithm": "PC", "beta": beta, "n": n, "seed": seed,
                        "error": str(e)})

    # ── FCI ───────────────────────────────────────────────────────────────────
    try:
        res_b, _ = fci(Xs_biased,   independence_test_method="fisherz", alpha=0.05)
        res_u, _ = fci(Xs_unbiased, independence_test_method="fisherz", alpha=0.05)
        record("FCI",
               _adj_from_causallearn(res_b, n_vars),
               _adj_from_causallearn(res_u, n_vars))
    except Exception as e:
        records.append({"algorithm": "FCI", "beta": beta, "n": n, "seed": seed,
                        "error": str(e)})

    # ── GES ───────────────────────────────────────────────────────────────────
    try:
        res_b = ges(Xs_biased)
        res_u = ges(Xs_unbiased)
        record("GES",
               _adj_from_causallearn(res_b, n_vars),
               _adj_from_causallearn(res_u, n_vars))
    except Exception as e:
        records.append({"algorithm": "GES", "beta": beta, "n": n, "seed": seed,
                        "error": str(e)})

    # ── GRaSP ─────────────────────────────────────────────────────────────────
    try:
        res_b = grasp(Xs_biased,   score_func="local_score_BIC")
        res_u = grasp(Xs_unbiased, score_func="local_score_BIC")
        record("GRaSP",
               _adj_from_causallearn(res_b, n_vars),
               _adj_from_causallearn(res_u, n_vars))
    except Exception as e:
        records.append({"algorithm": "GRaSP", "beta": beta, "n": n, "seed": seed,
                        "error": str(e)})

    # ── ICA-LiNGAM ────────────────────────────────────────────────────────────
    try:
        model_b = lingam.ICALiNGAM(random_state=cell_seed, max_iter=2000)
        model_b.fit(X_biased)
        model_u = lingam.ICALiNGAM(random_state=cell_seed, max_iter=2000)
        model_u.fit(X_unbiased)
        adj_b = (np.abs(model_b.adjacency_matrix_) > 0.05).astype(int)
        adj_u = (np.abs(model_u.adjacency_matrix_) > 0.05).astype(int)
        record("ICA-LiNGAM", adj_b, adj_u,
               _lingam_beta(model_b, RACE_IDX, LOAN_IDX),
               _lingam_beta(model_u, RACE_IDX, LOAN_IDX))
    except Exception as e:
        records.append({"algorithm": "ICA-LiNGAM", "beta": beta, "n": n,
                        "seed": seed, "error": str(e)})

    # ── DirectLiNGAM ──────────────────────────────────────────────────────────
    try:
        model_b = lingam.DirectLiNGAM()
        model_b.fit(X_biased)
        model_u = lingam.DirectLiNGAM()
        model_u.fit(X_unbiased)
        adj_b = (np.abs(model_b.adjacency_matrix_) > 0.05).astype(int)
        adj_u = (np.abs(model_u.adjacency_matrix_) > 0.05).astype(int)
        record("DirectLiNGAM", adj_b, adj_u,
               _lingam_beta(model_b, RACE_IDX, LOAN_IDX),
               _lingam_beta(model_u, RACE_IDX, LOAN_IDX))
    except Exception as e:
        records.append({"algorithm": "DirectLiNGAM", "beta": beta, "n": n,
                        "seed": seed, "error": str(e)})

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sensitivity(n_jobs: int = 1) -> pd.DataFrame:
    """Run the full grid and return results as a DataFrame."""
    cells = [
        (beta, n, seed)
        for beta in BETAS
        for n    in NS
        for seed in range(N_REPS)
    ]
    print(f"Running {len(cells)} cells "
          f"({len(BETAS)} betas × {len(NS)} sample sizes × {N_REPS} seeds) "
          f"with {len(['PC','FCI','GES','GRaSP','ICA-LiNGAM','DirectLiNGAM'])} algorithms each.")
    print(f"Parallel workers: {n_jobs}\n")

    all_records = []

    if n_jobs == 1:
        for beta, n, seed in tqdm(cells, desc="Sensitivity grid"):
            all_records.extend(run_one_cell(beta, n, seed))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(run_one_cell)(beta, n, seed)
            for beta, n, seed in cells
        )
        for r in results:
            all_records.extend(r)

    df = pd.DataFrame(all_records)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "sensitivity_results.pkl")
    df.to_pickle(out)
    print(f"\n✓ Sensitivity results saved to {out}")
    print(f"  Rows: {len(df)}  |  Algorithms: {df['algorithm'].unique()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel workers (-1 = all cores)")
    args = parser.parse_args()
    run_sensitivity(n_jobs=args.jobs)
