"""
experiments/run_compas.py
=========================
Loads, preprocesses, and runs all causal-discovery algorithms on the
ProPublica COMPAS dataset (compas-scores.csv).

Preprocessing filters (ProPublica standard)
-------------------------------------------
  |days_b_screening_arrest| <= 30
  is_recid != -1
  c_charge_degree != 'O'
  score_text != 'N/A'
  decile_score != -1

Variables encoded (9 variables)
--------------------------------
  Race        : 1 = African-American, 0 = other
  Sex         : 1 = Male, 0 = Female
  Age         : continuous
  Juv_Fel     : juvenile felony count  (clipped 0-10)
  Juv_Misd    : juvenile misdemeanor count (clipped 0-10)
  Priors      : prior conviction count (clipped 0-30)
  Charge      : 1 = Felony, 0 = Misdemeanor
  COMPAS_Score: decile score 1-10
  Recidivism  : 1 = re-arrested within 2 years

Output
------
  results/compas_results.pkl      — algorithm results dict
  results/compas_preprocessed.csv — cleaned dataset

Run
---
  python experiments/run_compas.py --data data/compas-scores.csv

Colab: notebook 03_compas_study.ipynb
"""

import os, sys, argparse, warnings, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")

# Variable names in column order
COMPAS_VARS = ["Race", "Sex", "Age", "Juv_Fel", "Juv_Misd",
               "Priors", "Charge", "COMPAS_Score", "Recidivism"]

# Indices of key variables
RACE_IDX  = COMPAS_VARS.index("Race")
SCORE_IDX = COMPAS_VARS.index("COMPAS_Score")
PRIOR_IDX = COMPAS_VARS.index("Priors")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Apply ProPublica standard filters and encode variables."""
    raw = pd.read_csv(csv_path)
    print(f"Raw rows: {len(raw)}")

    df = raw.copy()
    # ProPublica filters
    df = df[np.abs(df["days_b_screening_arrest"]) <= 30]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]
    df = df[df["decile_score"] != -1]
    print(f"After ProPublica filters: {len(df)} rows")

    # Encode variables
    out = pd.DataFrame()
    out["Race"]         = (df["race"] == "African-American").astype(int)
    out["Sex"]          = (df["sex"] == "Male").astype(int)
    out["Age"]          = df["age"].astype(float)
    out["Juv_Fel"]      = df["juv_fel_count"].clip(0, 10).astype(float)
    out["Juv_Misd"]     = df["juv_misd_count"].clip(0, 10).astype(float)
    out["Priors"]       = df["priors_count"].clip(0, 30).astype(float)
    out["Charge"]       = (df["c_charge_degree"] == "F").astype(int)
    out["COMPAS_Score"] = df["decile_score"].astype(float)
    out["Recidivism"]   = df["is_recid"].astype(int)

    out = out.dropna().reset_index(drop=True)
    print(f"Final n: {len(out)}\n")

    # Print disparity statistics
    aa  = out[out["Race"] == 1]
    oth = out[out["Race"] == 0]
    print("── Disparity statistics ──────────────────────────────")
    print(f"  African-American n={len(aa)}, Other n={len(oth)}")
    print(f"  Mean COMPAS Score: AA={aa['COMPAS_Score'].mean():.2f}, "
          f"Other={oth['COMPAS_Score'].mean():.2f}")
    dir_score = aa["COMPAS_Score"].mean() / oth["COMPAS_Score"].mean()
    print(f"  Disparate Impact Ratio (Score): {dir_score:.3f}")
    print(f"  Recidivism rate: AA={aa['Recidivism'].mean():.3f}, "
          f"Other={oth['Recidivism'].mean():.3f}")
    dir_recid = aa["Recidivism"].mean() / oth["Recidivism"].mean()
    print(f"  Disparate Impact Ratio (Recidivism): {dir_recid:.3f}")
    print()
    return out


def standardize(df: pd.DataFrame) -> np.ndarray:
    X = df[COMPAS_VARS].values.astype(float)
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mu) / std


# ── Algorithm adjacency extractor ─────────────────────────────────────────────

def _adj_from_causallearn(result, n_vars):
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
        return (np.abs(result.adjacency_matrix_) > 0.08).astype(int)
    raise ValueError(f"Unknown type: {type(result)}")


def _edges(adj, var_names):
    return [(var_names[i], var_names[j])
            for i in range(adj.shape[0])
            for j in range(adj.shape[1]) if adj[i, j] == 1]


def _bidirected_edges(result, n_vars, var_names):
    """Extract bidirected edges from FCI PAG."""
    bidir = []
    try:
        arr = np.array(result.G.graph)
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if arr[i, j] == 1 and arr[j, i] == 1:
                    bidir.append((var_names[i], var_names[j]))
    except Exception:
        pass
    return bidir


# ── Main algorithm runner ──────────────────────────────────────────────────────

def run_compas_algorithms(df: pd.DataFrame) -> dict:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.PermutationBased.GRaSP import grasp
    from causallearn.search.FCMBased import lingam

    X_raw = df[COMPAS_VARS].values.astype(float)
    X_std = standardize(df)
    n_vars = len(COMPAS_VARS)
    results = {}

    def report(name, adj, beta_hat=None, extra=None):
        race_score = bool(adj[RACE_IDX, SCORE_IDX])
        race_prior = bool(adj[RACE_IDX, PRIOR_IDX])
        print(f"\n{'─'*55}")
        print(f"  {name}")
        print(f"  Race→COMPAS_Score: {'DETECTED ✓' if race_score else 'not detected ✗'}", end="")
        if beta_hat is not None:
            print(f"  β={beta_hat:+.4f}", end="")
        print()
        print(f"  Race→Priors:       {'DETECTED ✓' if race_prior else 'not detected ✗'}")
        print(f"  Edges: {_edges(adj, COMPAS_VARS)}")
        if extra:
            for k, v in extra.items():
                print(f"  {k}: {v}")
        r = {
            "adj_matrix":        adj,
            "race_score":        race_score,
            "race_prior":        race_prior,
            "edges":             _edges(adj, COMPAS_VARS),
            "beta_hat":          beta_hat,
        }
        if extra:
            r.update(extra)
        results[name] = r

    # ── PC ────────────────────────────────────────────────────────────────────
    print("\nRunning PC (max_cond_depth=3)...")
    try:
        res = pc(X_std, alpha=0.05, indep_test="fisherz",
                 depth=3, show_progress=False)
        report("PC", _adj_from_causallearn(res, n_vars))
    except Exception as e:
        print(f"  PC ERROR: {e}")

    # ── FCI ───────────────────────────────────────────────────────────────────
    print("\nRunning FCI (max_cond_depth=3)...")
    try:
        res, _ = fci(X_std, independence_test_method="fisherz",
                     alpha=0.05, depth=3, show_progress=False)
        adj = _adj_from_causallearn(res, n_vars)
        bidir = _bidirected_edges(res, n_vars, COMPAS_VARS)
        report("FCI", adj, extra={"bidirected_edges": bidir,
                                  "n_bidirected": len(bidir)})
    except Exception as e:
        print(f"  FCI ERROR: {e}")

    # ── GES ───────────────────────────────────────────────────────────────────
    print("\nRunning GES...")
    try:
        res = ges(X_std)
        report("GES", _adj_from_causallearn(res, n_vars))
    except Exception as e:
        print(f"  GES ERROR: {e}")

    # ── GRaSP ─────────────────────────────────────────────────────────────────
    print("\nRunning GRaSP...")
    try:
        res = grasp(X_std, score_func="local_score_BIC")
        adj = _adj_from_causallearn(res, n_vars)
        # Flag implausible edges (arrows INTO immutable demographics)
        implausible = [
            (COMPAS_VARS[i], COMPAS_VARS[j])
            for (i, j) in [(SCORE_IDX, RACE_IDX), (PRIOR_IDX, RACE_IDX),
                           (PRIOR_IDX, 1)]   # Score->Race, Priors->Race, Priors->Sex
            if adj[i, j] == 1
        ]
        report("GRaSP", adj, extra={"implausible_edges": implausible,
                                     "WARNING": "Check for causally incoherent edges"})
    except Exception as e:
        print(f"  GRaSP ERROR: {e}")

    # ── ICA-LiNGAM ────────────────────────────────────────────────────────────
    print("\nRunning ICA-LiNGAM...")
    try:
        model = lingam.ICALiNGAM(random_state=42, max_iter=2000, tol=5e-4)
        model.fit(X_std)
        adj = (np.abs(model.adjacency_matrix_) > 0.08).astype(int)
        beta_hat = float(model.adjacency_matrix_[SCORE_IDX, RACE_IDX])
        report("ICA-LiNGAM", adj, beta_hat=beta_hat,
               extra={"causal_order": [COMPAS_VARS[i] for i in model.causal_order_],
                      "B_matrix": model.adjacency_matrix_})
    except Exception as e:
        print(f"  ICA-LiNGAM ERROR: {e}")

    # ── DirectLiNGAM ──────────────────────────────────────────────────────────
    print("\nRunning DirectLiNGAM...")
    try:
        model = lingam.DirectLiNGAM()
        model.fit(X_std)
        adj = (np.abs(model.adjacency_matrix_) > 0.08).astype(int)
        beta_hat = float(model.adjacency_matrix_[SCORE_IDX, RACE_IDX])
        report("DirectLiNGAM", adj, beta_hat=beta_hat,
               extra={"causal_order": [COMPAS_VARS[i] for i in model.causal_order_],
                      "B_matrix": model.adjacency_matrix_})
    except Exception as e:
        print(f"  DirectLiNGAM ERROR: {e}")

    return results


def main(csv_path: str):
    print("=" * 60)
    print("COMPAS Causal Discovery Pipeline")
    print("=" * 60)

    df = load_and_preprocess(csv_path)

    # Save preprocessed CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "compas_preprocessed.csv"), index=False)

    results = run_compas_algorithms(df)

    # Attach the preprocessed dataframe for ATE script
    results["_dataframe"] = df
    results["_var_names"] = COMPAS_VARS

    out = os.path.join(RESULTS_DIR, "compas_results.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✓ COMPAS results saved to {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/compas-scores.csv",
                        help="Path to compas-scores.csv")
    args = parser.parse_args()
    main(args.data)
