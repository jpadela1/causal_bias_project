"""
experiments/dowhy_ate.py
========================
Estimates the Average Treatment Effect (ATE) of Race on COMPAS Score
at three levels of covariate adjustment using backdoor regression.

ATE levels
----------
  1. No controls (raw correlation)
  2. Age + Charge Degree
  3. Full criminal history (Age, Charge, Juv_Fel, Juv_Misd, Priors)

Also computes correlation-based fairness metrics for Figure 6 comparison.

Output
------
  results/ate_results.pkl  — dict with ATE values and fairness metrics

Run
---
  python experiments/dowhy_ate.py

Colab: run inside notebook 03_compas_study.ipynb (ATE section)
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")


def load_compas_df() -> pd.DataFrame:
    """Load preprocessed COMPAS data (run run_compas.py first)."""
    preprocessed = os.path.join(RESULTS_DIR, "compas_preprocessed.csv")
    if os.path.exists(preprocessed):
        return pd.read_csv(preprocessed)
    # Fallback: try to load from compas_results.pkl
    pkl = os.path.join(RESULTS_DIR, "compas_results.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            r = pickle.load(f)
        return r["_dataframe"]
    raise FileNotFoundError(
        "Run experiments/run_compas.py first to generate preprocessed data."
    )


def backdoor_ate(df: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 controls: list) -> float:
    """
    Estimate ATE via linear regression with backdoor adjustment.

    ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
        ≈ coefficient on treatment in OLS(outcome ~ treatment + controls)

    Parameters are standardized before regression so coefficients are
    in comparable standardized units.
    """
    cols = [treatment] + controls + [outcome]
    data = df[cols].dropna().copy()

    # Standardize
    mu  = data.mean()
    std = data.std().replace(0, 1)
    data_std = (data - mu) / std

    X = data_std[[treatment] + controls].values
    y = data_std[outcome].values

    reg = LinearRegression().fit(X, y)
    # Coefficient on treatment (index 0) = ATE in standardized units
    return float(reg.coef_[0])


def disparate_impact_ratio(df, group_col, outcome_col, privileged_val=0):
    """DIR = mean(outcome | unprivileged) / mean(outcome | privileged)."""
    unprivileged = df[df[group_col] != privileged_val][outcome_col].mean()
    privileged   = df[df[group_col] == privileged_val][outcome_col].mean()
    return float(unprivileged / privileged) if privileged != 0 else float("nan")


def statistical_parity(df, group_col, outcome_col):
    """SP difference = mean(outcome|group=1) - mean(outcome|group=0)."""
    return float(df[df[group_col]==1][outcome_col].mean()
               - df[df[group_col]==0][outcome_col].mean())


def main():
    df = load_compas_df()
    print(f"Loaded COMPAS data: n={len(df)}")

    # ── ATE estimation at three control levels ────────────────────────────────
    print("\n── Backdoor ATE: Race → COMPAS_Score ────────────────────────")
    ate_no_controls = backdoor_ate(df, "Race", "COMPAS_Score", [])
    print(f"  ATE (no controls):              {ate_no_controls:+.4f}")

    ate_age_charge = backdoor_ate(df, "Race", "COMPAS_Score",
                                  ["Age", "Charge"])
    print(f"  ATE (Age + Charge):             {ate_age_charge:+.4f}")

    ate_full = backdoor_ate(df, "Race", "COMPAS_Score",
                            ["Age", "Charge", "Juv_Fel", "Juv_Misd", "Priors"])
    print(f"  ATE (full criminal history):    {ate_full:+.4f}")

    # ── Correlation-based fairness metrics ────────────────────────────────────
    print("\n── Correlation-based Fairness Metrics ───────────────────────")

    # Binary loan-style: treat COMPAS_Score > 5 as "flagged high risk"
    df["HighRisk"] = (df["COMPAS_Score"] > 5).astype(int)

    dir_score = disparate_impact_ratio(df, "Race", "COMPAS_Score")
    print(f"  DIR (COMPAS Score, continuous): {dir_score:.4f}")

    dir_highrisk = disparate_impact_ratio(df, "Race", "HighRisk")
    print(f"  DIR (High Risk flag):           {dir_highrisk:.4f}")

    dir_recid = disparate_impact_ratio(df, "Race", "Recidivism")
    print(f"  DIR (Actual Recidivism):        {dir_recid:.4f}")

    sp_score = statistical_parity(df, "Race", "COMPAS_Score")
    print(f"  Stat. Parity (Score):           {sp_score:+.4f}")

    # Normalize SP to ratio for the figure
    aa_mean  = df[df["Race"]==1]["COMPAS_Score"].mean()
    oth_mean = df[df["Race"]==0]["COMPAS_Score"].mean()
    sp_ratio = aa_mean / oth_mean
    print(f"  Stat. Parity ratio (AA/Other):  {sp_ratio:.4f}")

    # ── Package results ───────────────────────────────────────────────────────
    results = {
        # ATE values (standardized units)
        "ate_no_controls": ate_no_controls,
        "ate_age_charge":  ate_age_charge,
        "ate_full":        ate_full,
        # Fairness metrics
        "dir_score":       dir_score,
        "dir_highrisk":    dir_highrisk,
        "dir_recid":       dir_recid,
        "sp_ratio":        sp_ratio,
        "aa_mean_score":   aa_mean,
        "oth_mean_score":  oth_mean,
        # For figure annotations
        "n":               len(df),
        "pct_aa":          df["Race"].mean(),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "ate_results.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\n✓ ATE results saved to {out}")
    return results


if __name__ == "__main__":
    main()
