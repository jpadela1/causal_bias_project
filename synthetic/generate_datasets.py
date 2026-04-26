"""
synthetic/generate_datasets.py
==============================
Generates and saves the canonical synthetic datasets used in the paper.

Outputs (saved to results/)
--------
  synthetic_n1000_biased.npy     Dataset A  (n=1000,  beta=-0.15)
  synthetic_n1000_unbiased.npy   Dataset B  (n=1000,  beta=0.00)
  synthetic_n50000_biased.npy    Dataset C  (n=50000, beta=-0.15)
  synthetic_n50000_unbiased.npy  Dataset D  (n=50000, beta=0.00)

Run
---
  python synthetic/generate_datasets.py

Or in Colab: run notebook 01_synthetic_study.ipynb
"""

import os
import numpy as np
import pandas as pd

# Allow running from project root OR from synthetic/ subdirectory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic.scm import (
    generate_loan_data, standardize,
    CANONICAL_N, VALIDATION_N, CANONICAL_SEED,
    BETA_BIASED, BETA_UNBIASED, VAR_NAMES
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")


def make_and_save(n: int, beta: float, label: str) -> np.ndarray:
    """Generate one dataset, print summary stats, and save."""
    X = generate_loan_data(n, beta, seed=CANONICAL_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"synthetic_{label}.npy")
    np.save(path, X)

    df = pd.DataFrame(X, columns=VAR_NAMES)
    print(f"\n{'='*60}")
    print(f"Dataset: {label}  (n={n}, beta={beta})")
    print(f"Saved to: {path}")
    print(f"\nColumn means:\n{df.mean().round(2)}")
    print(f"\nLoan approval rate: {df['LoanApprv'].mean():.3f}")
    print(f"Race=0 approval rate: {df[df['Race']==0]['LoanApprv'].mean():.3f}")
    print(f"Race=1 approval rate: {df[df['Race']==1]['LoanApprv'].mean():.3f}")
    print(f"Disparate Impact Ratio: "
          f"{df[df['Race']==1]['LoanApprv'].mean() / df[df['Race']==0]['LoanApprv'].mean():.3f}")
    return X


if __name__ == "__main__":
    print("Generating canonical synthetic datasets...")

    datasets = {
        "n1000_biased":    (CANONICAL_N,  BETA_BIASED),
        "n1000_unbiased":  (CANONICAL_N,  BETA_UNBIASED),
        "n50000_biased":   (VALIDATION_N, BETA_BIASED),
        "n50000_unbiased": (VALIDATION_N, BETA_UNBIASED),
    }

    for label, (n, beta) in datasets.items():
        make_and_save(n, beta, label)

    print("\n✓ All canonical datasets saved.")
