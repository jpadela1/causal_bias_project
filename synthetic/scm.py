"""
synthetic/scm.py
================
Single source of truth for the loan-approval Structural Causal Model (SCM).

Both the canonical dataset generator and the sensitivity analysis import
from this module, guaranteeing they use identical data-generating equations.

Variables
---------
0  Race       Bernoulli(0.32)  — protected attribute (1 = minority)
1  Gender     Bernoulli(0.50)  — protected attribute (1 = female)
2  Education  N(16, 4)         — years of education
3  ZIP        f(Race, Educ)    — proxy for race (redlining)
4  Income     f(Race, Gender, Educ) — mediator
5  CreditSc   f(ZIP, Educ, Income)  — mediator
6  LoanApprv  f(Income, CreditSc, Race*beta) — outcome

Ground truth edges (Dataset A — biased, beta = -0.15):
  Race -> ZIP, Race -> Income, Race -> LoanApprv (direct discrimination)
  Gender -> Income
  Education -> Income, Education -> CreditSc
  ZIP -> CreditSc
  Income -> LoanApprv
  CreditSc -> LoanApprv

Ground truth edges (Dataset B — unbiased, beta = 0.00):
  Same as above WITHOUT Race -> LoanApprv

Hidden confounder:
  SES (not in dataframe) -> Education, SES -> Income
  FCI should detect the Education <-> Income bidirected edge.
"""

import numpy as np

# ── Named constants (import these instead of hard-coding numbers) ───────────
CANONICAL_N       = 1_000
VALIDATION_N      = 50_000
CANONICAL_SEED    = 42
BETA_BIASED       = -0.15    # planted direct discrimination coefficient
BETA_UNBIASED     =  0.00

# Variable names in column order — used by all scripts for labels
VAR_NAMES = ["Race", "Gender", "Education", "ZIP", "Income", "CreditSc", "LoanApprv"]

# Ground truth adjacency matrix gt[i, j] = 1 means edge i -> j
# Rows/cols: Race=0, Gender=1, Education=2, ZIP=3, Income=4, CreditSc=5, LoanApprv=6
def ground_truth_adj(biased: bool = True) -> np.ndarray:
    """Return the 7x7 ground truth adjacency matrix."""
    gt = np.zeros((7, 7), dtype=int)
    # Race -> ZIP, Income, (LoanApprv if biased)
    gt[0, 3] = 1   # Race -> ZIP
    gt[0, 4] = 1   # Race -> Income
    if biased:
        gt[0, 6] = 1   # Race -> LoanApprv  (direct discrimination)
    # Gender -> Income
    gt[1, 4] = 1
    # Education -> Income, CreditSc
    gt[2, 4] = 1
    gt[2, 5] = 1
    # ZIP -> CreditSc
    gt[3, 5] = 1
    # Income -> LoanApprv
    gt[4, 6] = 1
    # CreditSc -> LoanApprv
    gt[5, 6] = 1
    return gt


def generate_loan_data(n: int, beta: float, seed: int) -> np.ndarray:
    """
    Generate synthetic loan-approval data from the SCM.

    Parameters
    ----------
    n    : number of observations
    beta : Race -> LoanApprv coefficient (-0.15 = biased, 0.0 = unbiased)
    seed : random seed (use CANONICAL_SEED=42 for the paper's main datasets)

    Returns
    -------
    X : np.ndarray, shape (n, 7), columns in VAR_NAMES order
        Raw (un-standardized) data — feed directly to LiNGAM for interpretable β.
        Standardize before feeding to PC / FCI / GES / GRaSP.
    """
    rng = np.random.default_rng(seed)

    # Exogenous variables
    race   = rng.binomial(1, 0.32, n).astype(float)
    gender = rng.binomial(1, 0.50, n).astype(float)

    # Hidden SES confounder (not in the dataframe — FCI should detect residual)
    ses    = rng.normal(0, 1, n)

    # Education: influenced by hidden SES
    educ   = 16 + 4 * ses + rng.normal(0, 1, n)

    # ZIP: proxy for race (lower zip base for minority group)
    zip_code = -3.0 * race + 0.5 * educ + rng.normal(0, 1, n)

    # Income: influenced by race, gender, education, and hidden SES
    income = (30_000
              - 4_000 * race
              - 3_000 * gender
              + 2_000 * educ
              + 1_500 * ses          # SES -> Income (hidden path)
              + rng.normal(0, 3_000, n))

    # Credit score: influenced by ZIP, education, income
    credsc = np.clip(
        650 + 20 * zip_code + 10 * educ - 0.02 * income
        + rng.normal(0, 30, n),
        300, 850
    )

    # Loan approval score (continuous; binarize at 0)
    loan_score = (0.003 * (income - 30_000)
                  + 0.004 * (credsc - 650)
                  + 0.2   * educ
                  + beta  * race
                  + rng.normal(0, 0.1, n))
    loan = (loan_score > 0).astype(float)

    return np.column_stack([race, gender, educ, zip_code, income, credsc, loan])


def standardize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance standardization (for PC / FCI / GES / GRaSP)."""
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0   # guard against constant columns
    return (X - mu) / std
