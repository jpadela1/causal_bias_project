# Causal Discovery for Bias Detection — Research Code

## Project Structure

```
causal_bias_project/
│
├── README.md                        ← this file
├── requirements.txt                 ← pip install -r requirements.txt
│
├── synthetic/
│   ├── scm.py                       ← Structural Causal Model (single source of truth)
│   ├── generate_datasets.py         ← canonical Dataset A & B (n=1000, n=50000)
│   └── sensitivity_analysis.py     ← grid sweep β × n × seeds
│
├── experiments/
│   ├── run_synthetic.py             ← all 6 algorithms on synthetic data
│   ├── run_compas.py                ← all algorithms on COMPAS
│   └── dowhy_ate.py                 ← backdoor ATE estimation
│
├── figures/
│   ├── fig_utils.py                 ← shared plotting helpers
│   ├── fig_gt_biased.py             ← Figure 1: ground truth DAG (biased)
│   ├── fig_sensitivity.py           ← Figure 2: sensitivity analysis grid
│   ├── fig_directlingam_loan.py     ← Figure 3: DirectLiNGAM synthetic
│   ├── fig_compas_dag.py            ← Figure 4: COMPAS domain-knowledge DAG
│   ├── fig_directlingam_compas.py   ← Figure 5: DirectLiNGAM COMPAS
│   └── fig_corr_vs_causal.py       ← Figure 6: correlation vs causal bar chart
│
├── notebooks/
│   ├── 01_synthetic_study.ipynb     ← Google Colab: Study 1 (synthetic)
│   ├── 02_sensitivity_analysis.ipynb← Google Colab: Sensitivity grid
│   ├── 03_compas_study.ipynb        ← Google Colab: Study 2 (COMPAS)
│   └── 04_generate_all_figures.ipynb← Google Colab: render all paper figures
│
└── data/
    └── compas-scores.csv            ← ProPublica COMPAS dataset
```

## Google Colab Quick Start

Each notebook in `notebooks/` is self-contained and runs top-to-bottom
in Google Colab. Open a notebook, run the first cell (installs packages
and clones/mounts the project), then run all remaining cells.

**Recommended run order:**
1. `01_synthetic_study.ipynb`     — generates `results/synthetic_results.pkl`
2. `02_sensitivity_analysis.ipynb`— generates `results/sensitivity_results.pkl`
3. `03_compas_study.ipynb`        — generates `results/compas_results.pkl`
4. `04_generate_all_figures.ipynb`— reads all pkl files, writes PDF figures

## Requirements

See `requirements.txt`. Key packages:
- `causal-learn >= 0.1.3.8`
- `dowhy >= 0.11`
- `matplotlib >= 3.7`
- `networkx >= 3.0`
- `pandas`, `numpy`, `scipy`, `scikit-learn`
