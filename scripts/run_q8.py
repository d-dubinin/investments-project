"""
Orchestration script for Question 8 (Out-of-sample Portfolio Analysis).

Prerequisites
-------------
Run scripts/fetch_data.py first so that
  data/raw/fx_monthly_panel.csv
  data/raw/ir_monthly_wide.csv
exist in the project root.

Usage
-----
  cd investments-project
  python scripts/run_q8.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from tabulate import tabulate

from core.excess_return import ExcessReturnCalculator
from portfolios.oos import (
    CURRENCIES,
    EVAL_START,
    to_period_wide,
    compute_ewma_covariance,
    compute_expanding_mean,
    compute_delta_r,
    compute_momentum_signal,
    run_fm_regressions,
    compute_fm_forecast,
    compute_cmve_portfolio_returns,
    performance_table,
    fm_coefficient_table,
)

DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "output"
DATA_OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data …")
fx_panel   = pd.read_csv(DATA_RAW / "fx_monthly_panel.csv",  parse_dates=["date"])
rates_wide = pd.read_csv(DATA_RAW / "ir_monthly_wide.csv",   parse_dates=["date"])

# ── 2. Compute excess returns ─────────────────────────────────────────────────
calc = ExcessReturnCalculator()
returns_full = calc.compute(fx_panel, rates_wide)
holdout, evaluation = calc.split(returns_full)

# Full-sample wide returns (Jan 1991 – Dec 2024), needed for EWMA and expanding mean
returns_wide_full = to_period_wide(returns_full)

# ── 3. Q8a: EWMA Conditional Covariance ──────────────────────────────────────
print("\nQ8a – Computing EWMA covariance matrices (λ = 0.97) …")
sigma_t = compute_ewma_covariance(returns_wide_full, lam=0.97)

# Quick sanity check: Σ at Dec 1994 should be positive definite
sigma_dec94 = sigma_t[pd.Period("1994-12", "M")]
eigvals_dec94 = np.linalg.eigvalsh(sigma_dec94)
print(f"  Σ̂(Dec-1994) min eigenvalue : {eigvals_dec94.min():.2e}  "
      f"(positive = PD ✓)" if eigvals_dec94.min() > 0 else "  WARNING: not PD")

# ── 4. Q8b: Expanding-mean Conditional Mean ───────────────────────────────────
print("\nQ8b – Computing expanding-mean forecasts …")
mu_exp = compute_expanding_mean(returns_wide_full)
print(f"  Expanding mean shape : {mu_exp.shape}")
print(f"  First valid period   : {mu_exp.index[0]}")

# ── 5. Signals for Fama-MacBeth ───────────────────────────────────────────────
print("\nQ8c – Preparing FM signals …")
delta_r  = compute_delta_r(rates_wide)
momentum = compute_momentum_signal(fx_panel)

print(f"  δr   first / last period : {delta_r.index[0]} / {delta_r.index[-1]}")
print(f"  MOM  first valid period  : {momentum.dropna(how='all').index[0]}")

# ── 6. Run Fama-MacBeth regressions ───────────────────────────────────────────
print("\nRunning FM cross-sectional regressions …")
fm_estimates = run_fm_regressions(returns_wide_full, delta_r, momentum)
print(f"  FM estimates shape : {fm_estimates.shape}")
print(f"  Signal-month range : {fm_estimates.index[0]} → {fm_estimates.index[-1]}")

# ── 7. Q8d: FM Coefficient Statistics ────────────────────────────────────────
print("\n─── Q8d: Fama-MacBeth Coefficients (evaluation sample) ─────────────────")
coef_table = fm_coefficient_table(fm_estimates)
print(tabulate(coef_table, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
print()
print("Interpretation:")
print("  γ = sensitivity of excess return to carry signal (Δr^i).")
print("  φ = sensitivity of excess return to momentum signal (s^i_{t-12,t}).")
print("  γ units: monthly decimal / annual %  |  φ units: monthly decimal / decimal ratio.")

# ── 8. Q8c: FM Mean Forecast ──────────────────────────────────────────────────
print("\nQ8c – Computing FM forecasts for evaluation sample …")

# Formation months span Dec 1994 – Nov 2024 (portfolio returns: Jan 1995 – Dec 2024)
eval_return_periods = returns_wide_full.index[returns_wide_full.index >= EVAL_START]
formation_months    = pd.PeriodIndex([p - 1 for p in eval_return_periods], freq="M")

mu_fm = compute_fm_forecast(fm_estimates, delta_r, momentum, formation_months)
print(f"  FM forecast shape       : {mu_fm.shape}")
non_nan_rows = mu_fm.dropna(how="any").shape[0]
print(f"  Non-NaN formation months: {non_nan_rows}")

# ── 9. Q8e: CCV-Exp and CCV-FM Portfolios ────────────────────────────────────
print("\nQ8e – Constructing CCV-Exp and CCV-FM portfolios …")
ccv_returns = compute_cmve_portfolio_returns(
    returns_wide  = returns_wide_full,
    sigma_t       = sigma_t,
    mu_exp        = mu_exp,
    mu_fm         = mu_fm,
    target_vol    = 0.10,
)

print(f"  Portfolio shape    : {ccv_returns.shape}  (expected 360 rows)")
print(f"  Date range         : {ccv_returns.index.min().date()} → {ccv_returns.index.max().date()}")
print(f"  CCV-Exp NaN count  : {ccv_returns['CCV-Exp'].isna().sum()}")
print(f"  CCV-FM  NaN count  : {ccv_returns['CCV-FM'].isna().sum()}")

# ── 10. Q8f: Table 4 ──────────────────────────────────────────────────────────
print("\n═══ Table 4: Out-of-sample Portfolio Performance (Jan 1995 – Dec 2024) ═══")
table4 = performance_table(ccv_returns, ["CCV-Exp", "CCV-FM"])
print(tabulate(table4, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

# ── 11. Save outputs ──────────────────────────────────────────────────────────
ccv_returns.to_csv(DATA_OUT / "question8_ccv_returns.csv")
fm_estimates.to_csv(DATA_OUT / "question8_fm_estimates.csv")
coef_table.to_csv(DATA_OUT / "question8_fm_coeff_table.csv")
table4.to_csv(DATA_OUT / "table4_question8.csv")
print(f"\nSaved outputs to {DATA_OUT}/")
