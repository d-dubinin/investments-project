import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from tabulate import tabulate

from core.excess_return import ExcessReturnCalculator
from strategies.momentum import (
    compute_momentum_strategies,
    compute_strategy_summary,
    regress_ts_carry_on_momentum,
)
from strategies.dollar import (
    compute_dollar_strategies,
    regress_carry_on_momentum_and_dollar,
)
from portfolios.construction import compute_question7_portfolios
from portfolios.oos import (
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

import numpy as np

DATA_RAW = Path("data/raw")
DATA_OUT = Path("data/output")
DATA_OUT.mkdir(parents=True, exist_ok=True)

CURRENCIES = ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
fx_panel   = pd.read_csv(DATA_RAW / "fx_monthly_panel.csv", parse_dates=["date"])
rates_wide = pd.read_csv(DATA_RAW / "ir_monthly_wide.csv",  parse_dates=["date"])

calc = ExcessReturnCalculator()
returns      = calc.compute(fx_panel, rates_wide)
holdout, evaluation = calc.split(returns)

# ── Q1–Q4 ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Q1–Q4: Excess Returns & Carry Strategies")
print("="*60)

table1 = calc.compute_summary(evaluation, rates_wide)
print("\nTable 1: Monthly Summary Statistics")
print(tabulate(table1, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

carry_returns, carry_weights = calc.compute_carry(evaluation, rates_wide)

table2 = calc.compute_carry_summary(carry_returns)
print("\nTable 2: Carry Summary Statistics")
print(tabulate(table2, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

calc.plot_carry_weights(carry_weights)

corr = calc.compute_forward_discount_correlation(fx_panel, rates_wide)
print("\nCross-sectional correlations")
print(tabulate(corr, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

# ── Q5–Q6 ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Q5–Q6: Momentum & Dollar Strategies")
print("="*60)

momentum_returns, momentum_weights = compute_momentum_strategies(
    fx_panel=fx_panel,
    returns=evaluation,
    currencies=CURRENCIES,
)

factor_returns = carry_returns.merge(momentum_returns, on="date", how="inner")

table2_carry_mom = compute_strategy_summary(
    factor_returns,
    columns=["R_CS_CARRY", "R_TS_CARRY", "R_CS_MOM", "R_TS_MOM"],
)
print("\nTable 2: Carry and Momentum Strategy Summary Statistics")
print(tabulate(table2_carry_mom, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

reg_table_5b, r2_5b = regress_ts_carry_on_momentum(carry_returns, momentum_returns)
print("\nQ5(b): Regression of TS-CARRY on CS-MOM and TS-MOM")
print(tabulate(reg_table_5b, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
print(f"\nR^2 = {r2_5b:.4f}")

dollar_returns = compute_dollar_strategies(
    returns=evaluation,
    rates_wide=rates_wide,
    currencies=CURRENCIES,
)

factor_returns_full = (
    carry_returns
    .merge(momentum_returns, on="date", how="inner")
    .merge(dollar_returns,   on="date", how="inner")
)

table2_full = compute_strategy_summary(
    factor_returns_full,
    columns=["R_CS_CARRY", "R_TS_CARRY", "R_CS_MOM", "R_TS_MOM", "R_DOLLAR", "R_DOLLAR_CARRY"],
)
print("\nTable 2: Carry, Momentum, and Dollar Strategy Summary Statistics")
print(tabulate(table2_full, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

reg_table_6b, r2_6b = regress_carry_on_momentum_and_dollar(carry_returns, momentum_returns, dollar_returns)
print("\nQ6(b): Regression of TS-CARRY on CS-MOM, TS-MOM, and DOLLAR")
print(tabulate(reg_table_6b, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
print(f"\nR^2 = {r2_6b:.4f}")

# ── Q7 ────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Q7: In-sample Portfolio Analysis")
print("="*60)

table3, q7_returns, q7_weights = compute_question7_portfolios(
    returns=evaluation,
    carry_returns=carry_returns,
    momentum_returns=momentum_returns,
    dollar_returns=dollar_returns,
    currencies=CURRENCIES,
)

print("\nTable 3: In-sample Portfolio Analysis")
print(tabulate(table3, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

q7_returns.to_csv(DATA_OUT / "question7_portfolio_returns.csv", index=False)
table3.to_csv(DATA_OUT / "table3_question7.csv")
q7_weights.to_csv(DATA_OUT / "question7_weights.csv")

# ── Q8 ────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Q8: Out-of-sample Portfolio Analysis")
print("="*60)

returns_wide_full = to_period_wide(returns)

print("\nQ8a – Computing EWMA covariance matrices (λ = 0.97) …")
sigma_t = compute_ewma_covariance(returns_wide_full, lam=0.97)

print("\nQ8b – Computing expanding-mean forecasts …")
mu_exp = compute_expanding_mean(returns_wide_full)

print("\nQ8c – Preparing FM signals …")
delta_r  = compute_delta_r(rates_wide)
momentum = compute_momentum_signal(fx_panel)

print("\nRunning FM cross-sectional regressions …")
fm_estimates = run_fm_regressions(returns_wide_full, delta_r, momentum)

print("\nQ8d: Fama-MacBeth Coefficients (evaluation sample)")
coef_table = fm_coefficient_table(fm_estimates)
print(tabulate(coef_table, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

eval_return_periods = returns_wide_full.index[returns_wide_full.index >= EVAL_START]
formation_months    = pd.PeriodIndex([p - 1 for p in eval_return_periods], freq="M")

mu_fm = compute_fm_forecast(fm_estimates, delta_r, momentum, formation_months)

print("\nQ8e – Constructing CCV-Exp and CCV-FM portfolios …")
ccv_returns = compute_cmve_portfolio_returns(
    returns_wide=returns_wide_full,
    sigma_t=sigma_t,
    mu_exp=mu_exp,
    mu_fm=mu_fm,
    target_vol=0.10,
)

print("\nTable 4: Out-of-sample Portfolio Performance (Jan 1995 – Dec 2024)")
table4 = performance_table(ccv_returns, ["CCV-Exp", "CCV-FM"])
print(tabulate(table4, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

ccv_returns.to_csv(DATA_OUT / "question8_ccv_returns.csv")
fm_estimates.to_csv(DATA_OUT / "question8_fm_estimates.csv")
coef_table.to_csv(DATA_OUT / "question8_fm_coeff_table.csv")
table4.to_csv(DATA_OUT / "table4_question8.csv")

print(f"\nAll outputs saved to {DATA_OUT}/")
