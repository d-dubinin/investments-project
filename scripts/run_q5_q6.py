import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

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

DATA_RAW = ROOT / "data" / "raw"

fx_panel   = pd.read_csv(DATA_RAW / "fx_monthly_panel.csv", parse_dates=["date"])
rates_wide = pd.read_csv(DATA_RAW / "ir_monthly_wide.csv",  parse_dates=["date"])

calc = ExcessReturnCalculator()

returns = calc.compute(fx_panel, rates_wide)
holdout, evaluation = calc.split(returns)

carry_returns, carry_weights = calc.compute_carry(evaluation, rates_wide)

momentum_returns, momentum_weights = compute_momentum_strategies(
    fx_panel=fx_panel,
    returns=evaluation,
    currencies=calc.CURRENCIES,
)

factor_returns = carry_returns.merge(momentum_returns, on="date", how="inner")

table2 = compute_strategy_summary(
    factor_returns,
    columns=[
        "R_CS_CARRY",
        "R_TS_CARRY",
        "R_CS_MOM",
        "R_TS_MOM",
    ],
)

print("\nTable 2: Carry and Momentum Strategy Summary Statistics")
print(tabulate(table2, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

reg_table, r2 = regress_ts_carry_on_momentum(
    carry_returns=carry_returns,
    momentum_returns=momentum_returns,
)

print("\nQuestion 5(b): Regression of TS-CARRY on CS-MOM and TS-MOM")
print(tabulate(reg_table, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
print(f"\nR^2 = {r2:.4f}")


# Question 6(a): Dollar strategies
dollar_returns = compute_dollar_strategies(
    returns=evaluation,
    rates_wide=rates_wide,
    currencies=calc.CURRENCIES,
)

factor_returns = (
    carry_returns
    .merge(momentum_returns, on="date", how="inner")
    .merge(dollar_returns, on="date", how="inner")
)

table2_full = compute_strategy_summary(
    factor_returns,
    columns=[
        "R_CS_CARRY",
        "R_TS_CARRY",
        "R_CS_MOM",
        "R_TS_MOM",
        "R_DOLLAR",
        "R_DOLLAR_CARRY",
    ],
)

print("\nTable 2: Carry, Momentum, and Dollar Strategy Summary Statistics")
print(tabulate(table2_full, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

reg_table_6b, r2_6b = regress_carry_on_momentum_and_dollar(
    carry_returns=carry_returns,
    momentum_returns=momentum_returns,
    dollar_returns=dollar_returns,
)

print("\nQuestion 6(b): Regression of TS-CARRY on CS-MOM, TS-MOM, and DOLLAR")
print(tabulate(reg_table_6b, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
print(f"\nR^2 = {r2_6b:.4f}")
