import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from tabulate import tabulate

from core.excess_return import ExcessReturnCalculator

DATA_RAW = ROOT / "data" / "raw"

fx_panel   = pd.read_csv(DATA_RAW / "fx_monthly_panel.csv", parse_dates=["date"])
rates_wide = pd.read_csv(DATA_RAW / "ir_monthly_wide.csv",  parse_dates=["date"])

calc = ExcessReturnCalculator()

returns = calc.compute(fx_panel, rates_wide)
holdout, evaluation = calc.split(returns)
table1 = calc.compute_summary(evaluation, rates_wide)

print("\nTable 1: Monthly Summary Statistics")
print(tabulate(table1, headers='keys', tablefmt='rounded_outline', floatfmt='.4f'))

rets, weights = calc.compute_carry(evaluation, rates_wide)

table2 = calc.compute_carry_summary(rets)

print("\nTable 2: Carry Summary Statistics")
print(tabulate(table2, headers='keys', tablefmt='rounded_outline', floatfmt='.4f'))

calc.plot_carry_weights(weights)

corr = calc.compute_forward_discount_correlation(fx_panel, rates_wide)

print("\nCross-sectional correlations")
print(tabulate(corr, headers='keys', tablefmt='rounded_outline', floatfmt='.4f'))
