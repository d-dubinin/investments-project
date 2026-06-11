import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from core.excess_return import ExcessReturnCalculator
from strategies.momentum import compute_momentum_strategies
from strategies.dollar import compute_dollar_strategies
from portfolios.construction import compute_question7_portfolios

DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "output"
DATA_OUT.mkdir(parents=True, exist_ok=True)

currencies = ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD"]

fx_panel   = pd.read_csv(DATA_RAW / "fx_monthly_panel.csv", parse_dates=["date"])
rates_wide = pd.read_csv(DATA_RAW / "ir_monthly_wide.csv",  parse_dates=["date"])

calc = ExcessReturnCalculator()

returns_full = calc.compute(fx_panel, rates_wide)
holdout, evaluation = calc.split(returns_full)

carry_returns, carry_weights = calc.compute_carry(evaluation, rates_wide)

momentum_returns, momentum_weights = compute_momentum_strategies(
    fx_panel=fx_panel,
    returns=evaluation,
    currencies=currencies,
)

dollar_returns = compute_dollar_strategies(
    returns=evaluation,
    rates_wide=rates_wide,
    currencies=currencies,
)

table3, question7_returns, question7_weights = compute_question7_portfolios(
    returns=evaluation,
    carry_returns=carry_returns,
    momentum_returns=momentum_returns,
    dollar_returns=dollar_returns,
    currencies=currencies,
)

print("\n=== Table 3: In-sample Portfolio Analysis ===")
print(table3.to_string())

print("\n=== Question 7 Portfolio Weights ===")
print(question7_weights.round(4).to_string())

question7_returns.to_csv(DATA_OUT / "question7_portfolio_returns.csv", index=False)
table3.to_csv(DATA_OUT / "table3_question7.csv")
question7_weights.to_csv(DATA_OUT / "question7_weights.csv")
