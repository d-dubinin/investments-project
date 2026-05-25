import pandas as pd

from excess_return import ExcessReturnCalculator
from question5_momentum import compute_momentum_strategies
from question6_dollar import compute_dollar_strategies
from question7 import compute_question7_portfolios


currencies = ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD"]

fx_panel = pd.read_csv("data/fx_monthly_panel.csv", parse_dates=["date"])
rates_wide = pd.read_csv("data/ir_monthly_wide.csv", parse_dates=["date"])

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

# Save Q7 outputs for reproducibility and for easy inclusion in the report.
question7_returns.to_csv("question7_portfolio_returns.csv", index=False)
table3.to_csv("table3_question7.csv")
question7_weights.to_csv("question7_weights.csv")