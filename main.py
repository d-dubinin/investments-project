from fx_data_fetcher import FXDataFetcher
from interest_rate_fetcher import InterestRateFetcher
from excess_return import ExcessReturnCalculator

from tabulate import tabulate

WRDS_USERNAME = 'd_dubinin'

# FX
fx = FXDataFetcher(wrds_username=WRDS_USERNAME)
fx_panel = fx.fetch()
fx.save(fx_panel)

# Interest rates
ir = InterestRateFetcher()
rates_wide, rates_long = ir.fetch()
ir.save(rates_wide, rates_long)

# Excess returns
calc = ExcessReturnCalculator()
# Returns
returns = calc.compute(fx_panel, rates_wide)
# Split into holdout/evaluation
holdout, evaluation = calc.split(returns)
table1 = calc.compute_summary(evaluation, rates_wide)

# Table 1
print("\nTable 1: Monthly Summary Statistics")
print(tabulate(table1, headers='keys', tablefmt='rounded_outline', floatfmt='.4f'))

# Carry strategies

# Returns and Weights for two carry strategies
rets, weights = calc.compute_carry(evaluation, rates_wide)

# Table 2
table2 = calc.compute_carry_summary(rets)

print("\nTable 2: Carry Summary Statistics")
print(tabulate(table2, headers='keys', tablefmt='rounded_outline', floatfmt='.4f'))

# Plot of weights evolution
calc.plot_carry_weights(weights)

# Cross-sectional correlations
corr = calc.compute_forward_discount_correlation(fx_panel, rates_wide)

print("\nCross-sectional correlations")
print(tabulate(corr, headers='keys', tablefmt='rounded_outline', floatfmt='.4f'))