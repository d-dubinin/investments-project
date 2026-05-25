# question7_portfolios.py

import numpy as np
import pandas as pd
from scipy import stats


def compute_portfolio_summary(portfolio_returns: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = {}

    for col in columns:
        x = portfolio_returns[col].dropna()

        mean_ann = x.mean() * 12 * 100
        t_stat = stats.ttest_1samp(x, 0).statistic
        std_ann = x.std() * np.sqrt(12) * 100
        sharpe = mean_ann / std_ann

        rows[col] = {
            "Mean (ann. %)": round(mean_ann, 2),
            "T-statistic": round(t_stat, 2),
            "Std (ann. %)": round(std_ann, 2),
            "Sharpe ratio": round(sharpe, 2),
        }

    return pd.DataFrame(rows)


def scale_to_annual_volatility(returns: pd.Series, target_vol: float = 0.10) -> pd.Series:
    ann_vol = returns.std() * np.sqrt(12)
    return returns * (target_vol / ann_vol)


def equal_weighted_portfolio(returns_wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    n = returns_wide.shape[1]
    weights = pd.Series(1 / n, index=returns_wide.columns)
    portfolio_returns = returns_wide @ weights

    return portfolio_returns, weights


def risk_parity_portfolio(returns_wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    vol = returns_wide.std()
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    portfolio_returns = returns_wide @ weights

    return portfolio_returns, weights


def mean_variance_portfolio(returns_wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mu = returns_wide.mean()
    sigma = returns_wide.cov()

    raw_weights = np.linalg.solve(sigma.to_numpy(), mu.to_numpy())
    weights = pd.Series(raw_weights, index=returns_wide.columns)
    weights = weights / weights.sum()

    portfolio_returns = returns_wide @ weights

    return portfolio_returns, weights


def prepare_currency_returns_wide(returns: pd.DataFrame, currencies: list[str]) -> pd.DataFrame:
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret["ym"] = ret["date"].dt.to_period("M")

    returns_wide = (
        ret.pivot(index="ym", columns="currency", values="X")
        .sort_index()[currencies]
    )

    return returns_wide.dropna()


def prepare_factor_returns_wide(
    carry_returns: pd.DataFrame,
    momentum_returns: pd.DataFrame,
    dollar_returns: pd.DataFrame,
) -> pd.DataFrame:
    factor_returns = (
        carry_returns
        .merge(momentum_returns, on="date", how="inner")
        .merge(dollar_returns, on="date", how="inner")
        .dropna()
    )

    factor_returns["date"] = pd.to_datetime(factor_returns["date"])
    factor_returns["ym"] = factor_returns["date"].dt.to_period("M")

    factor_columns = [
        "R_CS_CARRY",
        "R_TS_CARRY",
        "R_CS_MOM",
        "R_TS_MOM",
        "R_DOLLAR",
        "R_DOLLAR_CARRY",
    ]

    factor_returns = factor_returns.set_index("ym").sort_index()

    return factor_returns[factor_columns].dropna()


def compute_question7_portfolios(
    returns: pd.DataFrame,
    carry_returns: pd.DataFrame,
    momentum_returns: pd.DataFrame,
    dollar_returns: pd.DataFrame,
    currencies: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    currency_returns = prepare_currency_returns_wide(returns, currencies)
    factor_returns = prepare_factor_returns_wide(
        carry_returns=carry_returns,
        momentum_returns=momentum_returns,
        dollar_returns=dollar_returns,
    )

    common_idx = currency_returns.index.intersection(factor_returns.index)
    currency_returns = currency_returns.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]

    portfolios = {}
    weights = {}

    portfolios["EW-CCY"], weights["EW-CCY"] = equal_weighted_portfolio(currency_returns)
    portfolios["RP-CCY"], weights["RP-CCY"] = risk_parity_portfolio(currency_returns)
    portfolios["MV-CCY"], weights["MV-CCY"] = mean_variance_portfolio(currency_returns)

    portfolios["EW-FAC"], weights["EW-FAC"] = equal_weighted_portfolio(factor_returns)
    portfolios["RP-FAC"], weights["RP-FAC"] = risk_parity_portfolio(factor_returns)
    portfolios["MV-FAC"], weights["MV-FAC"] = mean_variance_portfolio(factor_returns)

    portfolio_returns = pd.DataFrame(portfolios)

    for col in portfolio_returns.columns:
        portfolio_returns[col] = scale_to_annual_volatility(portfolio_returns[col])

    portfolio_returns["date"] = portfolio_returns.index.to_timestamp("M")
    portfolio_returns = portfolio_returns.reset_index(drop=True)

    table3_columns = [
        "EW-CCY",
        "RP-CCY",
        "MV-CCY",
        "EW-FAC",
        "RP-FAC",
        "MV-FAC",
    ]

    table3 = compute_portfolio_summary(portfolio_returns, table3_columns)
    weights_table = pd.concat(weights, axis=1)

    return table3, portfolio_returns, weights_table