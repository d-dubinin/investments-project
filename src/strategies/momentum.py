import numpy as np
import pandas as pd
from scipy import stats


def compute_momentum_strategies(
    fx_panel: pd.DataFrame,
    returns: pd.DataFrame,
    currencies: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute CS-MOM and TS-MOM strategy returns for Question 5.

    Parameters
    ----------
    fx_panel : DataFrame
        Monthly FX panel with columns: date | currency | spot | fwd1m

    returns : DataFrame
        Evaluation sample excess returns with columns: date | currency | X

    currencies : list[str]
        List of currency tickers, e.g. ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD"]

    Returns
    -------
    momentum_returns : DataFrame
        Columns: date | R_CS_MOM | R_TS_MOM

    momentum_weights : DataFrame
        CS-MOM weights over time.
    """

    N = len(currencies)

    # Spot prices in wide format: ym x currency
    fx = fx_panel.copy()
    fx["ym"] = fx["date"].dt.to_period("M")

    spot_wide = (
        fx.pivot(index="ym", columns="currency", values="spot")
          .sort_index()[currencies]
    )

    # Momentum signal: s_{t-12,t} = S_t / S_{t-12} - 1
    signal = spot_wide / spot_wide.shift(12) - 1

    # lag by one month so the signal is known at the formation date
    signal = signal.shift(1)

    # Returns in wide format: ym x currency
    ret = returns.copy()
    ret["ym"] = ret["date"].dt.to_period("M")

    ret_wide = (
        ret.pivot(index="ym", columns="currency", values="X")
           .sort_index()[currencies]
    )

    # Align signals and returns on the monthly index.
    common_idx = ret_wide.index.intersection(signal.dropna().index)

    ret_wide = ret_wide.loc[common_idx]
    signal = signal.loc[common_idx]

    # ------------------------------------------------------------
    # CS-MOM
    # Rank by past 12-month appreciation.
    # Rank 1 = lowest past appreciation, Rank N = highest.
    # ------------------------------------------------------------
    ranks = signal.rank(axis=1, method="average")

    raw_w = ranks - (N + 1) / 2
    long_sum = raw_w.where(raw_w > 0).sum(axis=1)

    w_cs = raw_w.div(long_sum, axis=0)

    R_cs_mom = (w_cs * ret_wide).sum(axis=1)

    # ------------------------------------------------------------
    # TS-MOM
    # Long currencies with positive past appreciation,
    # short currencies with negative past appreciation.
    # ------------------------------------------------------------
    w_ts = np.sign(signal) / N

    R_ts_mom = (w_ts * ret_wide).sum(axis=1)

    momentum_returns = pd.DataFrame({
        "R_CS_MOM": R_cs_mom,
        "R_TS_MOM": R_ts_mom,
    }).reset_index()

    momentum_returns["date"] = momentum_returns["ym"].dt.to_timestamp("M")
    momentum_returns = momentum_returns[["date", "R_CS_MOM", "R_TS_MOM"]].dropna()

    momentum_weights = w_cs.copy()
    momentum_weights["date"] = momentum_weights.index.to_timestamp("M")
    momentum_weights = momentum_weights.reset_index(drop=True)

    print(f"\nMomentum shape : {momentum_returns.shape}  (expected 360 rows)")
    print(
        f"Date range     : "
        f"{momentum_returns['date'].min().date()} -> "
        f"{momentum_returns['date'].max().date()}"
    )

    return momentum_returns, momentum_weights


def compute_strategy_summary(
    factor_returns: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Compute annualized mean, t-statistic, annualized volatility,
    and annualized Sharpe ratio for selected strategy return columns.
    """

    rows = {}

    for col in columns:
        x = factor_returns[col].dropna()

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


def regress_ts_carry_on_momentum(
    carry_returns: pd.DataFrame,
    momentum_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Regression for Question 5(b):

        R_TS_CARRY = alpha + beta_1 R_CS_MOM + beta_2 R_TS_MOM + error
    """

    df = carry_returns.merge(momentum_returns, on="date", how="inner").dropna()

    y = df["R_TS_CARRY"].to_numpy()
    X_raw = df[["R_CS_MOM", "R_TS_MOM"]].to_numpy()

    # Add intercept
    X = np.column_stack([np.ones(len(df)), X_raw])
    names = ["Intercept", "R_CS_MOM", "R_TS_MOM"]

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - X @ beta
    n, k = X.shape

    sigma2 = residuals @ residuals / (n - k)
    vcov = sigma2 * np.linalg.inv(X.T @ X)

    se = np.sqrt(np.diag(vcov))
    t_stats = beta / se

    r2 = 1 - (residuals @ residuals) / ((y - y.mean()) @ (y - y.mean()))

    table = pd.DataFrame({
        "Coefficient": beta,
        "t-statistic": t_stats,
    }, index=names)

    return table.round(4), round(float(r2), 4)
