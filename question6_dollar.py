import numpy as np
import pandas as pd


def compute_dollar_strategies(
    returns: pd.DataFrame,
    rates_wide: pd.DataFrame,
    currencies: list[str],
) -> pd.DataFrame:
    """
    Compute DOLLAR and DOLLAR-CARRY strategy returns for Question 6(a).

    Parameters
    ----------
    returns : DataFrame
        Evaluation sample returns with columns: date | currency | X.

    rates_wide : DataFrame
        Interest-rate data with columns: date | AUD | CAD | EUR | GBP | JPY | NZD | USD.
        The rates should already be monthly decimals, as in your saved ir_monthly_wide.csv.

    currencies : list[str]
        Currency list, e.g. ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD"].

    Returns
    -------
    dollar_returns : DataFrame
        Columns: date | R_DOLLAR | R_DOLLAR_CARRY.
    """

    N = len(currencies)

    # Convert returns to wide format: ym x currency
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret["ym"] = ret["date"].dt.to_period("M")

    ret_wide = (
        ret.pivot(index="ym", columns="currency", values="X")
           .sort_index()[currencies]
    )

    # DOLLAR: equal-weighted basket of all foreign currencies
    R_dollar = ret_wide.mean(axis=1)

    # Interest-rate differentials for DOLLAR-CARRY
    ir = rates_wide.copy()
    ir["date"] = pd.to_datetime(ir["date"])
    ir["ym"] = ir["date"].dt.to_period("M")
    ir = ir.set_index("ym").sort_index()

    # Average interest-rate differential vs USD, computed on the full sample.
    delta_r = ir[currencies].subtract(ir["USD"], axis=0)
    avg_delta_r = delta_r.mean(axis=1)

    # Use the differential known at formation time (end of month t-1) for the
    # return realized over month t. Shift by one month before aligning to the
    # returns index, exactly like the carry construction, to avoid a one-month
    # look-ahead in the sign of the position.
    avg_delta_r = avg_delta_r.shift(1).reindex(ret_wide.index)

    # DOLLAR-CARRY:
    # long the equal-weighted foreign-currency basket if avg rate differential is positive,
    # short the basket if avg rate differential is negative.
    R_dollar_carry = np.sign(avg_delta_r) * R_dollar

    dollar_returns = pd.DataFrame({
        "R_DOLLAR": R_dollar,
        "R_DOLLAR_CARRY": R_dollar_carry,
    }).reset_index()

    dollar_returns["date"] = dollar_returns["ym"].dt.to_timestamp("M")
    dollar_returns = dollar_returns[
        ["date", "R_DOLLAR", "R_DOLLAR_CARRY"]
    ].dropna().reset_index(drop=True)

    print(f"\nDollar shape : {dollar_returns.shape}  (expected 360 rows)")
    print(
        f"Date range   : "
        f"{dollar_returns['date'].min().date()} → "
        f"{dollar_returns['date'].max().date()}"
    )

    print("\n=== Dollar strategies sample ===")
    print(dollar_returns.head(12).to_string(index=False))

    return dollar_returns

def regress_carry_on_momentum_and_dollar(
    carry_returns: pd.DataFrame,
    momentum_returns: pd.DataFrame,
    dollar_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Question 6(b).

    Regress TS-CARRY on CS-MOM, TS-MOM, and DOLLAR:

        R_TS_CARRY = alpha
                     + beta_1 R_CS_MOM
                     + beta_2 R_TS_MOM
                     + beta_3 R_DOLLAR
                     + error

    Returns
    -------
    reg_table : DataFrame
        Coefficients and t-statistics.

    r2 : float
        Regression R^2.
    """

    df = (
        carry_returns
        .merge(momentum_returns, on="date", how="inner")
        .merge(dollar_returns, on="date", how="inner")
        .dropna()
    )

    y = df["R_TS_CARRY"].to_numpy()

    X_raw = df[["R_CS_MOM", "R_TS_MOM", "R_DOLLAR"]].to_numpy()

    # Add intercept
    X = np.column_stack([np.ones(len(df)), X_raw])

    names = [
        "Intercept",
        "R_CS_MOM",
        "R_TS_MOM",
        "R_DOLLAR",
    ]

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - X @ beta

    n, k = X.shape
    sigma2 = residuals @ residuals / (n - k)

    vcov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(vcov))
    t_stats = beta / se

    r2 = 1 - (residuals @ residuals) / ((y - y.mean()) @ (y - y.mean()))

    reg_table = pd.DataFrame({
        "Coefficient": beta,
        "t-statistic": t_stats,
    }, index=names)

    return reg_table.round(4), round(float(r2), 4)