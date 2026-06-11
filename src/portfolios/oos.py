"""
Out-of-sample Portfolio Analysis (Question 8)
=============================================
Implements:
  - EWMA conditional covariance  (Q8a)
  - Expanding-mean conditional mean forecast  (Q8b)
  - Fama-MacBeth conditional mean forecast     (Q8c / Q8d)
  - CCV-Exp and CCV-FM CMVE portfolios         (Q8e)
  - Table 4 performance statistics             (Q8f)

All functions operate on PeriodIndex (monthly) DataFrames.

"""

import numpy as np
import pandas as pd
from scipy import stats

# ── Constants ────────────────────────────────────────────────────────────────
CURRENCIES  = ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD"]
HOLDOUT_END = pd.Period("1994-12", "M")
EVAL_START  = pd.Period("1995-01", "M")
LAM         = 0.97
TARGET_VOL  = 0.10   # annualized


# ── Helpers ──────────────────────────────────────────────────────────────────

def to_period_wide(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert long returns (date | currency | X) to wide PeriodIndex DataFrame."""
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret["ym"]   = ret["date"].dt.to_period("M")
    wide = (
        ret.pivot(index="ym", columns="currency", values="X")
           .sort_index()[CURRENCIES]
    )
    wide.index.name = "ym"
    return wide


def to_period_spot(fx_panel: pd.DataFrame) -> pd.DataFrame:
    """Convert FX panel to wide spot PeriodIndex DataFrame."""
    fx = fx_panel.copy()
    fx["date"] = pd.to_datetime(fx["date"])
    fx["ym"]   = fx["date"].dt.to_period("M")
    wide = (
        fx.pivot(index="ym", columns="currency", values="spot")
          .sort_index()[CURRENCIES]
    )
    wide.index.name = "ym"
    return wide


# ── Q8a: EWMA Conditional Covariance ─────────────────────────────────────────

def compute_ewma_covariance(
    returns_wide: pd.DataFrame,
    lam: float = LAM,
) -> dict:
    """
    EWMA covariance: Σ̂_t = (1 - λ) X_t X_t^T + λ Σ̂_{t-1}

    Σ̂_0 is initialized with the sample covariance over the holdout period
    (all months in returns_wide up to HOLDOUT_END inclusive).

    For a portfolio formed at the end of month t, use Σ̂_t (updated with X_t).

    Parameters
    ----------
    returns_wide : PeriodIndex DataFrame, columns = CURRENCIES (monthly decimals)
    lam          : decay parameter (default 0.97)

    Returns
    -------
    dict  period -> np.ndarray (6 × 6)
    """
    holdout_mask = returns_wide.index <= HOLDOUT_END
    holdout_data = returns_wide.loc[holdout_mask].dropna()
    sigma = holdout_data.cov().to_numpy()   # Σ̂_0

    sigma_t = {}
    for period in returns_wide.index:
        row = returns_wide.loc[period]
        x   = row.to_numpy(dtype=float)
        if not np.any(np.isnan(x)):
            sigma = (1.0 - lam) * np.outer(x, x) + lam * sigma
        sigma_t[period] = sigma.copy()

    return sigma_t


# ── Q8b: Expanding-mean Conditional Mean ─────────────────────────────────────

def compute_expanding_mean(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """
    μ̂_{it} = (1 / N_t) Σ_{s=1}^{t} X_s^i
    where N_t counts all non-missing months up to t, including holdout.

    Returns a PeriodIndex DataFrame with the same index/columns as returns_wide.
    """
    return returns_wide.expanding(min_periods=1).mean()


# ── Q8c: Fama-MacBeth helpers ────────────────────────────────────────────────

def compute_delta_r(rates_wide: pd.DataFrame, monthly_decimal: bool = False) -> pd.DataFrame:
    """
    Δr^i_t = r^i_t - r^US_t

    Note: ir_monthly_wide.csv stores rates as monthly decimals (e.g. 0.010 for ~12%/yr).
    With monthly_decimal=False (default), no extra division is applied, so delta_r is
    also in monthly decimal units (e.g. 0.004 for a typical G10 differential).
    With monthly_decimal=True, rates are divided by 1200 (only useful if the CSV is
    ever regenerated with annual % rates).

    Returns a PeriodIndex DataFrame indexed by signal month t.
    """
    ir = rates_wide.copy()
    ir["date"] = pd.to_datetime(ir["date"])
    ir["ym"]   = ir["date"].dt.to_period("M")
    ir = ir.set_index("ym")[CURRENCIES + ["USD"]]
    if monthly_decimal:
        ir = ir / 1200                                         # % p.a. -> monthly decimal
    delta_r = ir[CURRENCIES].subtract(ir["USD"], axis=0)
    delta_r.index.name = "ym"
    return delta_r


def compute_momentum_signal(fx_panel: pd.DataFrame) -> pd.DataFrame:
    """
    s^i_{t-12,t} = S^i_t / S^i_{t-12} − 1   (12-month spot appreciation).

    First valid period: one year after the first available spot.
    Returns a PeriodIndex DataFrame indexed by signal month t.
    """
    spot = to_period_spot(fx_panel)
    mom  = spot / spot.shift(12) - 1
    mom.index.name = "ym"
    return mom


def run_fm_regressions(
    returns_wide: pd.DataFrame,
    delta_r:      pd.DataFrame,
    momentum:     pd.DataFrame,
) -> pd.DataFrame:
    """
    At each signal month t, run the cross-sectional regression (no intercept):

        X^i_{t+1} = γ_t Δr^i_t + φ_t s^i_{t-12,t} + ε^i_{t+1}

    across the N = 6 currencies.  Stores (γ̂_t, φ̂_t) for every valid t.

    The holdout period is also used; the problem says "starting the first month
    for which you have both signals and returns available."

    Parameters
    ----------
    returns_wide : full-sample wide returns (PeriodIndex, labeled with receipt month)
    delta_r      : interest-rate differentials (PeriodIndex = signal month t)
    momentum     : 12-month momentum signal (PeriodIndex = signal month t)

    Returns
    -------
    DataFrame  indexed by signal period t, columns = [gamma, phi]
    """
    records = []

    for t_period in delta_r.index:
        t1_period = t_period + 1

        # Check data availability ─────────────────────────────────────────────
        if t1_period not in returns_wide.index:
            continue
        if t_period not in momentum.index:
            continue

        dr   = delta_r.loc[t_period].to_numpy(dtype=float)
        s    = momentum.loc[t_period].to_numpy(dtype=float)
        Xnxt = returns_wide.loc[t1_period].to_numpy(dtype=float)

        if np.any(np.isnan(dr)) or np.any(np.isnan(s)) or np.any(np.isnan(Xnxt)):
            continue

        # OLS without intercept: Xnxt = [dr | s] @ beta ──────────────────────
        X_reg = np.column_stack([dr, s])
        beta, _, _, _ = np.linalg.lstsq(X_reg, Xnxt, rcond=None)

        records.append({"ym": t_period, "gamma": beta[0], "phi": beta[1]})

    df = pd.DataFrame(records).set_index("ym")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def compute_fm_forecast(
    fm_estimates: pd.DataFrame,
    delta_r:      pd.DataFrame,
    momentum:     pd.DataFrame,
    formation_months: pd.PeriodIndex,
) -> pd.DataFrame:
    """
    For each portfolio-formation month T, compute the FM mean forecast:

        μ̂^FM_{iT} = γ̄_T · Δr^i_T + φ̄_T · s^i_{T-12,T}

    where γ̄_T and φ̄_T are the expanding averages of all estimated FM
    coefficients available *strictly before* T+1 (i.e., from regressions
    whose outcome X^i_{t+1} was observed by the end of month T):

        γ̄_T = mean( γ̂_t  for  t ≤ T−1 )

    This guarantees no look-ahead: the regression at t predicts X_{t+1},
    so γ̂_t is first available at the end of month t+1.

    Parameters
    ----------
    fm_estimates     : output of run_fm_regressions()
    delta_r          : interest-rate differentials indexed by signal month
    momentum         : 12-month momentum signal indexed by signal month
    formation_months : PeriodIndex of months at which portfolios are formed

    Returns
    -------
    DataFrame  indexed by formation_months, columns = CURRENCIES (monthly decimal forecasts)
    """
    # Pre-compute expanding averages of FM estimates (indexed by signal month t).
    # fm_expanding.loc[t] = average of all γ̂_s for s ≤ t.
    fm_expanding = fm_estimates.expanding(min_periods=1).mean()

    rows = {}
    for T in formation_months:
        T_minus_1 = T - 1   # last signal month whose regression is available at end of T

        if T_minus_1 not in fm_expanding.index:
            rows[T] = np.full(len(CURRENCIES), np.nan)
            continue

        gamma_bar = fm_expanding.loc[T_minus_1, "gamma"]
        phi_bar   = fm_expanding.loc[T_minus_1, "phi"]

        if T not in delta_r.index or T not in momentum.index:
            rows[T] = np.full(len(CURRENCIES), np.nan)
            continue

        dr_T = delta_r.loc[T].to_numpy(dtype=float)
        s_T  = momentum.loc[T].to_numpy(dtype=float)

        if np.any(np.isnan(dr_T)) or np.any(np.isnan(s_T)):
            rows[T] = np.full(len(CURRENCIES), np.nan)
            continue

        rows[T] = gamma_bar * dr_T + phi_bar * s_T

    fm_df = pd.DataFrame.from_dict(rows, orient="index", columns=CURRENCIES)
    fm_df.index = pd.PeriodIndex(fm_df.index, freq="M")
    fm_df.index.name = "ym"
    return fm_df


# ── Q8e: CMVE Portfolio Construction ─────────────────────────────────────────

def _cmve_one_period(
    mu:     np.ndarray,
    Sigma:  np.ndarray,
    X_next: np.ndarray,
    target_vol: float,
) -> float:
    """
    Compute one CMVE portfolio return.

    w = k · Σ^{-1} μ,   k chosen so annualised vol = target_vol.
    Annualised portfolio variance:  12 · w' Σ w = 12 k² μ' Σ^{-1} μ  = target_vol²
    ⟹  k = target_vol / √(12 · μ' Σ^{-1} μ)
    """
    if np.any(np.isnan(mu)) or np.any(np.isnan(X_next)):
        return np.nan

    try:
        Sigma_inv_mu = np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        return np.nan

    port_var_monthly = float(mu @ Sigma_inv_mu)
    if port_var_monthly <= 0:
        return np.nan

    k = target_vol / np.sqrt(12.0 * port_var_monthly)
    return float(k * (Sigma_inv_mu @ X_next))


def compute_cmve_portfolio_returns(
    returns_wide:    pd.DataFrame,
    sigma_t:         dict,
    mu_exp:          pd.DataFrame,
    mu_fm:           pd.DataFrame,
    target_vol:      float = TARGET_VOL,
) -> pd.DataFrame:
    """
    Construct CCV-Exp and CCV-FM portfolio returns over the evaluation sample.

    For each formation month T (returns labeled T+1):
      w_T = k_T · Σ̂_T^{-1} · μ̂_T
      R_{T+1} = w_T' · X_{T+1}

    The EWMA covariance Σ̂_T is the same for both portfolios (from Q8a).
    The mean vector differs:
      CCV-Exp : μ̂_T from the expanding sample mean (Q8b)
      CCV-FM  : μ̂_T from the Fama-MacBeth forecast (Q8c)

    Parameters
    ----------
    returns_wide : full-sample returns (PeriodIndex)
    sigma_t      : dict period -> 6×6 np.ndarray (monthly covariance)
    mu_exp       : expanding mean DataFrame (PeriodIndex, CURRENCIES)
    mu_fm        : FM forecast DataFrame (PeriodIndex, CURRENCIES)
    target_vol   : annualised target volatility (default 0.10)

    Returns
    -------
    DataFrame  columns = [CCV-Exp, CCV-FM],  index = date (month-end Timestamp)
    """
    eval_return_periods = returns_wide.index[returns_wide.index >= EVAL_START]

    records = []
    for T1 in eval_return_periods:           # T1 = receipt month = T + 1
        T = T1 - 1                           # formation month

        if T not in sigma_t:
            continue
        if T1 not in returns_wide.index:
            continue

        X_next = returns_wide.loc[T1].to_numpy(dtype=float)
        if np.any(np.isnan(X_next)):
            continue

        Sigma = sigma_t[T]

        # Expanding mean
        mu_e = mu_exp.loc[T].to_numpy(dtype=float) if T in mu_exp.index else np.full(6, np.nan)
        r_exp = _cmve_one_period(mu_e, Sigma, X_next, target_vol)

        # FM forecast
        mu_f = mu_fm.loc[T].to_numpy(dtype=float) if T in mu_fm.index else np.full(6, np.nan)
        r_fm = _cmve_one_period(mu_f, Sigma, X_next, target_vol)

        records.append({
            "date":    T1.to_timestamp("M"),
            "CCV-Exp": r_exp,
            "CCV-FM":  r_fm,
        })

    df = pd.DataFrame(records).set_index("date")
    return df


# ── Q8d/f: Summary Statistics ─────────────────────────────────────────────────

def performance_table(portfolio_returns: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Annualised mean (%), t-statistic, annualised std (%), Sharpe ratio.
    """
    rows = {}
    for col in columns:
        x = portfolio_returns[col].dropna()
        mean_ann = x.mean() * 12 * 100
        t_stat   = stats.ttest_1samp(x, 0).statistic
        std_ann  = x.std() * np.sqrt(12) * 100
        sharpe   = mean_ann / std_ann
        rows[col] = {
            "Mean (ann. %)": round(mean_ann, 2),
            "T-statistic":   round(t_stat,   2),
            "Std (ann. %)":  round(std_ann,  2),
            "Sharpe ratio":  round(sharpe,   2),
        }
    return pd.DataFrame(rows)


def fm_coefficient_table(
    fm_estimates: pd.DataFrame,
    eval_start:   pd.Period = EVAL_START,
) -> pd.DataFrame:
    """
    Q8d: Time-series average of γ̂_t and φ̂_t over the evaluation sample,
    with Fama-MacBeth t-statistics (mean / (std / √T)).

    Note: the FM regression at signal month t uses outcome X_{t+1},
    so the last valid signal month for return December 2024 is November 2024.
    We report estimates whose signal month t falls within the evaluation sample
    (January 1995 – November 2024).
    """
    eval_end_signal = pd.Period("2024-11", "M")   # last regression whose outcome is in eval
    eval_mask = (fm_estimates.index >= eval_start) & (fm_estimates.index <= eval_end_signal)
    eval_est  = fm_estimates.loc[eval_mask]

    rows = {}
    for col, label in [("gamma", "γ (carry)"), ("phi", "φ (momentum)")]:
        x    = eval_est[col].dropna()
        mean = x.mean()
        std  = x.std()
        T    = len(x)
        tstat = mean / (std / np.sqrt(T))
        rows[label] = {
            "Mean": round(mean,  6),
            "Std":  round(std,   6),
            "N":    T,
            "T-statistic": round(tstat, 4),
        }
    return pd.DataFrame(rows)
