import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class ExcessReturnCalculator:

    FULL_START  = "1991-01-01"
    FULL_END    = "2024-12-31"
    HOLDOUT_END = "1994-12-31"
    EVAL_START  = "1995-01-01"
    CURRENCIES  = ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD"]

    def compute(self, fx_panel: pd.DataFrame, rates_wide: pd.DataFrame) -> pd.DataFrame:
        """
        Compute excess returns X_{t+1} = (S_{t+1}/S_t) * (1 + r_t^i) - (1 + r_t^US)
        """
        fx = fx_panel.copy()
        ir = rates_wide.copy()

        # Align on year-month period
        fx["ym"] = fx["date"].dt.to_period("M")
        ir["ym"] = ir["date"].dt.to_period("M")
        ir[self.CURRENCIES + ["USD"]] = ir[self.CURRENCIES + ["USD"]] / 1200

        # Merge interest rates into fx panel
        fx = fx.merge(ir[["ym"] + self.CURRENCIES + ["USD"]], on="ym", how="left")

        # Pick correct foreign rate per row
        fx["r_foreign"] = fx.apply(lambda row: row[row["currency"]], axis=1)
        fx["r_usd"]     = fx["USD"]

        # S_{t+1}: next month's spot
        fx = fx.sort_values(["currency", "ym"])
        fx["S_next"] = fx.groupby("currency")["spot"].shift(-1)

        # X_{t+1}
        fx["X"] = (fx["S_next"] / fx["spot"]) * (1 + fx["r_foreign"]) - (1 + fx["r_usd"])

        # Clean up
        result = fx.dropna(subset=["X"]).copy()
        result = result[["date", "currency", "X"]]
        result = result[
            (result["date"] >= self.FULL_START) &
            (result["date"] <= self.FULL_END)
        ]
        result = result.sort_values(["date", "currency"]).reset_index(drop=True)

        return result

    def split(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into holdout (Jan1991-Dec1994) and evaluation (Jan1995-Dec2024)."""
        holdout    = returns[returns["date"] <= self.HOLDOUT_END].reset_index(drop=True)
        evaluation = returns[returns["date"] >= self.EVAL_START].reset_index(drop=True)
        return holdout, evaluation

    def compute_summary(self, returns: pd.DataFrame, rates_wide: pd.DataFrame) -> pd.DataFrame:
        """
        Build Table 1: summary statistics for each currency.
        Uses the evaluation sample passed in.
        """
        ir = rates_wide.copy()
        ir["ym"] = ir["date"].dt.to_period("M")

        rows = {}
        for ccy in self.CURRENCIES:
            x = returns[returns["currency"] == ccy]["X"].dropna()

            mean_ann = x.mean() * 12 * 100
            t_stat   = stats.ttest_1samp(x, 0).statistic
            std_ann  = x.std() * (12 ** 0.5) * 100
            sharpe   = mean_ann / std_ann
            skew     = stats.skew(x)
            kurt     = stats.kurtosis(x)
            n        = len(x)
            r_bar    = ir[ccy].dropna().mean() * 1200

            rows[ccy] = {
                "Avg excess return" : round(mean_ann, 2),
                "T-statistic"       : round(t_stat,   2),
                "Std dev"           : round(std_ann,  2),
                "Sharpe ratio"      : round(sharpe,   2),
                "Skewness"          : round(skew,     2),
                "Excess kurtosis"   : round(kurt,     2),
                "N"                 : n,
                "Avg foreign rate"  : round(r_bar,    2),
            }

        return pd.DataFrame(rows)[self.CURRENCIES]
    
    def compute_carry(self, returns: pd.DataFrame, rates_wide: pd.DataFrame) -> pd.DataFrame:
        """
        Compute CS-CARRY and TS-CARRY strategy returns.
    
        arameters
        ----------
        returns    : date | currency | X   (evaluation sample, monthly decimals)
        rates_wide : date | AUD | ... | USD (monthly decimals)
    
        Returns
        -------
        pd.DataFrame : date | R_CS_CARRY | R_TS_CARRY
        """
        N = len(self.CURRENCIES)

        ir = rates_wide.copy()
        ir["ym"] = ir["date"].dt.to_period("M")

        # Pivot returns to wide: date × currency
        ret_wide = returns.copy()
        ret_wide["ym"] = ret_wide["date"].dt.to_period("M")
        ret_wide = ret_wide.pivot(index="ym", columns="currency", values="X")[self.CURRENCIES]

        # Merge interest rates aligned on ym
        ir = ir.set_index("ym")[self.CURRENCIES + ["USD"]]
        ir = ir.loc[ir.index.isin(ret_wide.index)] 

        # Compute interest rate differentials: dt r_t^i = r_t^i - r_t^US
        delta_r = ir[self.CURRENCIES].subtract(ir["USD"], axis=0)

        # ── CS-CARRY ─────────────────────────────────────────────────────────────
        # Rank each month (1=lowest, 6=highest)
        ranks = delta_r.rank(axis=1, method="average")

        # Weights: w = Z * (rank - (N+1)/2)
        raw_w    = ranks - (N + 1) / 2
        long_sum = raw_w[raw_w > 0].sum(axis=1)          # sum of positive weights per month
        Z        = 1.0 / long_sum                         # normalise so long side sums to 1
        w_cs     = raw_w.multiply(Z, axis=0)

        # Strategy return
        R_cs = (w_cs * ret_wide).sum(axis=1)

        # ── TS-CARRY ─────────────────────────────────────────────────────────────
        w_ts  = delta_r.apply(np.sign) / N
        R_ts  = (w_ts * ret_wide).sum(axis=1)

        # ── Combine ──────────────────────────────────────────────────────────────
        result = pd.DataFrame({
            "R_CS_CARRY": R_cs,
            "R_TS_CARRY": R_ts,
        }).reset_index()
        result["date"] = result["ym"].dt.to_timestamp("M")
        result = result[["date", "R_CS_CARRY", "R_TS_CARRY"]].dropna()

        weights = w_cs.copy()
        weights["date"] = weights.index.to_timestamp("M")
        weights = weights.reset_index(drop=True)

        self._print_carry_summary(result)
        return result, weights
    
    def plot_carry_weights(self, weights: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        for ccy in self.CURRENCIES:
            ax.plot(weights["date"], weights[ccy], label=ccy)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("CS-CARRY Portfolio Weights Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.legend()
        plt.tight_layout()
        plt.savefig("data/cs_carry_weights.png", dpi=150)
        plt.show()

    def compute_carry_summary(self, carry: pd.DataFrame) -> pd.DataFrame:
        """
        Build Table 2: summary statistics for CS-CARRY and TS-CARRY.
        """
        rows = {}
        for col in ["R_CS_CARRY", "R_TS_CARRY"]:
            x = carry[col].dropna()

            mean_ann = x.mean() * 12 * 100
            t_stat   = stats.ttest_1samp(x, 0).statistic
            std_ann  = x.std() * (12 ** 0.5) * 100
            sharpe   = mean_ann / std_ann

            rows[col] = {
                "Mean (ann. %)" : round(mean_ann, 2),
                "T-statistic"   : round(t_stat,   2),
                "Std (ann. %)"  : round(std_ann,  2),
                "Sharpe ratio"  : round(sharpe,   2),
            }
        table2 = pd.DataFrame(rows)
        return table2

    def compute_forward_discount_correlation(self, fx_panel: pd.DataFrame, rates_wide: pd.DataFrame) -> pd.DataFrame:
        fx = fx_panel.copy()
        fx["ym"] = fx["date"].dt.to_period("M")
        fx["fd"] = np.log(fx["spot"] / fx["fwd1m"])

        ir = rates_wide.copy()
        ir["ym"] = ir["date"].dt.to_period("M")
        ir = ir.set_index("ym")[self.CURRENCIES + ["USD"]]
        delta_r = ir[self.CURRENCIES].subtract(ir["USD"], axis=0).reset_index()

        correlations = {}
        for ccy in self.CURRENCIES:
            fd_ccy = fx[fx["currency"] == ccy][["ym", "fd"]].set_index("ym")
            dr_ccy = delta_r[["ym", ccy]].set_index("ym").rename(columns={ccy: "delta_r"})
            merged = fd_ccy.join(dr_ccy).dropna()
            correlations[ccy] = merged["fd"].corr(merged["delta_r"])

        corr_df = pd.DataFrame.from_dict(correlations, orient="index", columns=["corr(fd, Δr)"])
        corr_df.loc["Average"] = corr_df.mean()
        return corr_df
    
    @staticmethod
    def _print_carry_summary(result: pd.DataFrame) -> None:
        print(f"\nShape      : {result.shape}  (expected 360 rows)")
        print(f"Date range : {result['date'].min().date()} → {result['date'].max().date()}")
        print("\n=== Sample (first 12 rows) ===")
        print(result.head(12).to_string(index=False))

    
