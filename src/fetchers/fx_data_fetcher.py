import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path
import wrds


class FXDataFetcher:
    """Fetches and processes FX spot/forward rates from WRDS."""

    START_DATE = "1990-12-01"
    END_DATE   = "2024-12-31"

    SERIES = [
        ("AUD", "spot",  2427, True),
        ("CAD", "spot",  2429, True),
        ("EUR", "spot",  2561, False),
        ("GBP", "spot",  2428, True),
        ("JPY", "spot",  2538, True),
        ("NZD", "spot",  2441, True),
        ("AUD", "fwd1m", 2601, False),
        ("CAD", "fwd1m", 2616, True),
        ("EUR", "fwd1m", 2562, False),
        ("GBP", "fwd1m", 2539, False),
        ("JPY", "fwd1m", 2544, True),
        ("NZD", "fwd1m", 2676, False),
    ]

    FALLBACK_SPOT = [
        ("AUD", 2594, False),
        ("NZD", 2595, False),
    ]

    def __init__(self, wrds_username: str):
        self.wrds_username = wrds_username
        self.meta    = pd.DataFrame(self.SERIES,        columns=["currency", "rate_type", "exrateintcode", "invert"])
        self.meta_fb = pd.DataFrame(self.FALLBACK_SPOT, columns=["currency", "exrateintcode", "invert"])

    def _fetch_and_collapse(
        self,
        db: wrds.Connection,
        codes: list[int],
        meta_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fetch daily rates from WRDS and collapse to month-end observations."""
        codes_sql = ", ".join(str(c) for c in codes)
        raw = db.raw_sql(
            f"""
            SELECT exrateintcode, exratedate AS date, midrate AS rate
            FROM tr_ds_equities.ds2fxrate
            WHERE exrateintcode IN ({codes_sql})
              AND exratedate BETWEEN '{self.START_DATE}' AND '{self.END_DATE}'
            ORDER BY exrateintcode, exratedate
            """,
            date_cols=["date"],
        )

        raw = raw.merge(meta_df, on="exrateintcode", how="left")

        mask = raw["invert"].astype(bool)
        raw.loc[mask, "rate"] = 1.0 / raw.loc[mask, "rate"]

        raw["ym"] = raw["date"].dt.to_period("M")
        idx     = raw.groupby(["exrateintcode", "ym"])["date"].idxmax()
        monthly = raw.loc[idx].copy()
        monthly["date"] = monthly["ym"].dt.to_timestamp("M")
        monthly.drop(columns=["ym", "exrateintcode", "invert"], inplace=True)

        return monthly

    def fetch(self) -> pd.DataFrame:
        """
        Connect to WRDS, fetch all FX series, and return a clean panel.

        Returns
        -------
        pd.DataFrame
            Columns: date | currency | spot | fwd1m
        """
        print("Connecting to WRDS …")
        db = wrds.Connection(wrds_username=self.wrds_username)

        try:
            print("Fetching primary series …")
            monthly = self._fetch_and_collapse(db, self.meta["exrateintcode"].tolist(), self.meta)

            print("Fetching fallback spot series (AUD, NZD) …")
            fb = self._fetch_and_collapse(db, self.meta_fb["exrateintcode"].tolist(), self.meta_fb)
            fb["rate_type"] = "spot"
        finally:
            db.close()

        # Pivot to wide
        panel = (
            monthly
            .pivot_table(index=["date", "currency"], columns="rate_type", values="rate", aggfunc="first")
            .reset_index()
        )
        panel.columns.name = None
        panel = panel[["date", "currency", "spot", "fwd1m"]]

        # Fill missing spot from fallback
        fb_spot = fb[["date", "currency", "rate"]].rename(columns={"rate": "spot_fb"})
        panel   = panel.merge(fb_spot, on=["date", "currency"], how="left")
        missing = panel["spot"].isna() & panel["spot_fb"].notna()
        print(f"\nFilling {missing.sum()} missing spot values from fallback series.")
        panel.loc[missing, "spot"] = panel.loc[missing, "spot_fb"]
        panel.drop(columns="spot_fb", inplace=True)

        panel = panel.sort_values(["date", "currency"]).reset_index(drop=True)
        self._print_summary(panel)
        return panel

    def save(self, panel: pd.DataFrame, filename: str = "fx_monthly_panel.csv") -> None:
        folder = Path("data/raw")
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / filename
        panel.to_csv(out, index=False)
        print(f"Saved → {out}")

    @staticmethod
    def _print_summary(panel: pd.DataFrame) -> None:
        n_months = panel["date"].nunique()
        print(f"\nMonths     : {n_months}")
        print(f"Obs        : {len(panel)}  (expected {n_months * 6})")
        print(f"Currencies : {sorted(panel['currency'].unique())}")
        print(f"Date range : {panel['date'].min().date()} → {panel['date'].max().date()}")
        print(f"Missing spot  : {panel['spot'].isna().sum()}")
        print(f"Missing fwd1m : {panel['fwd1m'].isna().sum()}")
        print("\n=== Sample (first 18 rows) ===")
        print(panel.head(18).to_string(index=False))
