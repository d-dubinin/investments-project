import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from typing import Optional
from pathlib import Path


class InterestRateFetcher:
    """Fetches short-term interest rates from FRED via pandas-datareader."""

    START = datetime(1990, 12, 1)
    END   = datetime(2024, 12, 31)

    SERIES_MAP = {
        "AUD": [("IRSTCI01AUM156N", None,                   None)],
        "CAD": [("IRSTCI01CAM156N", None,                   None)],
        "EUR": [("IRSTCI01DEM156N", None,                   datetime(1998, 12, 31)),
                ("IRSTCI01EZM156N", datetime(1999, 1, 1),   None)],
        "GBP": [("IRSTCI01GBM156N", None,                   None)],
        "JPY": [("IRSTCI01JPM156N", None,                   None)],
        "NZD": [("IRSTCI01NZM156N", None,                   None)],
        "USD": [("IRSTCI01USM156N", None,                   None)],
    }

    def fetch(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch all interest rate series from FRED.

        Returns
        -------
        rates_wide : pd.DataFrame   columns: date | AUD | CAD | EUR | GBP | JPY | NZD | USD
        rates_long : pd.DataFrame   columns: date | currency | rate
        """
        frames = []
        for currency, segments in self.SERIES_MAP.items():
            print(f"Fetching {currency} …")
            parts = []
            for series_id, seg_start, seg_end in segments:
                df = web.DataReader(series_id, "fred", seg_start or self.START, seg_end or self.END)
                df = df.rename(columns={series_id: "rate"})
                df.index = pd.to_datetime(df.index)
                df = df.loc[(df.index >= pd.Timestamp(self.START)) &
                            (df.index <= pd.Timestamp(self.END))]
                parts.append(df)

            ccy_df = pd.concat(parts)
            ccy_df = ccy_df[~ccy_df.index.duplicated()].sort_index()
            ccy_df["currency"] = currency
            ccy_df = ccy_df.reset_index().rename(columns={"DATE": "date", "index": "date"})
            frames.append(ccy_df)

        rates_long = pd.concat(frames, ignore_index=True)
        rates_long.columns = [c.lower() for c in rates_long.columns]
        #rates_long["rate"] = rates_long["rate"] / 1200

        rates_wide = (
            rates_long
            .pivot(index="date", columns="currency", values="rate")
            .reset_index()
        )
        rates_wide.columns.name = None

        self._print_summary(rates_wide)
        return rates_wide, rates_long

    def save(
            self,
            rates_wide: pd.DataFrame,
            rates_long: pd.DataFrame,
            wide_filename: str = "ir_monthly_wide.csv",
            long_filename: str = "ir_monthly_long.csv",
            ) -> None:
        folder = Path("data/raw")
        folder.mkdir(parents=True, exist_ok=True)
        wide_out = folder / wide_filename
        long_out = folder / long_filename
        rates_wide.to_csv(wide_out, index=False)
        rates_long.to_csv(long_out, index=False)
        print(f"Saved → {wide_out}  and  {long_out}")

    @staticmethod
    def _print_summary(rates_wide: pd.DataFrame) -> None:
        print(f"\nShape (wide) : {rates_wide.shape}")
        print(f"Date range   : {rates_wide['date'].min().date()} → {rates_wide['date'].max().date()}")
        print(f"Missing values:\n{rates_wide.isna().sum()}")
        print("\n=== Sample (first 12 rows) ===")
        print(rates_wide.head(12).to_string(index=False))
