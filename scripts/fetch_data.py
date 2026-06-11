import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fetchers.fx_data_fetcher import FXDataFetcher
from fetchers.interest_rate_fetcher import InterestRateFetcher
from config import WRDS_USERNAME

fx = FXDataFetcher(wrds_username=WRDS_USERNAME)
fx_panel = fx.fetch()
fx.save(fx_panel)

ir = InterestRateFetcher()
rates_wide, rates_long = ir.fetch()
ir.save(rates_wide, rates_long)
