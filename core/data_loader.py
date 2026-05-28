import pandas as pd

from core.tushare_client import TushareClient
from utils.cache import SQLiteCache
from utils.logger import get_logger


class DataLoader:
    def __init__(self, client: TushareClient, cache: SQLiteCache):
        self.client = client
        self.cache = cache
        self.logger = get_logger()

    def load_stock_basic(self, trade_date: str = "", force_refresh: bool = False) -> pd.DataFrame:
        key = trade_date if trade_date else "latest"
        table = "stock_basic"

        if not force_refresh and self.cache.has(table, key):
            self.logger.info("Loading stock_basic from cache")
            df = self.cache.get(table, key)
            if df is not None and not df.empty:
                return df

        self.logger.info("Fetching stock_basic from Tushare")
        df = self.client.stock_basic(trade_date)
        if df.empty and trade_date:
            df = self._fallback_stock_basic(table, key, trade_date)
        if not df.empty:
            self.cache.put(table, key, df)
        return df

    def _fallback_stock_basic(self, table: str, key: str, trade_date: str) -> pd.DataFrame:
        """Walk back up to 10 calendar days to find a date with bak_basic data.
        Checks all cached dates first, then tries API with 65s waits for rate limit."""
        import time
        from datetime import timedelta
        dt = pd.to_datetime(trade_date, format="%Y%m%d")
        fallback_dates = [(dt - timedelta(days=i)).strftime("%Y%m%d") for i in range(1, 11)]

        # Phase 1: check cache for any recent date (fast path)
        for prev in fallback_dates:
            if self.cache.has(table, prev):
                df = self.cache.get(table, prev)
                if df is not None and not df.empty:
                    self.logger.info(f"Using stock_basic from {prev} (cached) as fallback for {trade_date}")
                    return df

        # Phase 2: no cached date found, try API with rate-limit waits
        self.logger.info("No cached fallback found, will try API with rate-limit waits")
        for prev in fallback_dates:
            self.logger.info("Waiting 65s for bak_basic rate limit (1/min)")
            time.sleep(65)
            self.logger.info(f"bak_basic empty for {trade_date}, trying {prev}")
            df = self.client.stock_basic(prev)
            if not df.empty:
                self.cache.put(table, prev, df)
                self.logger.info(f"Using stock_basic from {prev} as fallback for {trade_date}")
                return df
        self.logger.warning(f"No stock_basic data found within 10 days of {trade_date}")
        return pd.DataFrame()

    def load_daily_all(self, trade_date: str, force_refresh: bool = False) -> pd.DataFrame:
        table, key = "daily", trade_date

        if not force_refresh and self.cache.has(table, key):
            self.logger.info(f"Loading daily data ({trade_date}) from cache")
            df = self.cache.get(table, key)
            if df is not None and not df.empty:
                return df

        self.logger.info(f"Fetching daily data for {trade_date} from Tushare")
        df = self.client.daily(trade_date)
        if df.empty:
            df = self._fallback_daily(table, trade_date)
        if not df.empty:
            self.cache.put(table, key, df)
        return df

    def _fallback_daily(self, table: str, trade_date: str) -> pd.DataFrame:
        """Use the latest cached daily data as fallback."""
        latest = self.cache.get_latest_date(table)
        if latest and latest != trade_date:
            df = self.cache.get(table, latest)
            if df is not None and not df.empty:
                self.logger.info(f"Using daily data from {latest} (cached) as fallback for {trade_date}")
                return df
        self.logger.warning(f"No daily data found for {trade_date} and no cache fallback available")
        return pd.DataFrame()

    def load_fina_indicator(self, period: str, force_refresh: bool = False) -> pd.DataFrame:
        table, key = "fina_indicator", period

        if not force_refresh and self.cache.has(table, key):
            self.logger.info(f"Loading fina_indicator ({period}) from cache")
            df = self.cache.get(table, key)
            if df is not None and not df.empty:
                return df

        self.logger.info(f"Fetching fina_indicator for period={period}")
        df = self.client.fina_indicator(period)
        if not df.empty:
            self.cache.put(table, key, df)
        return df

    def load_daily_multi(self, end_date: str, lookback: int) -> pd.DataFrame:
        """Fetch daily data for N days ending at end_date. Returns concatenated DataFrame."""
        import time
        frames = []
        # Generate dates in reverse (newest first), Tushare date format is YYYYMMDD
        end_dt = pd.to_datetime(end_date, format="%Y%m%d")
        for i in range(lookback):
            dt = end_dt - pd.Timedelta(days=i)
            date_str = dt.strftime("%Y%m%d")
            table, key = "daily", date_str

            if self.cache.has(table, key):
                df = self.cache.get(table, key)
            else:
                self.logger.info(f"Fetching daily for {date_str}")
                df = self.client.daily(date_str)
                if not df.empty:
                    self.cache.put(table, key, df)
                time.sleep(0.3)

            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        self.logger.info(f"Loaded {len(frames)}/{lookback} days of daily data, {len(result)} total rows")
        return result

    def load_trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        table, key = "trade_cal", f"{start_date}_{end_date}"

        if self.cache.has(table, key):
            self.logger.info("Loading trade calendar from cache")
            df = self.cache.get(table, key)
            if df is not None and not df.empty:
                return df

        self.logger.info(f"Fetching trade calendar {start_date}~{end_date}")
        df = self.client.trade_cal(start_date, end_date)
        if not df.empty:
            self.cache.put(table, key, df)
        return df
