import pandas as pd

from core.tushare_client import TushareClient
from utils.cache import SQLiteCache
from utils.logger import get_logger


class DataLoader:
    def __init__(self, client: TushareClient, cache: SQLiteCache):
        self.client = client
        self.cache = cache
        self.logger = get_logger()

    def load_stock_basic(self, force_refresh: bool = False) -> pd.DataFrame:
        table, key = "stock_basic", "latest"

        if not force_refresh and self.cache.has(table, key):
            self.logger.info("Loading stock_basic from cache")
            df = self.cache.get(table, key)
            if df is not None and not df.empty:
                return df

        self.logger.info("Fetching stock_basic from Tushare")
        df = self.client.stock_basic()
        if not df.empty:
            self.cache.put(table, key, df)
        return df

    def load_daily_all(self, trade_date: str, force_refresh: bool = False) -> pd.DataFrame:
        table, key = "daily", trade_date

        if not force_refresh and self.cache.has(table, key):
            self.logger.info(f"Loading daily data ({trade_date}) from cache")
            df = self.cache.get(table, key)
            if df is not None and not df.empty:
                return df

        self.logger.info(f"Fetching daily data for {trade_date} from Tushare")
        df = self.client.daily(trade_date)
        if not df.empty:
            self.cache.put(table, key, df)
        return df

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
