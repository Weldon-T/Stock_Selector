import pandas as pd

from utils.date_utils import parse_date
from utils.logger import get_logger


class StockFilter:
    def __init__(self, config: dict):
        pool = config.get("stock_pool", {})
        self.markets = pool.get("markets", ["主板", "创业板", "科创板"])
        self.min_listing_days = pool.get("min_listing_days", 60)
        self.max_suspend_days = pool.get("max_suspend_days", 10)
        self.min_daily_amount = pool.get("min_daily_amount", 50_000_000)
        self.logger = get_logger()

    def filter_market(self, df: pd.DataFrame) -> pd.DataFrame:
        if "market" not in df.columns:
            self.logger.warning("No 'market' column; skipping market filter")
            return df
        before = len(df)
        result = df[df["market"].isin(self.markets)].copy()
        self.logger.info(f"Market filter: {before} -> {len(result)} stocks")
        return result

    def filter_st(self, df: pd.DataFrame) -> pd.DataFrame:
        if "is_st" not in df.columns:
            self.logger.warning("No 'is_st' column; skipping ST filter")
            return df
        before = len(df)
        result = df[df["is_st"] != 1].copy()
        self.logger.info(f"ST filter: {before} -> {len(result)} stocks")
        return result

    def filter_new_stocks(self, df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        if "list_date" not in df.columns:
            self.logger.warning("No 'list_date' column; skipping new-stock filter")
            return df
        before = len(df)
        cutoff_dt = pd.to_datetime(trade_date, format="%Y%m%d") - pd.Timedelta(days=self.min_listing_days)
        list_dates = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
        mask = list_dates.notna() & (list_dates <= cutoff_dt)
        result = df[mask].copy()
        self.logger.info(f"New stock filter (<{self.min_listing_days}d): {before} -> {len(result)} stocks")
        return result

    def filter_suspended(self, df: pd.DataFrame, daily_data: pd.DataFrame) -> pd.DataFrame:
        if daily_data.empty:
            self.logger.warning("No daily data; skipping suspension filter")
            return df
        before = len(df)
        active_codes = set(daily_data["ts_code"].unique())
        result = df[df["ts_code"].isin(active_codes)].copy()
        self.logger.info(f"Suspension filter: {before} -> {len(result)} stocks")
        return result

    def filter_liquidity(self, df: pd.DataFrame, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Remove stocks with average daily turnover below threshold."""
        if "amount" not in daily_data.columns:
            self.logger.warning("No 'amount' in daily data; skipping liquidity filter")
            return df
        before = len(df)
        avg_amount = daily_data.groupby("ts_code")["amount"].mean()
        liquid_codes = set(avg_amount[avg_amount >= self.min_daily_amount].index)
        result = df[df["ts_code"].isin(liquid_codes)].copy()
        self.logger.info(f"Liquidity filter (amount>={self.min_daily_amount/1000:.0f}M yuan): "
                         f"{before} -> {len(result)} stocks")
        return result

    def apply(self, df_basic: pd.DataFrame, df_daily: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        self.logger.info(f"Applying filters for trade_date={trade_date}")

        df = self.filter_market(df_basic)
        df = self.filter_st(df)
        df = self.filter_new_stocks(df, trade_date)
        df = self.filter_suspended(df, df_daily)
        df = self.filter_liquidity(df, df_daily)

        if df.empty:
            self.logger.warning("All stocks filtered out — pipeline will abort")
        else:
            self.logger.info(f"Filtered stock pool: {len(df)} stocks remain")

        return df
