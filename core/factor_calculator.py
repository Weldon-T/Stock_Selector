import numpy as np
import pandas as pd

from core.tushare_client import TushareClient
from utils.cache import SQLiteCache
from utils.logger import get_logger


class FactorCalculator:
    def __init__(self, config: dict, client: TushareClient, cache: SQLiteCache):
        self.config = config
        self.client = client
        self.cache = cache
        self.logger = get_logger()
        self.factors_config = config.get("factors", {})

    def calculate(self, stock_list: pd.DataFrame, daily_data: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        self.logger.info(f"Calculating factors for trade_date={trade_date}")

        df = stock_list[["ts_code", "name", "industry"]].copy()

        # Merge daily data (PE_TTM, PB)
        daily_cols = ["ts_code", "pe_ttm", "pb", "vol", "amount"]
        available_daily = [c for c in daily_cols if c in daily_data.columns]
        df = df.merge(daily_data[available_daily], on="ts_code", how="left")

        # Placeholder columns for factors to be implemented in M2/M3
        placeholder_factors = [
            "roe_ttm", "net_profit_yoy", "revenue_yoy", "debt_to_asset",
            "volume_ratio", "margin_chg_5d", "main_inflow_5d", "north_net_inflow",
        ]
        for col in placeholder_factors:
            if col not in df.columns:
                df[col] = np.nan

        df["financial_period"] = ""

        self.logger.info(f"Factor calculation complete: {len(df)} stocks, {len(df.columns)} columns")
        return df
