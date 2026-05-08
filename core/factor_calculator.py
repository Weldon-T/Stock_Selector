import numpy as np
import pandas as pd

from utils.logger import get_logger


class FactorCalculator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()
        self.factors_config = config.get("factors", {})

    def _get_financial_period(self, trade_date: str) -> str:
        """
        Determine the latest available financial report period.
        Chinese reporting deadlines: Q1 by Apr 30, Q2 by Aug 31, Q3 by Oct 31,
        Q4 by Apr 30 next year.
        """
        year = int(trade_date[:4])
        month = int(trade_date[5:7])

        if month >= 11:
            return f"{year}0930"
        elif month >= 9:
            return f"{year}0630"
        elif month >= 5:
            return f"{year}0331"
        else:
            return f"{year - 1}1231"

    def calculate(self, stock_list: pd.DataFrame, daily_data: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        self.logger.info(f"Calculating factors for trade_date={trade_date}")

        # Include financial fields from stock_list (bak_basic)
        base_cols = ["ts_code", "name", "industry"]
        fin_fields = ["rev_yoy", "profit_yoy", "eps", "bvps"]
        available_fin = [c for c in fin_fields if c in stock_list.columns]
        df = stock_list[base_cols + available_fin].copy()

        # Derive ROE from EPS / BVPS
        if "eps" in df.columns and "bvps" in df.columns:
            df["roe_ttm"] = np.where(df["bvps"].abs() > 1e-9, df["eps"] / df["bvps"], np.nan)
        else:
            df["roe_ttm"] = np.nan

        # Rename bak_basic fields to factor names
        rename_map = {}
        if "rev_yoy" in df.columns:
            rename_map["rev_yoy"] = "revenue_yoy"
        if "profit_yoy" in df.columns:
            rename_map["profit_yoy"] = "net_profit_yoy"
        df.rename(columns=rename_map, inplace=True)

        # Merge daily data (pe_ttm, pb)
        daily_cols = ["ts_code", "pe_ttm", "pb", "vol", "amount"]
        available_daily = [c for c in daily_cols if c in daily_data.columns]
        df = df.merge(daily_data[available_daily], on="ts_code", how="left")

        # debt_to_asset unavailable without total_liab
        df["debt_to_asset"] = np.nan

        # Placeholder columns for M3 capital_flow factors
        for col in ["volume_ratio", "margin_chg_5d", "main_inflow_5d", "north_net_inflow"]:
            if col not in df.columns:
                df[col] = np.nan

        # Determine financial period from trade_date
        period = self._get_financial_period(trade_date)
        df["financial_period"] = period

        self.logger.info(f"Factor calculation complete: {len(df)} stocks, "
                         f"financial fields={list(rename_map.values())}, period={period}")
        return df
