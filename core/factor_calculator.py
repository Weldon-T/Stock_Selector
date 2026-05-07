import numpy as np
import pandas as pd

from core.data_loader import DataLoader
from utils.logger import get_logger


class FactorCalculator:
    def __init__(self, config: dict, loader: DataLoader):
        self.config = config
        self.loader = loader
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

        df = stock_list[["ts_code", "name", "industry"]].copy()

        # Merge daily data (pe_ttm, pb)
        daily_cols = ["ts_code", "pe_ttm", "pb", "vol", "amount"]
        available_daily = [c for c in daily_cols if c in daily_data.columns]
        df = df.merge(daily_data[available_daily], on="ts_code", how="left")

        # Load financial indicators
        period = self._get_financial_period(trade_date)
        fina = self.loader.load_fina_indicator(period)
        financial_period = period

        if fina.empty:
            self.logger.warning(f"No fina_indicator data for period={period}")
            financial_period = ""
        else:
            # Deduplicate by ts_code, keeping latest ann_date
            if "ann_date" in fina.columns:
                fina = fina.sort_values("ann_date").drop_duplicates("ts_code", keep="last")

            fina_cols = ["ts_code"]
            col_map = {}
            for fina_col, our_col in [
                ("roe", "roe_ttm"),
                ("netprofit_yoy", "net_profit_yoy"),
                ("tr_yoy", "revenue_yoy"),
                ("debt_to_assets", "debt_to_asset"),
            ]:
                if fina_col in fina.columns:
                    fina_cols.append(fina_col)
                    col_map[fina_col] = our_col

            fina_sub = fina[fina_cols].rename(columns=col_map)
            df = df.merge(fina_sub, on="ts_code", how="left")

            self.logger.info(f"Loaded financial data from period={period}, "
                             f"{len(fina_sub)} rows, fields={list(col_map.values())}")

        # Placeholder columns for M3 capital_flow factors
        for col in ["volume_ratio", "margin_chg_5d", "main_inflow_5d", "north_net_inflow"]:
            if col not in df.columns:
                df[col] = np.nan

        df["financial_period"] = financial_period

        self.logger.info(f"Factor calculation complete: {len(df)} stocks, {len(df.columns)} columns")
        return df
