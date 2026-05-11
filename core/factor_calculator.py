import numpy as np
import pandas as pd

from utils.logger import get_logger


class FactorCalculator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger()
        self.factors_config = config.get("factors", {})

    def _get_financial_period(self, trade_date: str) -> str:
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

    def _compute_volatility(self, df: pd.DataFrame, multi_daily: pd.DataFrame) -> pd.Series:
        """Compute daily return std (annualized) from multi-day daily data."""
        if multi_daily is None or multi_daily.empty or "close" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        returns = multi_daily.sort_values(["ts_code", "trade_date"]).groupby("ts_code")["close"].pct_change()
        # pct_change() gives NaN for the first row of each group — that's correct
        vol = returns.groupby(multi_daily["ts_code"]).std() * np.sqrt(250)  # annualize
        vol.name = "volatility"
        return df[["ts_code"]].merge(vol, on="ts_code", how="left")["volatility"]

    def _compute_volume_ratio(self, df: pd.DataFrame, multi_daily: pd.DataFrame) -> pd.Series:
        """Compute vol / avg_vol_5d from multi-day daily data."""
        if multi_daily is None or multi_daily.empty:
            return pd.Series(np.nan, index=df.index)

        if "vol" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        # Average volume over all available days per stock (proxy for 5-day MA)
        avg_vol = multi_daily.groupby("ts_code")["vol"].mean()
        avg_vol.name = "avg_vol"

        df_with_avg = df.merge(avg_vol, on="ts_code", how="left")
        ratio = np.where(
            df_with_avg["avg_vol"].fillna(0) > 0,
            df_with_avg["vol"].fillna(0) / df_with_avg["avg_vol"],
            np.nan,
        )
        return pd.Series(ratio, index=df.index)

    def _compute_short_reversal(self, df: pd.DataFrame, multi_daily: pd.DataFrame) -> pd.Series:
        """5-day price reversal: negative of 5-day return. Higher = more bounce potential."""
        if multi_daily is None or multi_daily.empty or "close" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        results = {}
        for code, group in multi_daily.groupby("ts_code"):
            closes = group.sort_values("trade_date")["close"].values
            if len(closes) >= 5:
                ret_5d = (closes[-1] - closes[-5]) / closes[-5]
                results[code] = -ret_5d
            else:
                results[code] = np.nan

        rev = pd.Series(results, name="short_reversal")
        rev.index.name = "ts_code"
        return df[["ts_code"]].merge(rev, on="ts_code", how="left")["short_reversal"]

    def _compute_amplitude(self, df: pd.DataFrame, multi_daily: pd.DataFrame) -> pd.Series:
        """Average daily amplitude (high-low)/close over lookback window. Lower = more stable."""
        if multi_daily is None or multi_daily.empty:
            return pd.Series(np.nan, index=df.index)
        if not all(c in multi_daily.columns for c in ["high", "low", "close"]):
            return pd.Series(np.nan, index=df.index)

        multi_daily = multi_daily.copy()
        multi_daily["daily_amp"] = (multi_daily["high"] - multi_daily["low"]) / multi_daily["close"]
        amp = multi_daily.groupby("ts_code")["daily_amp"].mean()
        amp.name = "amplitude"
        return df[["ts_code"]].merge(amp, on="ts_code", how="left")["amplitude"]

    def _compute_short_momentum(self, df: pd.DataFrame, multi_daily: pd.DataFrame) -> pd.Series:
        """3-day price momentum: positive return over last 3 days."""
        if multi_daily is None or multi_daily.empty or "close" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        results = {}
        for code, group in multi_daily.groupby("ts_code"):
            closes = group.sort_values("trade_date")["close"].values
            if len(closes) >= 3:
                ret_3d = (closes[-1] - closes[-3]) / closes[-3]
                results[code] = ret_3d
            else:
                results[code] = np.nan

        mom = pd.Series(results, name="short_momentum")
        mom.index.name = "ts_code"
        return df[["ts_code"]].merge(mom, on="ts_code", how="left")["short_momentum"]

    def _compute_amount_stability(self, df: pd.DataFrame, multi_daily: pd.DataFrame) -> pd.Series:
        """Coefficient of variation of daily trading amount. Lower = less manipulation risk."""
        if multi_daily is None or multi_daily.empty or "amount" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        grouped = multi_daily.groupby("ts_code")["amount"]
        cv = grouped.std() / grouped.mean()
        cv.name = "amount_stability"
        return df[["ts_code"]].merge(cv, on="ts_code", how="left")["amount_stability"]

    def calculate(
        self,
        stock_list: pd.DataFrame,
        daily_data: pd.DataFrame,
        trade_date: str,
        multi_daily: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        self.logger.info(f"Calculating factors for trade_date={trade_date}")

        base_cols = ["ts_code", "name", "industry", "market"]
        bak_cols = ["pe", "pb", "eps", "bvps", "gpr", "npr", "rev_yoy", "profit_yoy", "total_assets"]
        available_bak = [c for c in bak_cols if c in stock_list.columns]
        df = stock_list[base_cols + available_bak].copy()

        # --- Value: EP = 1/PE, BP = 1/PB ---
        if "pe" in df.columns:
            df["ep_ttm"] = np.where(df["pe"] > 0, 1.0 / df["pe"], np.nan)
        else:
            df["ep_ttm"] = np.nan

        if "pb" in df.columns:
            df["bp"] = np.where(df["pb"] > 0, 1.0 / df["pb"], np.nan)
        else:
            df["bp"] = np.nan

        # --- Quality: ROE, 毛利率, 净利率 ---
        if "eps" in df.columns and "bvps" in df.columns:
            df["roe_ttm"] = np.where(df["bvps"].abs() > 1e-9, df["eps"] / df["bvps"], np.nan)
        else:
            df["roe_ttm"] = np.nan

        if "gpr" in df.columns:
            df["gross_margin"] = df["gpr"]
        else:
            df["gross_margin"] = np.nan

        if "npr" in df.columns:
            df["net_margin"] = df["npr"]
        else:
            df["net_margin"] = np.nan

        # --- Growth ---
        if "rev_yoy" in df.columns:
            df["revenue_yoy"] = df["rev_yoy"]
        if "profit_yoy" in df.columns:
            df["profit_yoy"] = df["profit_yoy"]

        # --- Size: small_cap = -ln(total_assets) ---
        if "total_assets" in df.columns:
            df["small_cap"] = -np.log(df["total_assets"].clip(lower=1.0))
        else:
            df["small_cap"] = np.nan

        # --- Technical: volume_ratio from daily data ---
        daily_cols = ["ts_code", "vol", "amount"]
        available_daily = [c for c in daily_cols if c in daily_data.columns]
        df = df.merge(daily_data[available_daily], on="ts_code", how="left")
        df["volume_ratio"] = self._compute_volume_ratio(df, multi_daily)
        df["volatility"] = self._compute_volatility(df, multi_daily)
        df["short_reversal"] = self._compute_short_reversal(df, multi_daily)
        df["short_momentum"] = self._compute_short_momentum(df, multi_daily)
        df["amplitude"] = self._compute_amplitude(df, multi_daily)
        df["amount_stability"] = self._compute_amount_stability(df, multi_daily)

        # --- Metadata ---
        df["financial_period"] = self._get_financial_period(trade_date)

        # Drop raw bak columns that were renamed or are no longer needed
        drop_raw = [c for c in bak_cols + ["vol", "amount"]
                    if c in df.columns and c not in
                    ("revenue_yoy", "profit_yoy", "roe_ttm")]
        df.drop(columns=drop_raw, inplace=True, errors="ignore")

        factor_names = ["ep_ttm", "bp", "roe_ttm", "gross_margin", "net_margin",
                        "revenue_yoy", "profit_yoy", "small_cap", "volume_ratio",
                        "volatility", "short_reversal", "short_momentum",
                        "amplitude", "amount_stability"]
        available = [f for f in factor_names if f in df.columns and df[f].notna().any()]
        self.logger.info(f"Factor calculation complete: {len(df)} stocks, "
                         f"factors with data: {available}")
        return df
