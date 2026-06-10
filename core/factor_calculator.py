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

    def _compute_industry_hotness(self, df: pd.DataFrame, multi_daily: pd.DataFrame | None,
                                  days: int = 10) -> pd.Series:
        """Industry hotness: median N-day return by industry, percentile ranked.
        Uses price data already in cache — no extra API call needed."""
        if multi_daily is None or multi_daily.empty or "close" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        stock_ret = self._n_day_return(multi_daily, days)
        merged = df[["ts_code", "industry"]].merge(
            stock_ret.rename("ret"), on="ts_code", how="left")

        industry_ret = merged.groupby("industry")["ret"].median()
        industry_rank = industry_ret.rank(pct=True)
        self.logger.info(f"Industry hotness ({days}d): {len(industry_ret)} industries, "
                         f"top3={industry_ret.nlargest(3).to_dict()}")

        return df["industry"].map(industry_rank)

    def _compute_sector_momentum(self, df: pd.DataFrame, multi_daily: pd.DataFrame | None,
                                 days: int = 20) -> pd.Series:
        """20-day industry median return, percentile ranked. Longer lookback for trend."""
        if multi_daily is None or multi_daily.empty or "close" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        stock_ret = self._n_day_return(multi_daily, days)
        merged = df[["ts_code", "industry"]].merge(
            stock_ret.rename("ret"), on="ts_code", how="left")

        industry_ret = merged.groupby("industry")["ret"].median()
        industry_rank = industry_ret.rank(pct=True)
        self.logger.info(f"Sector momentum ({days}d): top3={industry_ret.nlargest(3).to_dict()}")

        return df["industry"].map(industry_rank)

    def _compute_price_momentum(self, df: pd.DataFrame, multi_daily: pd.DataFrame | None,
                                days: int = 20) -> pd.Series:
        """Individual stock N-day return. Higher = stronger uptrend."""
        if multi_daily is None or multi_daily.empty or "close" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        ret = self._n_day_return(multi_daily, days)
        ret.name = "price_momentum"
        return df[["ts_code"]].merge(ret, on="ts_code", how="left")["price_momentum"]

    def _compute_volume_breakout(self, df: pd.DataFrame, multi_daily: pd.DataFrame | None,
                                 days: int = 10) -> pd.Series:
        """Recent volume / N-day average volume. >1 = volume expansion (hot money interest)."""
        if multi_daily is None or multi_daily.empty or "vol" not in multi_daily.columns:
            return pd.Series(np.nan, index=df.index)

        md = multi_daily.sort_values(["ts_code", "trade_date"])
        avg_vol = md.groupby("ts_code")["vol"].apply(
            lambda x: x.tail(days).mean() if len(x) >= days else x.mean()
        )
        latest_vol = md.groupby("ts_code")["vol"].last()

        # Avoid div-by-zero: where avg_vol is 0, use NaN
        ratio = latest_vol / avg_vol.replace(0, np.nan)
        ratio.name = "volume_breakout"
        self.logger.info(f"Volume breakout: median={ratio.median():.2f}, "
                         f"p90={ratio.quantile(0.9):.2f}")
        return df[["ts_code"]].merge(ratio, on="ts_code", how="left")["volume_breakout"]

    @staticmethod
    def _n_day_return(multi_daily: pd.DataFrame, days: int) -> pd.Series:
        """Per-stock N-day close return from multi_daily data."""
        results = {}
        for code, group in multi_daily.groupby("ts_code"):
            closes = group.sort_values("trade_date")["close"].values
            if len(closes) >= days + 1:
                results[code] = (closes[-1] - closes[-days - 1]) / closes[-days - 1]
            elif len(closes) >= 2:
                results[code] = (closes[-1] - closes[0]) / closes[0]
            else:
                results[code] = np.nan

        ret = pd.Series(results, name="n_day_ret")
        ret.index.name = "ts_code"
        return ret

    def _compute_mf_ratio(self, df: pd.DataFrame, moneyflow: pd.DataFrame | None) -> pd.Series:
        """Individual stock 5-day net main force inflow / total buy amount."""
        if moneyflow is None or moneyflow.empty:
            return pd.Series(np.nan, index=df.index)

        buy_cols = ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"]
        available_buy = [c for c in buy_cols if c in moneyflow.columns]
        if not available_buy or "net_mf_amount" not in moneyflow.columns:
            return pd.Series(np.nan, index=df.index)

        mf = moneyflow.copy()
        mf["total_buy"] = mf[available_buy].sum(axis=1)

        stock_flow = mf.groupby("ts_code").agg({"net_mf_amount": "sum", "total_buy": "sum"})
        stock_flow["mf_ratio"] = np.where(
            stock_flow["total_buy"] > 0,
            stock_flow["net_mf_amount"] / stock_flow["total_buy"],
            np.nan,
        )
        return df[["ts_code"]].merge(stock_flow[["mf_ratio"]], on="ts_code", how="left")["mf_ratio"]

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
        moneyflow: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        self.logger.info(f"Calculating factors for trade_date={trade_date}")

        base_cols = ["ts_code", "name", "industry", "market"]
        bak_cols = ["pe", "pb", "eps", "bvps", "gpr", "npr", "rev_yoy", "profit_yoy", "total_assets", "dv_ratio"]
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

        # --- Dividend yield ---
        if "dv_ratio" in df.columns:
            df["dividend_yield"] = df["dv_ratio"]
        else:
            df["dividend_yield"] = np.nan

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

        # --- Price-based hotness: industry level ---
        df["industry_hotness"] = self._compute_industry_hotness(df, multi_daily, days=10)
        # --- Moneyflow-based ratio (kept for backward compat; returns NaN when no data) ---
        df["mf_ratio"] = self._compute_mf_ratio(df, moneyflow)

        # --- Growth/momentum factors for offensive strategies ---
        df["sector_momentum"] = self._compute_sector_momentum(df, multi_daily, days=20)
        df["price_momentum"] = self._compute_price_momentum(df, multi_daily, days=20)
        df["volume_breakout"] = self._compute_volume_breakout(df, multi_daily, days=10)

        # --- Metadata ---
        df["financial_period"] = self._get_financial_period(trade_date)

        # Drop raw bak columns that were renamed or are no longer needed
        drop_raw = [c for c in bak_cols + ["vol", "amount"]
                    if c in df.columns and c not in
                    ("revenue_yoy", "profit_yoy", "roe_ttm")]
        df.drop(columns=drop_raw, inplace=True, errors="ignore")

        factor_names = ["ep_ttm", "bp", "roe_ttm", "gross_margin", "net_margin",
                        "revenue_yoy", "profit_yoy", "small_cap", "dividend_yield",
                        "volume_ratio",
                        "volatility", "short_reversal", "short_momentum",
                        "amplitude", "amount_stability",
                        "industry_hotness", "mf_ratio",
                        "sector_momentum", "price_momentum", "volume_breakout"]
        available = [f for f in factor_names if f in df.columns and df[f].notna().any()]
        self.logger.info(f"Factor calculation complete: {len(df)} stocks, "
                         f"factors with data: {available}")
        return df
