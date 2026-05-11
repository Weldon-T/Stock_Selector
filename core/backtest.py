import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from core.tushare_client import TushareClient
from core.data_loader import DataLoader
from core.filter import StockFilter
from core.factor_calculator import FactorCalculator
from core.stock_scorer import StockScorer
from utils.cache import SQLiteCache
from utils.date_utils import format_date
from utils.logger import get_logger


def _quarter_end_dates(end_date: str):
    """Generate the 4 most recent quarter-end dates (YYYYMMDD)."""
    end_dt = pd.to_datetime(end_date, format="%Y%m%d")
    dates = []
    y, m = end_dt.year, end_dt.month
    for _ in range(6):
        if m <= 3:
            m, y = 12, y - 1
        elif m <= 6:
            m = 3
        elif m <= 9:
            m = 6
        else:
            m = 9
        day = 31 if m in (12, 3) else 30
        dt_str = f"{y}{m:02d}{day}"
        if dt_str <= end_date:
            dates.append(dt_str)
        if len(dates) >= 4:
            break
        m -= 1
    return sorted(dates)


def _resolve_date(loader, date_str: str) -> str | None:
    """Find the nearest valid trading date at or before date_str."""
    dt = pd.to_datetime(date_str, format="%Y%m%d")
    for i in range(10):
        d = (dt - pd.Timedelta(days=i)).strftime("%Y%m%d")
        df = loader.load_daily_all(d)
        if not df.empty:
            return d
    return None


class Backtest:
    def __init__(self, config: dict):
        bt_cfg = config.get("backtest", {})
        self.hold_months = bt_cfg.get("hold_months", 3)
        self.config = config
        self.logger = get_logger()

    def _select_at_date(self, loader, stock_filter, factor_calc, scorer,
                        end_date: str) -> pd.DataFrame | None:
        """Run multi-quarter selection for a single date. Returns selected stocks."""
        quarter_targets = _quarter_end_dates(end_date)
        quarter_weights = [0.2, 0.2, 0.3, 0.3]

        # Step 1: Run each quarter
        quarter_results = []
        for qd, w in zip(quarter_targets, quarter_weights):
            resolved = _resolve_date(loader, qd)
            if not resolved:
                continue

            df_basic = loader.load_stock_basic(resolved)
            if df_basic.empty:
                continue

            df_daily = loader.load_daily_all(resolved)
            if df_daily.empty:
                continue

            df_multi_daily = loader.load_daily_multi(resolved, lookback=self.config.get("multi_daily_lookback", 10))
            df_filtered = stock_filter.apply(df_basic, df_daily, resolved)
            if df_filtered.empty:
                continue

            df_factors = factor_calc.calculate(df_filtered, df_daily, resolved, df_multi_daily)
            if len(df_factors) > 100:
                quarter_results.append({"quarter": resolved, "weight": w, "df": df_factors})

        if len(quarter_results) < 2:
            return None

        # Step 2: Aggregate weighted ranks
        factor_names = [f["name"] for f in scorer.factors]
        rank_cols = [f"{fn}_rank" for fn in factor_names]

        min_periods = max(1, len(quarter_results) // 2)
        ts_counter = defaultdict(int)
        for qr in quarter_results:
            for code in qr["df"]["ts_code"]:
                ts_counter[code] += 1
        valid_codes = {c for c, n in ts_counter.items() if n >= min_periods}

        rank_sums: dict = {c: {rc: 0.0 for rc in rank_cols} for c in valid_codes}
        value_sums: dict = {c: {fn: 0.0 for fn in factor_names} for c in valid_codes}
        weight_sums: dict = {c: 0.0 for c in valid_codes}

        for qr in quarter_results:
            w = qr["weight"]
            scored = scorer.score_all(qr["df"], sector_neutral=True)
            for _, row in scored.iterrows():
                code = row["ts_code"]
                if code not in valid_codes:
                    continue
                for rc in rank_cols:
                    val = row.get(rc, np.nan)
                    if not np.isnan(val):
                        rank_sums[code][rc] += val * w
                for fn in factor_names:
                    val = row.get(fn, np.nan)
                    if not np.isnan(val):
                        value_sums[code][fn] += val * w
                weight_sums[code] += w

        # Overwrite technical factors with latest quarter only
        TECH_FACTORS = {"short_reversal", "short_momentum", "volatility", "amplitude", "amount_stability", "volume_ratio"}
        latest_qr = quarter_results[-1]
        latest_scored = scorer.score_all(latest_qr["df"], sector_neutral=True)
        latest_lookup = latest_scored.set_index("ts_code")
        for code in valid_codes:
            if code not in latest_lookup.index:
                continue
            row = latest_lookup.loc[code]
            for fn in TECH_FACTORS:
                if fn in latest_lookup.columns:
                    val = row[fn]
                    if not np.isnan(val):
                        value_sums[code][fn] = val
                rc = f"{fn}_rank"
                if rc in latest_lookup.columns:
                    val = row[rc]
                    if not np.isnan(val):
                        rank_sums[code][rc] = val

        # Step 3: Build scored DataFrame
        records = []
        for code in valid_codes:
            if weight_sums[code] == 0:
                continue
            avg_ranks = {rc: rank_sums[code][rc] / weight_sums[code] for rc in rank_cols}
            avg_values = {fn: value_sums[code][fn] / weight_sums[code] for fn in factor_names}
            final_score = sum(avg_ranks.values()) / len(rank_cols)

            name, industry, market = "", "", ""
            for qr in reversed(quarter_results):
                r = qr["df"][qr["df"]["ts_code"] == code]
                if not r.empty:
                    name = r.iloc[0].get("name", "")
                    industry = r.iloc[0].get("industry", "")
                    market = r.iloc[0].get("market", "")
                    break

            records.append({
                "ts_code": code, "name": name, "industry": industry, "market": market,
                "final_score": final_score,
                **avg_values, **avg_ranks,
            })

        df_all = pd.DataFrame(records).sort_values("final_score", ascending=False)

        # Per-market selection
        select_count = self.config.get("select_count", {"主板": 50, "创业板": 20, "科创板": 20})
        parts = []
        for mkt, count in select_count.items():
            subset = df_all[df_all["market"] == mkt].head(count)
            parts.append(subset)

        return pd.concat(parts, ignore_index=True) if parts else None

    def _compute_returns(self, loader, codes: list[str], from_date: str,
                         to_date: str) -> dict[str, float]:
        """Compute forward returns for a list of stocks. Loads daily data once per period."""
        df_from = loader.load_daily_all(from_date)
        df_to = loader.load_daily_all(to_date)

        if df_from.empty or df_to.empty:
            return {}

        close_from = df_from.set_index("ts_code")["close"]
        close_to = df_to.set_index("ts_code")["close"]

        returns = {}
        for code in codes:
            if code in close_from.index and code in close_to.index:
                p_from = close_from[code]
                p_to = close_to[code]
                if p_from > 0 and p_to > 0:
                    returns[code] = (p_to - p_from) / p_from
        return returns

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger = self.logger
        logger.info(f"=== Backtest: {format_date(start_date)} ~ {format_date(end_date)}, "
                     f"hold={self.hold_months}mo ===")

        cache_dir = self.config.get("paths", {}).get("cache_dir", "./cache")
        cache = SQLiteCache(cache_dir)
        client = TushareClient(self.config["tushare_token"])
        if not client.login():
            logger.error("Tushare login failed.")
            sys.exit(1)

        loader = DataLoader(client, cache)
        stock_filter = StockFilter(self.config)
        factor_calc = FactorCalculator(self.config)
        scorer = StockScorer(self.config)

        # Generate rebalancing dates at hold_months intervals
        rebalance_dates = []
        dt = pd.to_datetime(start_date, format="%Y%m%d")
        end_dt = pd.to_datetime(end_date, format="%Y%m%d")
        while dt <= end_dt:
            # End of current month as target; resolve to nearest trade date
            month_end = dt + pd.offsets.MonthEnd(0)
            d_str = month_end.strftime("%Y%m%d")
            if d_str >= start_date:
                resolved = _resolve_date(loader, d_str)
                if resolved and resolved not in rebalance_dates:
                    rebalance_dates.append(resolved)
            # Advance to first day of next interval
            dt = month_end + pd.DateOffset(months=self.hold_months - 1, days=1)

        logger.info(f"Rebalance dates: {[format_date(d) for d in rebalance_dates]}")

        # Run backtest
        periods = []
        for i, rb_date in enumerate(rebalance_dates):
            logger.info(f"--- Rebalance {i+1}/{len(rebalance_dates)}: {format_date(rb_date)} ---")

            # Select stocks
            selected = self._select_at_date(loader, stock_filter, factor_calc, scorer, rb_date)
            if selected is None or selected.empty:
                logger.warning(f"  No stocks selected, skipping")
                continue

            # Determine forward date
            rb_dt = pd.to_datetime(rb_date, format="%Y%m%d")
            fwd_dt = rb_dt + pd.DateOffset(months=self.hold_months)
            fwd_str = fwd_dt.strftime("%Y%m%d")
            fwd_date = _resolve_date(loader, fwd_str)
            if not fwd_date:
                logger.warning(f"  No forward date near {fwd_str}, skipping")
                continue

            # Compute forward returns (batch: load daily once per period)
            port_codes = selected["ts_code"].tolist()
            port_rets = self._compute_returns(loader, port_codes, rb_date, fwd_date)
            if len(port_rets) < 10:
                logger.warning(f"  Only {len(port_rets)} valid returns, skipping")
                continue
            port_ret = np.mean(list(port_rets.values()))

            # Benchmark: all FILTERED stocks (same universe we select from)
            df_basic = loader.load_stock_basic(rb_date)
            df_daily = loader.load_daily_all(rb_date)
            df_filtered_all = stock_filter.apply(df_basic, df_daily, rb_date)
            if df_filtered_all.empty:
                bench_ret = 0.0
            else:
                bench_codes = df_filtered_all["ts_code"].unique().tolist()
                bench_rets = self._compute_returns(loader, bench_codes, rb_date, fwd_date)
                bench_ret = np.mean(list(bench_rets.values())) if bench_rets else 0.0

            periods.append({
                "rebalance_date": rb_date,
                "forward_date": fwd_date,
                "stocks_selected": len(selected),
                "valid_returns": len(port_rets),
                "port_return": port_ret,
                "bench_return": bench_ret,
                "excess_return": port_ret - bench_ret,
            })

            logger.info(f"  port={port_ret:.4f} bench={bench_ret:.4f} excess={port_ret - bench_ret:.4f}")

        if len(periods) < 2:
            logger.error(f"Only {len(periods)} valid periods, need >= 2")
            return pd.DataFrame()

        df_periods = pd.DataFrame(periods)

        # Compute statistics
        port_returns = df_periods["port_return"].values
        bench_returns = df_periods["bench_return"].values
        excess_returns = df_periods["excess_return"].values

        # Cumulative return
        port_cum = np.prod(1 + port_returns) - 1
        bench_cum = np.prod(1 + bench_returns) - 1

        # Annualized
        n_periods = len(port_returns)
        years = n_periods * self.hold_months / 12
        port_ann = (1 + port_cum) ** (1 / years) - 1 if years > 0 else 0
        bench_ann = (1 + bench_cum) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio (annualized from per-period returns)
        rf_per_period = 0.02 * self.hold_months / 12  # 2% annual risk-free
        excess = port_returns - rf_per_period
        periods_per_year = 12 / self.hold_months
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year) if np.std(excess) > 0 else 0

        # Max drawdown
        cum_series = np.cumprod(1 + port_returns)
        peak = np.maximum.accumulate(cum_series)
        drawdowns = (cum_series - peak) / peak
        max_dd = drawdowns.min()

        # Win rate
        win_rate = np.mean(port_returns > bench_returns)

        # Calmar ratio
        calmar = port_ann / abs(max_dd) if max_dd != 0 else 0

        # Print summary
        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Periods:              {n_periods} quarters")
        print(f"  Hold per period:      {self.hold_months} months")
        print(f"  Time span:            {years:.1f} years")
        print(f"  Portfolio cumulative: {port_cum:.2%}")
        print(f"  Benchmark cumulative: {bench_cum:.2%}")
        print(f"  Excess cumulative:    {port_cum - bench_cum:.2%}")
        print(f"  Portfolio annualized: {port_ann:.2%}")
        print(f"  Benchmark annualized: {bench_ann:.2%}")
        print(f"  Sharpe ratio:         {sharpe:.2f}")
        print(f"  Max drawdown:         {max_dd:.2%}")
        print(f"  Win rate vs bench:    {win_rate:.1%}")
        print(f"  Calmar ratio:         {calmar:.2f}")
        print("-" * 60)

        # Save detailed results
        output_dir = Path(self.config.get("paths", {}).get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"回测结果_{start_date}_{end_date}.csv"
        df_periods.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Backtest detail saved to: {output_path}")

        return df_periods
