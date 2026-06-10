#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Selection Tool - Multi-factor scoring engine for A-share market.

Usage:
    python main.py                                # Today's selection
    python main.py --date 2024-09-30              # Historical selection
    python main.py --date 2026-05-07 --multi-quarter  # 4-quarter weighted, sector-neutral
    python main.py --backtest --start 2024-01-01 --end 2024-12-31  # Backtest
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from core.tushare_client import TushareClient
from core.data_loader import DataLoader
from core.filter import StockFilter
from core.factor_calculator import FactorCalculator
from core.stock_scorer import StockScorer
from core.backtest import Backtest
from utils.logger import setup_logger, get_logger
from utils.date_utils import parse_date, format_date, get_latest_trade_date
from utils.cache import SQLiteCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A-share Multi-factor Stock Selection Tool"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Trade date in YYYY-MM-DD format (default: latest trade day)",
    )
    parser.add_argument(
        "--multi-quarter", action="store_true",
        help="Run 4-quarter (1yr) weighted scoring with sector-neutral ranking",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run in backtest mode",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Strategy profile: value or smallcap (default: from config.yaml)",
    )
    return parser.parse_args()


def load_config(path: str, strategy: str | None = None) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        print(f"FATAL: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    token = os.getenv("TUSHARE_TOKEN") or config.get("tushare_token", "")
    if not token or token == "your_token_here":
        print("FATAL: Tushare token not configured.")
        print("Please set TUSHARE_TOKEN in .env file")
        print("Get a free token at: https://tushare.pro")
        sys.exit(1)
    config["tushare_token"] = token

    _resolve_strategy(config, strategy)
    return config


def _resolve_strategy(config: dict, strategy: str | None) -> None:
    """Lift a strategy profile's keys to config top-level.

    Strategy-specific keys (factors, stock_pool, select_count, hold_months)
    are promoted so downstream code reads them from their usual paths.
    """
    strategies = config.get("strategies", {})
    if not strategies:
        return

    name = strategy or config.get("strategy", "value")
    if name not in strategies:
        print(f"FATAL: Unknown strategy '{name}'. Available: {list(strategies)}")
        sys.exit(1)

    profile = strategies[name]
    for key in ("factors", "stock_pool", "select_count"):
        if key in profile:
            config[key] = profile[key]
    if "hold_months" in profile:
        config.setdefault("backtest", {})
        config["backtest"]["hold_months"] = profile["hold_months"]


# ============================================================================
# Single-date pipeline
# ============================================================================

def _print_top10(df: pd.DataFrame, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"  Top 10 Stocks — {label}")
    print("=" * 70)
    display_cols = ["ts_code", "name", "industry", "market", "final_score"]
    available = [c for c in display_cols if c in df.columns]
    top10 = df[available].head(10)
    print(top10.to_string(index=False))
    print("-" * 70)
    print(f"Total selected: {len(df)} stocks\n")


def run_pipeline(config: dict, trade_date: str) -> None:
    logger = get_logger()
    output_dir = Path(config.get("paths", {}).get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Stock Selection Pipeline: {format_date(trade_date)} ===")

    cache = SQLiteCache(config.get("paths", {}).get("cache_dir", "./cache"))
    client = TushareClient(config["tushare_token"])
    if not client.login():
        logger.error("Tushare login failed. Aborting.")
        sys.exit(1)

    loader = DataLoader(client, cache)
    stock_filter = StockFilter(config)
    factor_calc = FactorCalculator(config)
    scorer = StockScorer(config)

    df_basic = loader.load_stock_basic(trade_date)
    if df_basic.empty:
        logger.error("Failed to load stock_basic. Aborting.")
        sys.exit(1)

    df_daily = loader.load_daily_all(trade_date)
    if df_daily.empty:
        logger.error(f"No daily data for {trade_date}. Aborting.")
        sys.exit(1)

    df_multi_daily = loader.load_daily_multi(trade_date, lookback=config.get("multi_daily_lookback", 10))
    df_moneyflow = loader.load_moneyflow_multi(trade_date, lookback=5)
    df_filtered = stock_filter.apply(df_basic, df_daily, trade_date)
    if df_filtered.empty:
        logger.warning("No stocks passed the filter. Exiting.")
        return

    df_factors = factor_calc.calculate(df_filtered, df_daily, trade_date, df_multi_daily, df_moneyflow)
    df_scored = scorer.score_all(df_factors, sector_neutral=True)

    # Per-market selection
    select_count = config.get("select_count", {"主板": 50, "创业板": 20, "科创板": 20})
    parts = []
    for market, count in select_count.items():
        subset = df_scored[df_scored["market"] == market].head(count)
        parts.append(subset)
        logger.info(f"  {market}: selected {len(subset)}/{count}")
    df_result = pd.concat(parts, ignore_index=True)
    df_result = df_result.sort_values("final_score", ascending=False).reset_index(drop=True)

    output_path = output_dir / f"选股结果_{trade_date}.csv"
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {output_path} ({len(df_result)} stocks)")
    _print_top10(df_result, format_date(trade_date))


# ============================================================================
# Multi-quarter pipeline
# ============================================================================


def _quarter_end_dates(end_date: str):
    """Generate the 4 most recent quarter-end dates (YYYYMMDD)."""
    end_dt = pd.to_datetime(end_date, format="%Y%m%d")
    dates = []
    # Walk back through months to find 4 completed quarters
    y, m = end_dt.year, end_dt.month
    for _ in range(6):  # search up to 6 quarters back
        # Go to previous quarter end
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
        m -= 1  # step back into previous quarter
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


def _run_quarter_date(loader, stock_filter, factor_calc, d: str,
                     lookback: int = 10) -> pd.DataFrame | None:
    """Run single-date pipeline for one quarter and return factor DataFrame."""
    df_basic = loader.load_stock_basic(d)
    if df_basic.empty:
        return None

    df_daily = loader.load_daily_all(d)
    if df_daily.empty:
        return None

    df_multi_daily = loader.load_daily_multi(d, lookback=lookback)
    df_moneyflow = loader.load_moneyflow_multi(d, lookback=5)
    df_filtered = stock_filter.apply(df_basic, df_daily, d)
    if df_filtered.empty:
        return None

    return factor_calc.calculate(df_filtered, df_daily, d, df_multi_daily, df_moneyflow)


def run_multi_quarter(config: dict, end_date: str) -> None:
    """4-quarter weighted scoring with sector-neutral ranking, per-market output."""
    logger = get_logger()
    output_dir = Path(config.get("paths", {}).get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Multi-Quarter Pipeline: 1yr back from {format_date(end_date)} ===")

    cache = SQLiteCache(config.get("paths", {}).get("cache_dir", "./cache"))
    client = TushareClient(config["tushare_token"])
    if not client.login():
        logger.error("Tushare login failed. Aborting.")
        sys.exit(1)

    loader = DataLoader(client, cache)
    stock_filter = StockFilter(config)
    factor_calc = FactorCalculator(config)
    scorer = StockScorer(config)

    # 1. Generate quarter dates and resolve to trade dates
    quarter_targets = _quarter_end_dates(end_date)
    logger.info(f"Quarter targets: {[format_date(d) for d in quarter_targets]}")

    quarter_weights = [0.2, 0.2, 0.3, 0.3]  # Q1..Q4, newer = heavier

    quarter_results: list[dict] = []
    for qd, w in zip(quarter_targets, quarter_weights):
        resolved = _resolve_date(loader, qd)
        if not resolved:
            logger.warning(f"  {format_date(qd)}: no trading date found, skipping")
            continue

        logger.info(f"  {format_date(qd)} -> {format_date(resolved)} (weight={w})")
        df = _run_quarter_date(loader, stock_filter, factor_calc, resolved,
                              lookback=config.get("multi_daily_lookback", 10))
        if df is not None and len(df) > 100:
            quarter_results.append({"quarter": resolved, "weight": w, "df": df})
            logger.info(f"    {len(df)} stocks")
        else:
            logger.warning(f"    failed or only {len(df) if df is not None else 0} stocks")

    if len(quarter_results) < 3:
        logger.error(f"Only {len(quarter_results)} valid quarters, need >= 3. Aborting.")
        sys.exit(1)

    # 2. Aggregate: weighted average of ranks + raw values across quarters
    min_periods = max(1, len(quarter_results) // 2)
    ts_counter: dict[str, int] = defaultdict(int)
    for qr in quarter_results:
        for code in qr["df"]["ts_code"]:
            ts_counter[code] += 1
    valid_codes = {c for c, n in ts_counter.items() if n >= min_periods}
    logger.info(f"{len(valid_codes)} stocks appear in >= {min_periods} quarters")

    factor_names = [f["name"] for f in scorer.factors]
    rank_cols = [f"{fn}_rank" for fn in factor_names]

    rank_sums: dict[str, dict] = {c: {rc: 0.0 for rc in rank_cols} for c in valid_codes}
    value_sums: dict[str, dict] = {c: {fn: 0.0 for fn in factor_names} for c in valid_codes}
    weight_sums: dict[str, float] = {c: 0.0 for c in valid_codes}

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

    # 3. Build final output
    records = []
    for code in valid_codes:
        if weight_sums[code] == 0:
            continue
        avg_ranks = {rc: rank_sums[code][rc] / weight_sums[code] for rc in rank_cols}
        avg_values = {fn: value_sums[code][fn] / weight_sums[code] for fn in factor_names}
        final_score = sum(avg_ranks.values()) / len(rank_cols) if rank_cols else 0.0

        name, industry, market = "", "", ""
        for qr in reversed(quarter_results):
            row = qr["df"][qr["df"]["ts_code"] == code]
            if not row.empty:
                name = row.iloc[0].get("name", "")
                industry = row.iloc[0].get("industry", "")
                market = row.iloc[0].get("market", "")
                break

        records.append({
            "ts_code": code, "name": name, "industry": industry, "market": market,
            "final_score": final_score, "quarters_present": ts_counter[code],
            **avg_values, **avg_ranks,
        })

    df_all = pd.DataFrame(records)
    id_cols = ["ts_code", "name", "industry", "market", "final_score", "quarters_present"]
    ordered = id_cols + factor_names + rank_cols
    available = [c for c in ordered if c in df_all.columns]
    df_all = df_all[available].sort_values("final_score", ascending=False)

    # 4. Per-market selection
    select_count = config.get("select_count", {"主板": 50, "创业板": 20, "科创板": 20})
    all_parts = []
    for market, count in select_count.items():
        subset = df_all[df_all["market"] == market].head(count)
        all_parts.append(subset)
        logger.info(f"  {market}: selected {len(subset)}/{count}")

    df_result = pd.concat(all_parts, ignore_index=True)
    df_result = df_result.sort_values("final_score", ascending=False).reset_index(drop=True)

    # 5. Output
    output_path = output_dir / f"选股结果_多季度_{end_date}.csv"
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {output_path} ({len(df_result)} stocks)")

    _print_top10(df_result, f"Multi-Quarter → {format_date(end_date)} (sector-neutral)")

    print("  Per-market breakdown:")
    for m in ["主板", "创业板", "科创板"]:
        cnt = len(df_result[df_result["market"] == m])
        print(f"    {m}: {cnt} stocks")


# ============================================================================
# Main
# ============================================================================

def main():
    load_dotenv()

    args = parse_args()
    config = load_config(args.config, args.strategy)

    log_cfg = config.get("logging", {})
    log_dir = config.get("paths", {}).get("log_dir", "./logs")
    setup_logger(
        log_dir=log_dir,
        log_file=log_cfg.get("file", "run.log"),
        level=log_cfg.get("level", "INFO"),
    )
    logger = get_logger()
    logger.info("Stock Selection Tool started")

    if args.backtest:
        if not args.start or not args.end:
            logger.error("Backtest mode requires --start and --end dates")
            sys.exit(1)
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
        bt = Backtest(config)
        bt.run(start_date, end_date)
    else:
        trade_date = parse_date(args.date)
        if args.multi_quarter:
            run_multi_quarter(config, trade_date)
        else:
            run_pipeline(config, trade_date)

    logger.info("Done.")


if __name__ == "__main__":
    main()
