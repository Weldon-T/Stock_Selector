#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Selection Tool - Multi-factor scoring engine for A-share market.

Usage:
    python main.py                                # Today's selection
    python main.py --date 2024-09-30              # Historical selection
    python main.py --date 2025-12-31 --multi-quarter  # 3yr weighted
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
        help="Run 12-quarter (3yr) weighted scoring",
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
    return parser.parse_args()


def load_config(path: str) -> dict:
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

    return config


def _quarter_end_dates(end_date: str, years_back: int = 3):
    """Generate quarter-end dates (12 dates) going back from end_date."""
    end_dt = pd.to_datetime(end_date, format="%Y%m%d")
    dates = []
    for y in range(years_back):
        year = end_dt.year - y
        for month in (12, 9, 6, 3):
            # Use last calendar day of quarter month
            day = 31 if month in (12, 3) else 30
            dt_str = f"{year}{month:02d}{day}"
            # Don't go past end_date
            if dt_str <= end_date:
                dates.append(dt_str)
    return sorted(dates)


def _resolve_trade_date(client, loader, date_str: str, max_back: int = 5) -> str | None:
    """Try to find a trade date with data, stepping back up to max_back days."""
    dt = pd.to_datetime(date_str, format="%Y%m%d")
    for i in range(max_back):
        d = (dt - pd.Timedelta(days=i)).strftime("%Y%m%d")
        df = loader.load_daily_all(d)
        if not df.empty:
            return d
    return None


def _print_top10(df: pd.DataFrame, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"  Top 10 Stocks — {label}")
    print("=" * 70)
    display_cols = ["ts_code", "name", "industry", "final_score"]
    available = [c for c in display_cols if c in df.columns]
    top10 = df[available].head(10)
    print(top10.to_string(index=False))
    print("-" * 70)
    print(f"Total selected: {len(df)} stocks\n")


def run_single_date(config, client, loader, stock_filter, factor_calc, scorer,
                    trade_date: str) -> pd.DataFrame | None:
    """Run pipeline for a single date. Returns scored DataFrame (all stocks) or None."""
    logger = get_logger()
    logger.info(f"--- {format_date(trade_date)} ---")

    df_basic = loader.load_stock_basic(trade_date)
    if df_basic.empty:
        logger.warning(f"No stock_basic for {trade_date}, skipping")
        return None

    df_daily = loader.load_daily_all(trade_date)
    if df_daily.empty:
        logger.warning(f"No daily data for {trade_date}, skipping")
        return None

    df_multi_daily = loader.load_daily_multi(trade_date, lookback=6)

    df_filtered = stock_filter.apply(df_basic, df_daily, trade_date)
    if df_filtered.empty:
        logger.warning(f"No stocks passed filter on {trade_date}")
        return None

    df_factors = factor_calc.calculate(df_filtered, df_daily, trade_date, df_multi_daily)
    return scorer.score_all(df_factors)


def run_pipeline(config: dict, trade_date: str, output_dir: Path) -> None:
    """Single-date pipeline with CSV output."""
    logger = get_logger()
    logger.info(f"=== Stock Selection Pipeline: {format_date(trade_date)} ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = SQLiteCache(config.get("paths", {}).get("cache_dir", "./cache"))
    client = TushareClient(config["tushare_token"])
    if not client.login():
        logger.error("Tushare login failed. Aborting.")
        sys.exit(1)

    loader = DataLoader(client, cache)
    stock_filter = StockFilter(config)
    factor_calc = FactorCalculator(config)
    scorer = StockScorer(config)

    df_result = run_single_date(config, client, loader, stock_filter, factor_calc,
                                scorer, trade_date)
    if df_result is None or df_result.empty:
        logger.warning("No results to save. Exiting.")
        return

    output_path = output_dir / f"选股结果_{trade_date}.csv"
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {output_path}")
    _print_top10(df_result, format_date(trade_date))


def run_multi_quarter(config: dict, end_date: str, output_dir: Path) -> None:
    """Multi-quarter weighted scoring pipeline."""
    logger = get_logger()
    logger.info(f"=== Multi-Quarter Pipeline: 3yr back from {format_date(end_date)} ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = SQLiteCache(config.get("paths", {}).get("cache_dir", "./cache"))
    client = TushareClient(config["tushare_token"])
    if not client.login():
        logger.error("Tushare login failed. Aborting.")
        sys.exit(1)

    loader = DataLoader(client, cache)
    stock_filter = StockFilter(config)
    factor_calc = FactorCalculator(config)
    scorer = StockScorer(config)

    # Generate 12 quarter-end dates
    quarter_dates = _quarter_end_dates(end_date, years_back=3)
    logger.info(f"Quarter dates to process: {[format_date(d) for d in quarter_dates]}")

    # Resolve each to an actual trading date
    resolved_dates = []
    for d in quarter_dates:
        resolved = _resolve_trade_date(client, loader, d)
        if resolved:
            resolved_dates.append(resolved)
            logger.info(f"  {format_date(d)} -> {format_date(resolved)}")
        else:
            logger.warning(f"  {format_date(d)} -> no data within 5 days, skipping")

    if len(resolved_dates) < 4:
        logger.error(f"Only {len(resolved_dates)} valid dates found, need at least 4. Aborting.")
        sys.exit(1)

    # Year-bucket weights: most-recent year = 0.5, year-2 = 0.3, year-3 = 0.2
    year_weights = {0: 0.5, 1: 0.3, 2: 0.2}  # offset from end_date year

    # Collect per-date scores
    date_scores: list[dict] = []  # [{trade_date, year_offset, df}]
    end_year = int(end_date[:4])

    for d in resolved_dates:
        df = run_single_date(config, client, loader, stock_filter, factor_calc,
                             scorer, d)
        if df is not None and not df.empty:
            year_offset = end_year - int(d[:4])
            w = year_weights.get(year_offset, 0.1)
            date_scores.append({
                "trade_date": d,
                "year_offset": year_offset,
                "weight": w / 4,  # 4 quarters per year
                "df": df,
            })
            logger.info(f"  {format_date(d)}: {len(df)} stocks, weight={w/4:.4f}")

    if not date_scores:
        logger.error("No valid quarterly results. Aborting.")
        sys.exit(1)

    # Aggregate: weighted average of percentile ranks per ts_code
    rank_cols = [f"{f['name']}_rank" for f in scorer.factors]
    rank_cols = [c for c in rank_cols if c in date_scores[0]["df"].columns]

    # Collect all ts_codes that appear in at least half the periods
    min_periods = max(1, len(date_scores) // 2)
    ts_counter: dict[str, int] = defaultdict(int)
    for ds in date_scores:
        for code in ds["df"]["ts_code"]:
            ts_counter[code] += 1
    valid_codes = {c for c, n in ts_counter.items() if n >= min_periods}
    logger.info(f"{len(valid_codes)} stocks appear in >= {min_periods} periods")

    # Build weighted rank aggregation
    rank_sums: dict[str, dict] = {c: {rc: 0.0 for rc in rank_cols} for c in valid_codes}
    weight_sums: dict[str, float] = {c: 0.0 for c in valid_codes}

    for ds in date_scores:
        w = ds["weight"]
        df = ds["df"]
        for _, row in df.iterrows():
            code = row["ts_code"]
            if code not in valid_codes:
                continue
            for rc in rank_cols:
                val = row.get(rc, np.nan)
                if not np.isnan(val):
                    rank_sums[code][rc] += val * w
            weight_sums[code] += w

    # Compute weighted average ranks → final_score
    records = []
    for code in valid_codes:
        if weight_sums[code] == 0:
            continue
        avg_ranks = {rc: rank_sums[code][rc] / weight_sums[code] for rc in rank_cols}
        final_score = sum(avg_ranks.values()) / len(rank_cols) if rank_cols else 0.0

        # Pick name/industry from latest period
        name, industry = "", ""
        for ds in reversed(date_scores):
            row = ds["df"][ds["df"]["ts_code"] == code]
            if not row.empty:
                name = row.iloc[0].get("name", "")
                industry = row.iloc[0].get("industry", "")
                break

        records.append({
            "ts_code": code,
            "name": name,
            "industry": industry,
            "final_score": final_score,
            "periods_present": ts_counter[code],
            **avg_ranks,
        })

    df_result = pd.DataFrame(records)
    df_result = df_result.sort_values("final_score", ascending=False)
    df_result = df_result.head(config.get("select_count", 50))
    df_result = df_result.reset_index(drop=True)

    # Output
    output_path = output_dir / f"选股结果_多季度_{end_date}.csv"
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {output_path}")

    _print_top10(df_result, f"Multi-Quarter → {format_date(end_date)}")


def main():
    load_dotenv()

    args = parse_args()
    config = load_config(args.config)

    log_cfg = config.get("logging", {})
    log_dir = config.get("paths", {}).get("log_dir", "./logs")
    setup_logger(
        log_dir=log_dir,
        log_file=log_cfg.get("file", "run.log"),
        level=log_cfg.get("level", "INFO"),
    )
    logger = get_logger()
    logger.info("Stock Selection Tool started")

    paths = config.get("paths", {})
    output_dir = Path(paths.get("output_dir", "./output"))

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
            run_multi_quarter(config, trade_date, output_dir)
        else:
            run_pipeline(config, trade_date, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
