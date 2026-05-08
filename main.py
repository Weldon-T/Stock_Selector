#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stock Selection Tool - Multi-factor scoring engine for A-share market.

Usage:
    python main.py                           # Today's selection
    python main.py --date 2024-09-30         # Historical selection
    python main.py --backtest --start 2024-01-01 --end 2024-12-31  # Backtest
"""

import argparse
import os
import sys
from pathlib import Path

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


def run_pipeline(config: dict, trade_date: str) -> None:
    logger = get_logger()
    logger.info(f"=== Stock Selection Pipeline: {format_date(trade_date)} ===")

    paths = config.get("paths", {})
    cache_dir = paths.get("cache_dir", "./cache")
    output_dir = paths.get("output_dir", "./output")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize components
    cache = SQLiteCache(cache_dir)
    client = TushareClient(config["tushare_token"])
    if not client.login():
        logger.error("Tushare login failed. Aborting.")
        sys.exit(1)

    loader = DataLoader(client, cache)
    stock_filter = StockFilter(config)
    factor_calc = FactorCalculator(config)
    scorer = StockScorer(config)

    # Step 1: Load data
    df_basic = loader.load_stock_basic(trade_date)
    if df_basic.empty:
        logger.error("Failed to load stock_basic. Aborting.")
        sys.exit(1)

    df_daily = loader.load_daily_all(trade_date)
    if df_daily.empty:
        logger.error(f"No daily data for {trade_date}. Aborting.")
        sys.exit(1)

    # Step 2: Filter stock pool
    df_filtered = stock_filter.apply(df_basic, df_daily, trade_date)
    if df_filtered.empty:
        logger.warning("No stocks passed the filter. Exiting.")
        return

    # Step 3: Calculate factors
    df_factors = factor_calc.calculate(df_filtered, df_daily, trade_date)

    # Step 4: Score and rank
    df_result = scorer.score(df_factors)

    # Step 5: Output CSV
    output_path = Path(output_dir) / f"选股结果_{trade_date}.csv"
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {output_path}")

    # Step 6: Print top 10 summary
    print("\n" + "=" * 70)
    print(f"  Top 10 Stocks — {format_date(trade_date)}")
    print("=" * 70)
    display_cols = ["ts_code", "name", "industry", "final_score"]
    available = [c for c in display_cols if c in df_result.columns]
    top10 = df_result[available].head(10)
    print(top10.to_string(index=False))
    print("-" * 70)
    print(f"Total selected: {len(df_result)} stocks\n")


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
        run_pipeline(config, trade_date)

    logger.info("Done.")


if __name__ == "__main__":
    main()
