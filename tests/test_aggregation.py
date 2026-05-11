"""
Verify that multi-quarter aggregation correctly separates fundamental and technical factors.
"""
import numpy as np
import pandas as pd
import pytest
from core.stock_scorer import StockScorer


TECH_FACTORS = {"short_reversal", "short_momentum", "volatility", "amplitude", "amount_stability", "volume_ratio"}


def simulate_aggregation(quarter_dfs, factor_names, quarter_weights=None):
    """
    Simplified version of the multi-quarter aggregation logic from main.py/backtest.py.
    Returns avg_ranks and avg_values for a single stock.
    """
    if quarter_weights is None:
        quarter_weights = [0.2, 0.2, 0.3, 0.3]

    config = {
        "factors": {
            "fake": {fn: {"weight": 0.1, "enabled": True, "direction": "positive"}
                      for fn in factor_names}
        }
    }
    scorer = StockScorer(config)
    rank_cols = [f"{fn}_rank" for fn in factor_names]

    # Simulate aggregation
    rank_sums = {rc: 0.0 for rc in rank_cols}
    value_sums = {fn: 0.0 for fn in factor_names}
    weight_sum = 0.0

    for q_df, w in zip(quarter_dfs, quarter_weights):
        scored = scorer.score_all(q_df, sector_neutral=True)
        row = scored.iloc[0]  # single stock
        for rc in rank_cols:
            val = row.get(rc, np.nan)
            if not np.isnan(val):
                rank_sums[rc] += val * w
        for fn in factor_names:
            val = row.get(fn, np.nan)
            if not np.isnan(val):
                value_sums[fn] += val * w
        weight_sum += w

    # Overwrite technical factors with latest quarter
    latest_scored = scorer.score_all(quarter_dfs[-1], sector_neutral=True)
    latest_row = latest_scored.iloc[0]
    for fn in TECH_FACTORS & set(factor_names):
        if fn in latest_scored.columns:
            val = latest_row[fn]
            if not np.isnan(val):
                value_sums[fn] = val
        rc = f"{fn}_rank"
        if rc in latest_scored.columns:
            val = latest_row[rc]
            if not np.isnan(val):
                rank_sums[rc] = val

    avg_ranks = {rc: rank_sums[rc] / weight_sum for rc in rank_cols}
    avg_values = {fn: value_sums[fn] / weight_sum for fn in factor_names}
    return avg_ranks, avg_values


def make_quarter_df(factor_name, values_by_quarter):
    """Create quarter DataFrames with one stock and varying factor values."""
    dfs = []
    for q_val in values_by_quarter:
        dfs.append(pd.DataFrame({
            "ts_code": ["A.SH"],
            "name": ["StockA"],
            "industry": ["Tech"],
            "market": ["主板"],
            factor_name: [q_val],
        }))
    return dfs


class TestMultiQuarterAggregation:
    """Verify fundamental factors are averaged, technical factors use latest quarter."""

    def test_fundamental_averaged(self):
        """ep_ttm values [0.10, 0.20, 0.30, 0.40] with weights [0.2,0.2,0.3,0.3]
        → weighted avg = 0.10*0.2 + 0.20*0.2 + 0.30*0.3 + 0.40*0.3 = 0.27"""
        dfs = make_quarter_df("ep_ttm", [0.10, 0.20, 0.30, 0.40])
        _, values = simulate_aggregation(dfs, ["ep_ttm"])
        assert values["ep_ttm"] == pytest.approx(0.27, abs=0.01)

    def test_technical_is_latest_only(self):
        """short_reversal values [0.05, 0.03, 0.08, 0.12] across quarters
        → should be 0.12 (latest quarter only), NOT the weighted average.
        Weighted avg would be 0.05*0.2+0.03*0.2+0.08*0.3+0.12*0.3 = 0.076"""
        dfs = make_quarter_df("short_reversal", [0.05, 0.03, 0.08, 0.12])
        _, values = simulate_aggregation(dfs, ["short_reversal"])
        # Latest quarter = 0.12, NOT weighted avg = 0.076
        assert values["short_reversal"] == pytest.approx(0.12, abs=0.001)

    def test_mixed_fundamental_and_technical(self):
        """Verify both types coexist correctly in the same aggregation."""
        dfs = []
        for ep_val, rev_val in zip([0.10, 0.20, 0.30, 0.40], [0.05, 0.03, 0.08, 0.12]):
            dfs.append(pd.DataFrame({
                "ts_code": ["A.SH"],
                "name": ["StockA"],
                "industry": ["Tech"],
                "market": ["主板"],
                "ep_ttm": [ep_val],
                "short_reversal": [rev_val],
            }))

        _, values = simulate_aggregation(dfs, ["ep_ttm", "short_reversal"])

        # Fundamental: weighted average
        assert values["ep_ttm"] == pytest.approx(0.27, abs=0.01)
        # Technical: latest quarter only
        assert values["short_reversal"] == pytest.approx(0.12, abs=0.001)

    def test_volatility_is_technical(self):
        """volatility is in TECH_FACTORS and should use latest quarter only."""
        dfs = make_quarter_df("volatility", [0.40, 0.35, 0.30, 0.25])
        _, values = simulate_aggregation(dfs, ["volatility"])
        assert values["volatility"] == pytest.approx(0.25, abs=0.001)

    def test_volume_ratio_is_technical(self):
        dfs = make_quarter_df("volume_ratio", [1.5, 1.2, 0.9, 1.1])
        _, values = simulate_aggregation(dfs, ["volume_ratio"])
        assert values["volume_ratio"] == pytest.approx(1.1, abs=0.001)
