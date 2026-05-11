import numpy as np
import pandas as pd
import pytest
from core.stock_scorer import StockScorer


@pytest.fixture
def config():
    return {
        "factors": {
            "value": {
                "ep_ttm": {"weight": 0.06, "enabled": True, "direction": "positive"},
                "bp": {"weight": 0.04, "enabled": True, "direction": "positive"},
            },
            "quality": {
                "roe_ttm": {"weight": 0.10, "enabled": False, "direction": "positive"},  # disabled
            },
            "momentum": {
                "short_reversal": {"weight": 0.15, "enabled": True, "direction": "positive"},
                "volatility": {"weight": 0.12, "enabled": True, "direction": "negative"},
            },
        },
    }


class TestFactorConfig:
    def test_disabled_excluded(self, config):
        scorer = StockScorer(config)
        names = {f["name"] for f in scorer.factors}
        assert "roe_ttm" not in names
        assert "ep_ttm" in names
        assert "short_reversal" in names

    def test_weight_normalization(self, config):
        scorer = StockScorer(config)
        weights = sum(f["weight"] for f in scorer.factors)
        assert weights == pytest.approx(1.0, abs=0.001)

    def test_weight_proportional(self, config):
        scorer = StockScorer(config)
        ep = next(f for f in scorer.factors if f["name"] == "ep_ttm")
        bp = next(f for f in scorer.factors if f["name"] == "bp")
        # 0.06 vs 0.04 raw → normalized should maintain 3:2 ratio
        assert ep["weight"] / bp["weight"] == pytest.approx(1.5, abs=0.01)

    def test_direction_recorded(self, config):
        scorer = StockScorer(config)
        vol = next(f for f in scorer.factors if f["name"] == "volatility")
        assert vol["direction"] == "negative"

        rev = next(f for f in scorer.factors if f["name"] == "short_reversal")
        assert rev["direction"] == "positive"


class TestScoring:
    @pytest.fixture
    def scorer(self, config):
        return StockScorer(config)

    @pytest.fixture
    def sample_df(self):
        """10 stocks with known factor values, split across 2 markets for sector-neutral testing."""
        np.random.seed(42)
        return pd.DataFrame({
            "ts_code": [f"{i:06d}.SH" for i in range(10)],
            "name": [f"Stock{i}" for i in range(10)],
            "industry": ["A"] * 5 + ["B"] * 5,
            "market": ["主板"] * 5 + ["创业板"] * 5,
            "ep_ttm": [0.02, 0.05, 0.10, 0.03, 0.08, 0.01, 0.12, np.nan, 0.04, 0.06],
            "bp": [0.5, 0.3, 0.8, 0.4, 0.2, 0.6, 0.1, 0.7, 0.9, 0.3],
            "short_reversal": [0.05, -0.02, 0.10, 0.0, -0.05, 0.08, -0.10, 0.03, 0.01, 0.15],
            "volatility": [0.30, 0.25, 0.40, 0.20, 0.35, 0.15, 0.50, 0.28, 0.22, 0.45],
        })

    def test_score_all_returns_all(self, scorer, sample_df):
        result = scorer.score_all(sample_df)
        assert len(result) == 10
        assert "final_score" in result.columns

    def test_final_score_in_range(self, scorer, sample_df):
        result = scorer.score_all(sample_df)
        # final_score is weighted average of 0-1 ranks, so range is [0, 1]
        assert result["final_score"].between(0, 1).all()

    def test_negative_direction_inverts_rank(self, scorer, sample_df):
        """High volatility should get low rank (inverted)."""
        result = scorer.score_all(sample_df)
        # Highest volatility (0.50) should have the lowest volatility_rank
        max_vol_idx = sample_df["volatility"].idxmax()
        max_vol_rank = result.loc[max_vol_idx, "volatility_rank"]
        min_vol_idx = sample_df["volatility"].idxmin()
        min_vol_rank = result.loc[min_vol_idx, "volatility_rank"]
        assert max_vol_rank < min_vol_rank  # higher vol → lower rank (negative direction)

    def test_nan_handled(self, scorer, sample_df):
        """NaN factor values should not crash; rank filled with 0.5."""
        result = scorer.score_all(sample_df)
        # ep_ttm has one NaN → its rank should be 0.5
        nan_idx = sample_df[sample_df["ep_ttm"].isna()].index[0]
        assert result.loc[nan_idx, "ep_ttm_rank"] == pytest.approx(0.5, abs=0.01)

    def test_sector_neutral_ranking(self, scorer, sample_df):
        """Sector-neutral mode should produce different ranks than cross-market."""
        result_cross = scorer.score_all(sample_df, sector_neutral=False)
        result_sector = scorer.score_all(sample_df, sector_neutral=True)

        # Ranks can differ between modes for stocks in different industries
        # Industry A has 5 stocks, B has 5 stocks — sector-neutral ranks within each
        # cross-market ranks across all 10 — they should differ
        assert not np.allclose(
            result_cross["ep_ttm_rank"].values,
            result_sector["ep_ttm_rank"].values,
            atol=1e-9,
        )
