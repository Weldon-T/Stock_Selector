import numpy as np
import pandas as pd
import pytest
from core.factor_calculator import FactorCalculator


@pytest.fixture
def calc():
    return FactorCalculator({"factors": {}})


def make_multi_daily(codes, closes_list, highs=None, lows=None, amounts=None):
    """Build a multi_daily DataFrame with 10 rows per stock."""
    records = []
    for i, code in enumerate(codes):
        closes = closes_list[i]
        for j, close in enumerate(closes):
            rec = {
                "ts_code": code,
                "trade_date": f"2026010{j+1}",
                "close": float(close),
            }
            if highs:
                rec["high"] = float(highs[i][j])
            if lows:
                rec["low"] = float(lows[i][j])
            if amounts:
                rec["amount"] = float(amounts[i][j])
            records.append(rec)
    return pd.DataFrame(records)


@pytest.fixture
def stock_df():
    return pd.DataFrame({"ts_code": ["A.SH", "B.SZ", "C.SH"]})


# ============================================================================
# short_reversal
# ============================================================================

class TestShortReversal:
    def test_positive_reversal(self, calc, stock_df):
        """Stock that dropped → reversal positive. Stock that rose → reversal negative."""
        md = make_multi_daily(
            ["A.SH", "B.SZ"],
            [
                # A: 5 closes [10, 9.5, 9, 8.5, 8] — dropped 20%, reversal = +0.20
                [10, 9.5, 9, 8.5, 8],
                # B: 5 closes [10, 10.5, 11, 11.5, 12] — rose 20%, reversal = -0.20
                [10, 10.5, 11, 11.5, 12],
            ],
        )
        result = calc._compute_short_reversal(stock_df, md)

        # A: (8-10)/10 = -0.20, reversal = +0.20
        assert result.loc[0] == pytest.approx(0.20, abs=0.005)
        # B: (12-10)/10 = +0.20, reversal = -0.20
        assert result.loc[1] == pytest.approx(-0.20, abs=0.005)

    def test_insufficient_data(self, calc, stock_df):
        """Stocks with < 5 days should get NaN."""
        md = make_multi_daily(
            ["A.SH"],
            [[10, 10.5]],  # only 2 days
        )
        result = calc._compute_short_reversal(stock_df, md)
        assert np.isnan(result.loc[0])

    def test_empty_multi_daily(self, calc, stock_df):
        result = calc._compute_short_reversal(stock_df, pd.DataFrame())
        assert result.isna().all()

    def test_no_close_column(self, calc, stock_df):
        md = pd.DataFrame({"ts_code": ["A.SH"], "trade_date": ["20260101"], "bad_col": [1.0]})
        result = calc._compute_short_reversal(stock_df, md)
        assert result.isna().all()


# ============================================================================
# amplitude
# ============================================================================

class TestAmplitude:
    def test_basic_amplitude(self, calc, stock_df):
        """Amplitude = mean((high - low) / close) over window."""
        md = make_multi_daily(
            ["A.SH", "B.SZ", "C.SH"],
            closes_list=[[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]] * 3,
            highs=[[11, 11, 11, 11, 11, 11, 11, 11, 11, 11]] * 3,
            lows=[[9, 9, 9, 9, 9, 9, 9, 9, 9, 9]] * 3,
        )
        result = calc._compute_amplitude(stock_df, md)
        # (11-9)/10 = 0.2 for all days, mean = 0.2
        assert result.loc[0] == pytest.approx(0.2, abs=0.001)
        assert result.loc[1] == pytest.approx(0.2, abs=0.001)

    def test_varying_amplitude(self, calc, stock_df):
        md = make_multi_daily(
            ["A.SH"],
            closes_list=[[10, 10, 10]],
            highs=[[11, 12, 10.5]],
            lows=[[9, 9, 9.5]],
        )
        result = calc._compute_amplitude(stock_df, md)
        # (11-9)/10=0.2, (12-9)/10=0.3, (10.5-9.5)/10=0.1 → mean=0.2
        assert result.loc[0] == pytest.approx(0.2, abs=0.001)

    def test_empty_multi_daily(self, calc, stock_df):
        result = calc._compute_amplitude(stock_df, pd.DataFrame())
        assert result.isna().all()

    def test_missing_hl_columns(self, calc, stock_df):
        md = pd.DataFrame({"ts_code": ["A.SH"], "close": [10.0]})
        result = calc._compute_amplitude(stock_df, md)
        assert result.isna().all()


# ============================================================================
# amount_stability
# ============================================================================

class TestAmountStability:
    def test_perfectly_stable(self, calc, stock_df):
        """Same amount every day → CV = 0."""
        md = make_multi_daily(
            ["A.SH"],
            closes_list=[[10] * 5],
            amounts=[[1000] * 5],
        )
        result = calc._compute_amount_stability(stock_df, md)
        assert result.loc[0] == pytest.approx(0.0, abs=0.001)

    def test_variable_amount(self, calc, stock_df):
        """Amount with known mean and std."""
        md = make_multi_daily(
            ["A.SH"],
            closes_list=[[10] * 4],
            amounts=[[100, 200, 300, 200]],
        )
        result = calc._compute_amount_stability(stock_df, md)
        # mean=200, std≈81.65, CV≈0.408
        assert result.loc[0] == pytest.approx(0.408, abs=0.01)

    def test_empty_multi_daily(self, calc, stock_df):
        result = calc._compute_amount_stability(stock_df, pd.DataFrame())
        assert result.isna().all()

    def test_no_amount_column(self, calc, stock_df):
        md = pd.DataFrame({"ts_code": ["A.SH"], "close": [10.0]})
        result = calc._compute_amount_stability(stock_df, md)
        assert result.isna().all()
