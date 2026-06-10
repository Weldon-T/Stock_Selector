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


# ============================================================================
# n_day_return (static helper used by momentum factors)
# ============================================================================

class TestNDayReturn:
    def test_basic_10_day_return(self):
        """10-day return from 11 data points."""
        closes = [10] + [10.5] * 10  # 10 → 10.5, return = 0.05
        records = []
        for j, close in enumerate(closes):
            records.append({"ts_code": "A.SH", "trade_date": f"202601{j+1:02d}", "close": float(close)})
        md = pd.DataFrame(records)
        result = FactorCalculator._n_day_return(md, 10)
        assert result.loc["A.SH"] == pytest.approx(0.05, abs=0.001)

    def test_insufficient_falls_back_to_available(self):
        """Only 3 days of data for 10-day request → uses all available."""
        records = [{"ts_code": "A.SH", "trade_date": f"202601{j+1:02d}", "close": float(c)}
                   for j, c in enumerate([10, 11, 12])]  # 3 days, return=0.20
        md = pd.DataFrame(records)
        result = FactorCalculator._n_day_return(md, 10)
        assert result.loc["A.SH"] == pytest.approx(0.20, abs=0.001)

    def test_single_day_is_nan(self):
        records = [{"ts_code": "A.SH", "trade_date": "20260101", "close": 10.0}]
        md = pd.DataFrame(records)
        result = FactorCalculator._n_day_return(md, 10)
        assert np.isnan(result.loc["A.SH"])

    def test_multiple_stocks(self):
        records = []
        for code, closes in [("A.SH", [10, 10.5, 11, 11.5, 12, 12.5]),
                              ("B.SZ", [20, 19, 18, 17, 16, 15])]:
            for j, close in enumerate(closes):
                records.append({"ts_code": code, "trade_date": f"202601{j+1:02d}", "close": float(close)})
        md = pd.DataFrame(records)
        result = FactorCalculator._n_day_return(md, 5)
        # A: (12.5-10)/10 = 0.25, B: (15-20)/20 = -0.25
        assert result.loc["A.SH"] == pytest.approx(0.25, abs=0.001)
        assert result.loc["B.SZ"] == pytest.approx(-0.25, abs=0.001)


# ============================================================================
# sector_momentum
# ============================================================================

class TestSectorMomentum:
    @pytest.fixture
    def stock_df_with_sectors(self):
        return pd.DataFrame({
            "ts_code": ["A.SH", "B.SZ", "C.SH", "D.SZ"],
            "industry": ["Tech", "Tech", "Bank", "Bank"],
        })

    def test_ranks_industries_by_median_return(self, calc, stock_df_with_sectors):
        """Tech stocks up 20% and 10% (median=15%), Bank stocks flat and -10% (median=-5%).
        Tech should rank higher."""
        records = []
        for code, closes in [("A.SH", [10, 12]), ("B.SZ", [10, 11]),
                              ("C.SH", [10, 10]), ("D.SZ", [10, 9])]:
            for j, close in enumerate(closes):
                records.append({"ts_code": code, "trade_date": f"202601{j+1:02d}", "close": float(close)})
        md = pd.DataFrame(records)
        result = calc._compute_sector_momentum(stock_df_with_sectors, md, days=1)

        # Tech industry should rank higher than Bank
        tech_rank = result[stock_df_with_sectors["industry"] == "Tech"].iloc[0]
        bank_rank = result[stock_df_with_sectors["industry"] == "Bank"].iloc[0]
        assert tech_rank > bank_rank

    def test_empty_multi_daily(self, calc, stock_df_with_sectors):
        result = calc._compute_sector_momentum(stock_df_with_sectors, pd.DataFrame(), days=10)
        assert result.isna().all()


# ============================================================================
# price_momentum
# ============================================================================

class TestPriceMomentum:
    @pytest.fixture
    def stock_df(self):
        return pd.DataFrame({"ts_code": ["A.SH", "B.SZ"]})

    def test_computes_n_day_return(self, calc, stock_df):
        records = []
        for code, closes in [("A.SH", [10, 11, 12]), ("B.SZ", [10, 9, 8])]:
            for j, close in enumerate(closes):
                records.append({"ts_code": code, "trade_date": f"202601{j+1:02d}", "close": float(close)})
        md = pd.DataFrame(records)
        result = calc._compute_price_momentum(stock_df, md, days=2)
        # A: (12-10)/10 = 0.20, B: (8-10)/10 = -0.20
        assert result.loc[0] == pytest.approx(0.20, abs=0.001)
        assert result.loc[1] == pytest.approx(-0.20, abs=0.001)

    def test_empty_multi_daily(self, calc, stock_df):
        result = calc._compute_price_momentum(stock_df, pd.DataFrame(), days=10)
        assert result.isna().all()


# ============================================================================
# volume_breakout
# ============================================================================

class TestVolumeBreakout:
    @pytest.fixture
    def stock_df(self):
        return pd.DataFrame({"ts_code": ["A.SH", "B.SZ"]})

    def test_volume_surge(self, calc, stock_df):
        """Stock A: stable vol (~100), last day surges to 300 → ratio ≈ 3.0."""
        records = []
        for code, vols in [("A.SH", [100, 105, 95, 100, 300]),
                            ("B.SZ", [200, 200, 200, 200, 200])]:
            for j, vol in enumerate(vols):
                records.append({"ts_code": code, "trade_date": f"202601{j+1:02d}",
                               "close": 10.0, "vol": float(vol)})
        md = pd.DataFrame(records)
        result = calc._compute_volume_breakout(stock_df, md, days=4)
        # A: avg of last 4 = (105+95+100+300)/4=150, last=300, ratio=2.0
        # But tail(4) from sorted: last 4 would be the last 4 entries after sorting by date
        # Actually avg_vol = mean of last 4 entries in sorted order = mean([105, 95, 100, 300]) = 150
        # ratio = 300/150 = 2.0
        assert result.loc[0] == pytest.approx(2.0, abs=0.01)
        # B: avg=200, last=200, ratio=1.0
        assert result.loc[1] == pytest.approx(1.0, abs=0.01)

    def test_empty_multi_daily(self, calc, stock_df):
        result = calc._compute_volume_breakout(stock_df, pd.DataFrame(), days=10)
        assert result.isna().all()

    def test_no_vol_column(self, calc, stock_df):
        md = pd.DataFrame({"ts_code": ["A.SH"], "close": [10.0]})
        result = calc._compute_volume_breakout(stock_df, md, days=5)
        assert result.isna().all()


# ============================================================================
# industry_hotness (price-based, replaced moneyflow)
# ============================================================================

class TestIndustryHotness:
    @pytest.fixture
    def stock_df(self):
        return pd.DataFrame({
            "ts_code": ["A.SH", "B.SZ"],
            "industry": ["Tech", "Bank"],
        })

    def test_ranks_industries(self, calc, stock_df):
        """Tech stock up 50%, Bank flat → Tech gets higher rank."""
        records = []
        for code, closes in [("A.SH", [10, 15]), ("B.SZ", [10, 10])]:
            for j, close in enumerate(closes):
                records.append({"ts_code": code, "trade_date": f"202601{j+1:02d}", "close": float(close)})
        md = pd.DataFrame(records)
        result = calc._compute_industry_hotness(stock_df, md, days=1)
        # Tech should rank higher
        assert result.loc[0] > result.loc[1]

    def test_empty_multi_daily(self, calc, stock_df):
        result = calc._compute_industry_hotness(stock_df, pd.DataFrame(), days=10)
        assert result.isna().all()


# ============================================================================
# mf_ratio (moneyflow-based, graceful degradation)
# ============================================================================

class TestMfRatio:
    @pytest.fixture
    def stock_df(self):
        return pd.DataFrame({"ts_code": ["A.SH", "B.SZ"]})

    def test_returns_nan_when_no_moneyflow(self, calc, stock_df):
        """Without moneyflow data, should return all NaN gracefully."""
        result = calc._compute_mf_ratio(stock_df, None)
        assert result.isna().all()

    def test_returns_nan_when_empty(self, calc, stock_df):
        result = calc._compute_mf_ratio(stock_df, pd.DataFrame())
        assert result.isna().all()
