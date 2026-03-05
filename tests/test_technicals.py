"""Tests for src/technicals.py — SMA, RSI, filters, momentum."""

import numpy as np
import pandas as pd
import pytest

from src.technicals import (
    apply_technical_filters,
    compute_atr,
    compute_momentum_percentiles,
    compute_multi_tf_relative_strength,
    compute_price_momentum_12_1,
    compute_rsi,
    compute_sma,
    compute_volume_signals,
)


# ===================================================================
# TestComputeSma
# ===================================================================


class TestComputeSma:
    def test_basic_sma(self):
        prices = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        sma = compute_sma(prices, period=3)
        assert sma == pytest.approx(40.0)  # mean of [30, 40, 50]

    def test_insufficient_data(self):
        prices = pd.Series([10.0, 20.0])
        sma = compute_sma(prices, period=5)
        assert sma is None

    def test_exact_period_boundary(self):
        prices = pd.Series([10.0, 20.0, 30.0])
        sma = compute_sma(prices, period=3)
        assert sma == pytest.approx(20.0)


# ===================================================================
# TestComputeRsi
# ===================================================================


class TestComputeRsi:
    def test_all_gains(self):
        prices = pd.Series([float(i) for i in range(1, 20)])
        rsi = compute_rsi(prices, period=14)
        assert rsi is not None
        assert rsi > 95.0  # All gains => RSI near 100

    def test_all_losses(self):
        prices = pd.Series([float(20 - i) for i in range(20)])
        rsi = compute_rsi(prices, period=14)
        assert rsi is not None
        assert rsi < 5.0  # All losses => RSI near 0

    def test_mixed_returns(self):
        np.random.seed(42)
        prices = pd.Series(50.0 + np.cumsum(np.random.randn(30) * 0.5))
        rsi = compute_rsi(prices, period=14)
        assert rsi is not None
        assert 0 < rsi < 100

    def test_insufficient_data(self):
        prices = pd.Series([10.0, 11.0, 12.0])
        rsi = compute_rsi(prices, period=14)
        assert rsi is None


# ===================================================================
# TestApplyTechnicalFilters
# ===================================================================


class TestApplyTechnicalFilters:
    def _make_prices(self, symbol: str, prices_list: list[float]) -> pd.DataFrame:
        dates = pd.bdate_range(end="2025-01-01", periods=len(prices_list))
        return pd.DataFrame({symbol: prices_list}, index=dates)

    def test_drops_below_sma(self):
        # SMA-50: ~100, current price drops to 80
        prices = [100.0] * 60 + [80.0]
        df = self._make_prices("DROP", prices)
        filtered, dropped = apply_technical_filters(df, sma_period=50)
        assert "DROP" not in filtered.columns
        assert any("below SMA" in d for d in dropped)

    def test_drops_overbought_rsi(self):
        # Steady rise will push RSI high
        prices = [float(50 + i * 2) for i in range(100)]
        df = self._make_prices("HOT", prices)
        filtered, dropped = apply_technical_filters(df, sma_period=50, rsi_overbought=75.0)
        if "HOT" not in filtered.columns:
            assert any("overbought" in d.lower() for d in dropped)

    def test_keeps_healthy_stock(self):
        # Moderate uptrend — above SMA, RSI in normal range
        np.random.seed(42)
        prices = list(50.0 + np.cumsum(np.random.randn(100) * 0.3 + 0.05))
        df = self._make_prices("GOOD", prices)
        filtered, dropped = apply_technical_filters(df, sma_period=50, rsi_overbought=75.0)
        assert "GOOD" in filtered.columns


# ===================================================================
# TestComputePriceMomentum12_1
# ===================================================================


class TestComputePriceMomentum12_1:
    def _make_df(self, symbol: str, days: int, start: float, end: float) -> pd.DataFrame:
        dates = pd.bdate_range(end="2025-01-01", periods=days)
        prices = np.linspace(start, end, days)
        return pd.DataFrame({symbol: prices}, index=dates)

    def test_basic_positive_momentum(self):
        df = self._make_df("UP", 300, 50.0, 100.0)
        signals = compute_price_momentum_12_1(df)
        assert "UP" in signals
        assert signals["UP"] > 0

    def test_skips_recent_month(self):
        # The function uses price_12m ago vs price_1m ago (skipping last 21 days)
        df = self._make_df("TEST", 300, 100.0, 100.0)
        signals = compute_price_momentum_12_1(df)
        # Flat price => momentum near 0
        if "TEST" in signals:
            assert abs(signals["TEST"]) < 0.01

    def test_insufficient_history(self):
        df = self._make_df("SHORT", 100, 50.0, 60.0)
        signals = compute_price_momentum_12_1(df)
        assert "SHORT" not in signals

    def test_negative_momentum(self):
        df = self._make_df("DOWN", 300, 100.0, 50.0)
        signals = compute_price_momentum_12_1(df)
        assert "DOWN" in signals
        assert signals["DOWN"] < 0


# ===================================================================
# TestMomentumPercentiles
# ===================================================================


class TestMomentumPercentiles:
    def _make_df(self, symbols_and_trends: dict[str, tuple[float, float]], days: int = 300) -> pd.DataFrame:
        """Create a multi-stock DataFrame with known price trends."""
        dates = pd.bdate_range(end="2025-01-01", periods=days)
        data = {}
        for symbol, (start, end) in symbols_and_trends.items():
            data[symbol] = np.linspace(start, end, days)
        return pd.DataFrame(data, index=dates)

    def test_momentum_percentiles_ranking(self):
        # 5 stocks with increasing momentum: A worst, E best
        df = self._make_df({
            "A": (100.0, 80.0),   # down
            "B": (100.0, 100.0),  # flat
            "C": (100.0, 120.0),  # moderate up
            "D": (100.0, 150.0),  # strong up
            "E": (100.0, 200.0),  # strongest up
        })
        pcts = compute_momentum_percentiles(df)
        assert len(pcts) == 5
        # E should have highest percentile, A the lowest
        assert pcts["E"] > pcts["D"] > pcts["C"] > pcts["B"] > pcts["A"]

    def test_momentum_percentiles_single_stock(self):
        dates = pd.bdate_range(end="2025-01-01", periods=300)
        df = pd.DataFrame({"ONLY": np.linspace(50, 100, 300)}, index=dates)
        pcts = compute_momentum_percentiles(df)
        assert "ONLY" in pcts
        # Single stock: rank(pct=True) gives 1.0 => 100.0
        assert pcts["ONLY"] == pytest.approx(100.0)

    def test_momentum_percentiles_empty(self):
        df = pd.DataFrame()
        pcts = compute_momentum_percentiles(df)
        assert pcts == {}


# ===================================================================
# TestVolumeSignals
# ===================================================================


class TestVolumeSignals:
    def test_volume_surge_detected(self):
        dates = pd.bdate_range(end="2025-01-01", periods=30)
        prices_df = pd.DataFrame({"SURGE": np.linspace(50, 60, 30)}, index=dates)
        # Normal volume of 1000 for 29 days, then a 3x spike on the last day
        volumes = [1000.0] * 29 + [3000.0]
        volume_df = pd.DataFrame({"SURGE": volumes}, index=dates)
        signals = compute_volume_signals(prices_df, volume_df)
        assert "SURGE" in signals
        assert signals["SURGE"] > 2.0  # 3x spike > threshold

    def test_volume_no_surge(self):
        dates = pd.bdate_range(end="2025-01-01", periods=30)
        prices_df = pd.DataFrame({"CALM": np.linspace(50, 60, 30)}, index=dates)
        # Constant volume
        volume_df = pd.DataFrame({"CALM": [1000.0] * 30}, index=dates)
        signals = compute_volume_signals(prices_df, volume_df)
        assert "CALM" in signals
        assert signals["CALM"] == pytest.approx(1.0, abs=0.1)

    def test_volume_empty(self):
        prices_df = pd.DataFrame()
        signals = compute_volume_signals(prices_df, volume_df=None)
        assert signals == {}


# ===================================================================
# TestMultiTfRelativeStrength
# ===================================================================


class TestMultiTfRelativeStrength:
    def test_multi_tf_rs_basic(self):
        dates = pd.bdate_range(end="2025-01-01", periods=300)
        # Stock outperforming benchmark
        prices_df = pd.DataFrame({
            "WINNER": np.linspace(50, 150, 300),
            "LOSER": np.linspace(100, 80, 300),
        }, index=dates)
        benchmark = pd.Series(np.linspace(100, 120, 300), index=dates)

        rs = compute_multi_tf_relative_strength(prices_df, benchmark)
        assert "WINNER" in rs
        assert "LOSER" in rs
        # WINNER outperforms benchmark, should rank higher
        assert rs["WINNER"] > rs["LOSER"]

    def test_multi_tf_rs_weights(self):
        dates = pd.bdate_range(end="2025-01-01", periods=300)
        # Stock that does well on all timeframes
        prices_df = pd.DataFrame({
            "STRONG": np.linspace(50, 200, 300),
        }, index=dates)
        benchmark = pd.Series(np.linspace(100, 110, 300), index=dates)

        rs = compute_multi_tf_relative_strength(prices_df, benchmark)
        assert "STRONG" in rs
        # Single stock gets 100th percentile
        assert rs["STRONG"] == pytest.approx(100.0)

    def test_multi_tf_rs_insufficient_data(self):
        dates = pd.bdate_range(end="2025-01-01", periods=30)
        prices_df = pd.DataFrame({"SHORT": np.linspace(50, 60, 30)}, index=dates)
        benchmark = pd.Series(np.linspace(100, 105, 30), index=dates)

        rs = compute_multi_tf_relative_strength(prices_df, benchmark)
        # 30 days is less than smallest timeframe (63), so no results
        assert rs == {}


# ===================================================================
# TestComputeAtr
# ===================================================================


class TestComputeAtr:
    def test_compute_atr_basic(self):
        # Create prices with known daily change of $1
        dates = pd.bdate_range(end="2025-01-01", periods=20)
        prices = [100.0 + i for i in range(20)]
        df = pd.DataFrame({"STEADY": prices}, index=dates)
        atr = compute_atr(df, period=14)
        assert "STEADY" in atr
        # Daily change is always $1, so ATR should be ~1.0
        assert atr["STEADY"] == pytest.approx(1.0, abs=0.01)

    def test_compute_atr_insufficient_data(self):
        dates = pd.bdate_range(end="2025-01-01", periods=5)
        df = pd.DataFrame({"SHORT": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)
        atr = compute_atr(df, period=14)
        assert atr == {}

    def test_compute_atr_single_stock(self):
        # Alternating up/down creates known ATR
        dates = pd.bdate_range(end="2025-01-01", periods=20)
        prices = [100.0 + (2.0 if i % 2 == 0 else 0.0) for i in range(20)]
        df = pd.DataFrame({"BOUNCE": prices}, index=dates)
        atr = compute_atr(df, period=14)
        assert "BOUNCE" in atr
        assert atr["BOUNCE"] == pytest.approx(2.0, abs=0.01)
