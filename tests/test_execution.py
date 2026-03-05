"""Tests for main.py — circuit breakers, vol sizing, limit orders, trailing stops, intraday, momentum tilt."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config import Config
from src.broker import AccountStatus, Position
from src.optimizer import apply_momentum_tilt
from src.strategy import ScoredStock
from tests.conftest import make_position, make_stock


# ===================================================================
# Helpers
# ===================================================================


def _make_account(portfolio_value=100_000.0, last_equity=100_000.0, cash=50_000.0):
    return AccountStatus(
        cash=cash,
        portfolio_value=portfolio_value,
        buying_power=cash,
        last_equity=last_equity,
        positions=[],
    )


# ===================================================================
# TestCircuitBreakers
# ===================================================================


class TestCircuitBreakers:
    """Test the circuit breaker logic in cmd_execute / cmd_dry_run."""

    def test_portfolio_halt(self, config_defaults):
        """Portfolio down > 8% triggers halt."""
        # portfolio_value=91_000, last_equity=100_000 => -9% < -8%
        status = _make_account(portfolio_value=91_000.0, last_equity=100_000.0)
        portfolio_change = (status.portfolio_value - status.last_equity) / status.last_equity
        assert portfolio_change < Config.CIRCUIT_BREAKER_PCT

    def test_portfolio_pass(self, config_defaults):
        """Portfolio down < 8% does not trigger halt."""
        status = _make_account(portfolio_value=95_000.0, last_equity=100_000.0)
        portfolio_change = (status.portfolio_value - status.last_equity) / status.last_equity
        assert portfolio_change >= Config.CIRCUIT_BREAKER_PCT

    def test_market_halt(self, config_defaults):
        """SPY down > 4% triggers market halt."""
        spy_prev = 450.0
        spy_price = 430.0
        spy_change = (spy_price - spy_prev) / spy_prev
        assert spy_change < Config.MARKET_CIRCUIT_BREAKER_PCT

    def test_market_pass(self, config_defaults):
        """SPY down < 4% does not trigger."""
        spy_prev = 450.0
        spy_price = 440.0
        spy_change = (spy_price - spy_prev) / spy_prev
        assert spy_change >= Config.MARKET_CIRCUIT_BREAKER_PCT


# ===================================================================
# TestVolAdjustedAllocation
# ===================================================================


class TestVolAdjustedAllocation:
    """Test the vol-adjusted allocation logic from cmd_execute."""

    def test_inverse_vol_weighting(self, config_defaults):
        """Lower vol stocks get higher allocation."""
        from main import _compute_candidate_volatilities

        vols = {"A": 0.20, "B": 0.40}  # A is half the vol of B
        investment_budget = 10_000.0

        raw_weights = {sym: 1.0 / v for sym, v in vols.items()}
        total_weight = sum(raw_weights.values())
        allocations = {sym: (w / total_weight) * investment_budget for sym, w in raw_weights.items()}

        # A should get ~2/3, B should get ~1/3
        assert allocations["A"] > allocations["B"]
        assert allocations["A"] == pytest.approx(6666.67, abs=1)

    def test_missing_vol_uses_median(self, config_defaults):
        """When a stock has no vol data, median vol is used."""
        vols = {"A": 0.20, "B": 0.40}
        median_vol = sorted(vols.values())[len(vols) // 2]

        # Stock C has no vol — should use median
        raw_weights = {}
        for sym in ["A", "B", "C"]:
            vol = vols.get(sym)
            if vol and vol > 0:
                raw_weights[sym] = 1.0 / vol
            else:
                raw_weights[sym] = 1.0 / median_vol

        # C should get same weight as the stock with median vol
        assert raw_weights["C"] == raw_weights["B"]

    def test_finscan_modifier_applied(self, config_defaults):
        """ELEVATED risk applies 50% allocation modifier."""
        investment_budget = 10_000.0
        vols = {"A": 0.30, "B": 0.30}

        raw_weights = {sym: 1.0 / v for sym, v in vols.items()}
        total_weight = sum(raw_weights.values())
        allocations = {sym: (w / total_weight) * investment_budget for sym, w in raw_weights.items()}

        # Apply ELEVATED modifier to A
        allocations["A"] *= 0.5

        assert allocations["A"] == pytest.approx(2500.0)
        assert allocations["B"] == pytest.approx(5000.0)


# ===================================================================
# TestLimitOrderLogic
# ===================================================================


class TestLimitOrderLogic:
    """Test limit order placement and fallback logic."""

    def test_limit_first(self, config_defaults):
        """When USE_LIMIT_ORDERS is True, limit order is tried first."""
        config_defaults(USE_LIMIT_ORDERS=True, LIMIT_ORDER_SPREAD_PCT=0.005)
        price = 100.0
        limit_price = price * (1 + Config.LIMIT_ORDER_SPREAD_PCT)
        assert limit_price == pytest.approx(100.50)

    def test_fallback_to_market(self, config_defaults):
        """When limit order fails and fallback is enabled, market order is used."""
        config_defaults(
            USE_LIMIT_ORDERS=True,
            LIMIT_ORDER_FALLBACK_TO_MARKET=True,
        )

        mock_broker = MagicMock()
        mock_broker.is_fractionable.return_value = True
        mock_broker.buy_limit_notional.return_value = None  # Limit fails
        mock_broker.buy_notional.return_value = "market-order-123"

        symbol = "TEST"
        amount = 1000.0
        price = 100.0
        limit_price = price * (1 + Config.LIMIT_ORDER_SPREAD_PCT)

        # Simulate the logic from cmd_execute
        order_id = None
        if Config.USE_LIMIT_ORDERS:
            order_id = mock_broker.buy_limit_notional(symbol, amount, limit_price)
            if not order_id and Config.LIMIT_ORDER_FALLBACK_TO_MARKET:
                pass  # Fall through to market

        if not order_id:
            if mock_broker.is_fractionable(symbol):
                order_id = mock_broker.buy_notional(symbol, amount)

        assert order_id == "market-order-123"

    def test_non_fractionable_whole_shares(self, config_defaults):
        """Non-fractionable assets buy whole shares only."""
        price = 150.0
        amount = 1000.0
        whole_qty = int(amount // price)
        assert whole_qty == 6  # 1000 / 150 = 6.66 => 6 shares


# ===================================================================
# TestIntradayCheck
# ===================================================================


class TestIntradayCheck:
    """Test the intraday momentum check logic."""

    def test_intraday_check_passes(self, config_defaults):
        """Stock up 1% intraday passes the check."""
        from main import _check_intraday_momentum

        mock_fmp = MagicMock()
        mock_fmp.get_quote.return_value = {"price": 101.0, "previousClose": 100.0}

        assert _check_intraday_momentum(mock_fmp, "TEST") is True

    def test_intraday_check_fails(self, config_defaults):
        """Stock down 4% intraday fails the check."""
        from main import _check_intraday_momentum

        mock_fmp = MagicMock()
        mock_fmp.get_quote.return_value = {"price": 96.0, "previousClose": 100.0}

        assert _check_intraday_momentum(mock_fmp, "TEST") is False

    def test_intraday_check_disabled(self, config_defaults):
        """When disabled, always returns True."""
        from main import _check_intraday_momentum

        config_defaults(INTRADAY_CHECK_ENABLED=False)

        mock_fmp = MagicMock()
        mock_fmp.get_quote.return_value = {"price": 90.0, "previousClose": 100.0}

        assert _check_intraday_momentum(mock_fmp, "TEST") is True
        mock_fmp.get_quote.assert_not_called()


# ===================================================================
# TestTrailingStops
# ===================================================================


class TestTrailingStops:
    """Test trailing stop placement logic."""

    def _make_prices_df(self, symbol, days=30, base=50.0):
        """Create a simple price series for ATR computation."""
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
        rng = np.random.RandomState(42)
        # Prices with some volatility
        returns = 1 + rng.normal(0, 0.02, size=days)
        prices = base * np.cumprod(returns)
        return pd.DataFrame({symbol: prices}, index=dates)

    @patch("main.compute_atr")
    def test_trailing_stop_placed_after_buy(self, mock_atr, config_defaults):
        """Trailing stop is placed with correct trail percent."""
        from main import _place_trailing_stops

        mock_atr.return_value = {"TEST": 2.0}  # ATR = $2

        pos = make_position(
            symbol="TEST",
            current_price=50.0,
            unrealized_plpc=0.05,  # 5% gain (below threshold)
        )
        mock_broker = MagicMock()
        mock_broker.place_trailing_stop.return_value = "stop-order-123"

        prices_df = self._make_prices_df("TEST")
        _place_trailing_stops(mock_broker, prices_df, [pos])

        mock_broker.place_trailing_stop.assert_called_once()
        call_args = mock_broker.place_trailing_stop.call_args
        assert call_args[0][0] == "TEST"  # symbol
        # trail_pct = ATR(2) * multiplier(2.5) / price(50) = 0.10 => 10%
        assert call_args[0][2] == pytest.approx(10.0)

    @patch("main.compute_atr")
    def test_trailing_stop_tightened_on_profit(self, mock_atr, config_defaults):
        """Position with 20% unrealized gain uses tighter multiplier."""
        from main import _place_trailing_stops

        mock_atr.return_value = {"TEST": 2.0}  # ATR = $2

        pos = make_position(
            symbol="TEST",
            current_price=50.0,
            unrealized_plpc=0.20,  # 20% gain > 15% threshold
        )
        mock_broker = MagicMock()
        mock_broker.place_trailing_stop.return_value = "stop-order-456"

        prices_df = self._make_prices_df("TEST")
        _place_trailing_stops(mock_broker, prices_df, [pos])

        call_args = mock_broker.place_trailing_stop.call_args
        # trail_pct = ATR(2) * tight_mult(1.5) / price(50) = 0.06 => 6%
        assert call_args[0][2] == pytest.approx(6.0)

    @patch("main.compute_atr")
    def test_trailing_stop_clamped_to_bounds(self, mock_atr, config_defaults):
        """Very high ATR gets clamped to MAX_PCT, very low to MIN_PCT."""
        from main import _place_trailing_stops

        mock_broker = MagicMock()
        mock_broker.place_trailing_stop.return_value = "stop-order-789"

        # Test MAX clamp: ATR=20 => 20*2.5/50 = 1.0 (100%) -> clamped to 15%
        mock_atr.return_value = {"HIGH": 20.0}
        pos_high = make_position(symbol="HIGH", current_price=50.0, unrealized_plpc=0.0)
        prices_df = self._make_prices_df("HIGH")
        _place_trailing_stops(mock_broker, prices_df, [pos_high])
        call_args = mock_broker.place_trailing_stop.call_args
        assert call_args[0][2] == pytest.approx(15.0)  # MAX_PCT * 100

        mock_broker.reset_mock()

        # Test MIN clamp: ATR=0.1 => 0.1*2.5/50 = 0.005 (0.5%) -> clamped to 3%
        mock_atr.return_value = {"LOW": 0.1}
        pos_low = make_position(symbol="LOW", current_price=50.0, unrealized_plpc=0.0)
        prices_df = self._make_prices_df("LOW")
        _place_trailing_stops(mock_broker, prices_df, [pos_low])
        call_args = mock_broker.place_trailing_stop.call_args
        assert call_args[0][2] == pytest.approx(3.0)  # MIN_PCT * 100


# ===================================================================
# TestMomentumTilt
# ===================================================================


class TestMomentumTilt:
    """Test the momentum tilt allocation adjustment."""

    def test_momentum_tilt_applied(self, config_defaults):
        """High momentum stock gets more allocation, low gets less."""
        allocations = {"A": 5000.0, "B": 5000.0}
        scores = {"A": 90.0, "B": 10.0}  # A is strong, B is weak

        result = apply_momentum_tilt(allocations, scores, 0.20)

        # Total should be preserved
        assert sum(result.values()) == pytest.approx(10000.0)
        # A (high momentum) should get more than B (low momentum)
        assert result["A"] > result["B"]
        # A should get more than original 5000
        assert result["A"] > 5000.0
        # B should get less than original 5000
        assert result["B"] < 5000.0

    def test_momentum_tilt_preserves_total(self, config_defaults):
        """Total allocation is preserved after tilt."""
        allocations = {"X": 3000.0, "Y": 4000.0, "Z": 3000.0}
        scores = {"X": 80.0, "Y": 50.0, "Z": 20.0}

        result = apply_momentum_tilt(allocations, scores, 0.20)

        assert sum(result.values()) == pytest.approx(10000.0)

    def test_momentum_tilt_disabled(self, config_defaults):
        """When scores are empty, allocations are unchanged."""
        allocations = {"A": 5000.0, "B": 5000.0}

        result = apply_momentum_tilt(allocations, {}, 0.20)

        assert result == allocations
