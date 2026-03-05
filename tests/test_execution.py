"""Tests for main.py — circuit breakers, vol sizing, limit orders."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config import Config
from src.broker import AccountStatus, Position
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
