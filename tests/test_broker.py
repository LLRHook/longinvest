"""Tests for src/broker.py — order placement, fractionable handling."""

from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.broker import AccountStatus, AlpacaBroker, Position


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def broker(mock_client, config_defaults):
    with patch("src.broker.TradingClient", return_value=mock_client):
        b = AlpacaBroker()
    b.client = mock_client
    return b


class TestBuyNotional:
    def test_success(self, broker, mock_client):
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_client.submit_order.return_value = mock_order

        result = broker.buy_notional("AAPL", 1000.0)
        assert result == "order-123"
        mock_client.submit_order.assert_called_once()

    def test_failure_returns_none(self, broker, mock_client):
        mock_client.submit_order.side_effect = Exception("API error")
        result = broker.buy_notional("AAPL", 1000.0)
        assert result is None


class TestBuyQty:
    def test_success(self, broker, mock_client):
        mock_order = MagicMock()
        mock_order.id = "order-456"
        mock_client.submit_order.return_value = mock_order

        result = broker.buy_qty("AAPL", 10)
        assert result == "order-456"


class TestBuyLimitNotional:
    def test_success(self, broker, mock_client):
        mock_order = MagicMock()
        mock_order.id = "limit-123"
        mock_client.submit_order.return_value = mock_order

        result = broker.buy_limit_notional("AAPL", 1000.0, 150.0)
        assert result == "limit-123"

    def test_zero_limit_price(self, broker, mock_client):
        result = broker.buy_limit_notional("AAPL", 1000.0, 0.0)
        assert result is None


class TestBuyLimitQty:
    def test_success(self, broker, mock_client):
        mock_order = MagicMock()
        mock_order.id = "limit-qty-789"
        mock_client.submit_order.return_value = mock_order

        result = broker.buy_limit_qty("AAPL", 5, 150.0)
        assert result == "limit-qty-789"


class TestSellAll:
    def test_success(self, broker, mock_client):
        mock_order = MagicMock()
        mock_order.id = "sell-all-1"
        mock_client.close_position.return_value = mock_order

        result = broker.sell_all("AAPL")
        assert result == "sell-all-1"

    def test_failure(self, broker, mock_client):
        mock_client.close_position.side_effect = Exception("No position")
        result = broker.sell_all("AAPL")
        assert result is None


class TestSellNotional:
    def test_success(self, broker, mock_client):
        mock_order = MagicMock()
        mock_order.id = "sell-not-1"
        mock_client.submit_order.return_value = mock_order

        result = broker.sell_notional("AAPL", 500.0)
        assert result == "sell-not-1"


class TestIsFractionable:
    def test_fractionable_true(self, broker, mock_client):
        mock_asset = MagicMock()
        mock_asset.fractionable = True
        mock_client.get_asset.return_value = mock_asset

        assert broker.is_fractionable("AAPL") is True

    def test_fractionable_false(self, broker, mock_client):
        mock_asset = MagicMock()
        mock_asset.fractionable = False
        mock_client.get_asset.return_value = mock_asset

        assert broker.is_fractionable("BRK.A") is False

    def test_error_returns_false(self, broker, mock_client):
        mock_client.get_asset.side_effect = Exception("Not found")
        assert broker.is_fractionable("XXX") is False


class TestGetOpenOrders:
    def test_returns_orders(self, broker, mock_client):
        mock_client.get_orders.return_value = [MagicMock(), MagicMock()]
        orders = broker.get_open_orders("AAPL")
        assert len(orders) == 2

    def test_error_returns_empty(self, broker, mock_client):
        mock_client.get_orders.side_effect = Exception("Error")
        assert broker.get_open_orders("AAPL") == []


class TestCancelOpenOrders:
    def test_cancels_all(self, broker, mock_client):
        order1 = MagicMock()
        order1.id = "o1"
        order2 = MagicMock()
        order2.id = "o2"
        mock_client.get_orders.return_value = [order1, order2]

        count = broker.cancel_open_orders("AAPL")
        assert count == 2


class TestGetAccountStatus:
    def test_mapping(self, broker, mock_client):
        mock_account = MagicMock()
        mock_account.cash = "50000.00"
        mock_account.portfolio_value = "100000.00"
        mock_account.buying_power = "50000.00"
        mock_account.last_equity = "99000.00"
        mock_client.get_account.return_value = mock_account
        mock_client.get_all_positions.return_value = []

        status = broker.get_account_status()
        assert status.cash == 50000.0
        assert status.portfolio_value == 100000.0
        assert isinstance(status, AccountStatus)


class TestPaperOnlyGuard:
    def test_non_paper_raises(self, config_defaults):
        from config import Config
        Config.ALPACA_PAPER = False
        with pytest.raises(ValueError, match="paper"):
            with patch("src.broker.TradingClient"):
                AlpacaBroker()
        Config.ALPACA_PAPER = True
