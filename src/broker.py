import logging
import math
from dataclasses import dataclass
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, TrailingStopOrderRequest

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    qty: Decimal
    market_value: float
    avg_entry_price: float
    current_price: float
    lastday_price: float
    # Total unrealized P/L (from entry price)
    unrealized_pl: float
    unrealized_plpc: float
    # Intraday P/L (today only)
    change_today: float  # Stock's daily % change
    unrealized_intraday_pl: float
    unrealized_intraday_plpc: float


@dataclass
class AccountStatus:
    cash: float
    portfolio_value: float
    buying_power: float
    last_equity: float  # Yesterday's closing portfolio value
    positions: list[Position]


class AlpacaBroker:
    def __init__(self):
        if not Config.ALPACA_PAPER:
            raise ValueError("Only paper trading is supported")

        self.client = TradingClient(
            api_key=Config.ALPACA_API_KEY,
            secret_key=Config.ALPACA_SECRET_KEY,
            paper=True,
        )

    def get_account_status(self) -> AccountStatus:
        account = self.client.get_account()
        positions = self.client.get_all_positions()

        return AccountStatus(
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            last_equity=float(account.last_equity),
            positions=[
                Position(
                    symbol=p.symbol,
                    qty=p.qty,
                    market_value=float(p.market_value),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price) if p.current_price else 0.0,
                    lastday_price=float(p.lastday_price) if p.lastday_price else 0.0,
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                    change_today=float(p.change_today) if p.change_today else 0.0,
                    unrealized_intraday_pl=float(p.unrealized_intraday_pl) if p.unrealized_intraday_pl else 0.0,
                    unrealized_intraday_plpc=float(p.unrealized_intraday_plpc) if p.unrealized_intraday_plpc else 0.0,
                )
                for p in positions
            ],
        )

    def is_fractionable(self, symbol: str) -> bool:
        """Check if an asset supports fractional share trading."""
        try:
            asset = self.client.get_asset(symbol)
            return asset.fractionable
        except Exception as e:
            logger.warning(f"Could not check fractionability for {symbol}, assuming False: {e}")
            return False

    def buy_notional(self, symbol: str, notional: float) -> str | None:
        """Place a market buy order for a dollar amount (fractional shares).

        Only works for fractionable assets. Use buy_qty for non-fractionable assets.
        """
        try:
            notional = round(notional, 2)
            order_request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(order_request)
            logger.info(f"Buy order placed: {symbol} for ${notional:.2f}, order_id={order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to place buy order for {symbol}: {e}")
            return None

    def buy_qty(self, symbol: str, qty: int) -> str | None:
        """Place a market buy order for a whole number of shares."""
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(order_request)
            logger.info(f"Buy order placed: {symbol} for {qty} shares, order_id={order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to place buy order for {symbol}: {e}")
            return None

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol, or None if not held."""
        try:
            p = self.client.get_open_position(symbol)
            return Position(
                symbol=p.symbol,
                qty=p.qty,
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price) if p.current_price else 0.0,
                lastday_price=float(p.lastday_price) if p.lastday_price else 0.0,
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                change_today=float(p.change_today) if p.change_today else 0.0,
                unrealized_intraday_pl=float(p.unrealized_intraday_pl) if p.unrealized_intraday_pl else 0.0,
                unrealized_intraday_plpc=float(p.unrealized_intraday_plpc) if p.unrealized_intraday_plpc else 0.0,
            )
        except Exception:
            return None

    def sell_all(self, symbol: str) -> str | None:
        """Close an entire position for a symbol."""
        try:
            order = self.client.close_position(symbol)
            logger.info(f"Closed position: {symbol}, order_id={order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return None

    def sell_notional(self, symbol: str, notional: float) -> str | None:
        """Sell a dollar amount of a position."""
        try:
            notional = round(notional, 2)
            order_request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(order_request)
            logger.info(f"Sell order placed: {symbol} for ${notional:.2f}, order_id={order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to place sell order for {symbol}: {e}")
            return None

    def place_trailing_stop(self, symbol: str, qty: float, trail_percent: float) -> str | None:
        """Place a trailing stop sell order.

        Uses GTC for whole-share quantities and DAY for fractional quantities,
        since Alpaca requires fractional orders to be DAY orders.

        Args:
            symbol: Stock symbol
            qty: Number of shares to cover
            trail_percent: Trail percentage (e.g. 20.0 for 20%)
        """
        try:
            is_whole = qty == math.floor(qty)
            tif = TimeInForce.GTC if is_whole else TimeInForce.DAY
            order_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=tif,
                trail_percent=str(trail_percent),
            )
            order = self.client.submit_order(order_request)
            logger.info(f"Trailing stop placed: {symbol}, qty={qty}, trail={trail_percent}%, tif={tif.value}, order_id={order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to place trailing stop for {symbol}: {e}")
            return None

    def get_open_orders(self, symbol: str) -> list:
        """List open orders for a symbol."""
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol],
            )
            return self.client.get_orders(request)
        except Exception as e:
            logger.error(f"Failed to get open orders for {symbol}: {e}")
            return []

    def cancel_open_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count of cancelled orders."""
        orders = self.get_open_orders(symbol)
        cancelled = 0
        for order in orders:
            try:
                self.client.cancel_order_by_id(order.id)
                cancelled += 1
                logger.info(f"Cancelled order {order.id} for {symbol}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order.id} for {symbol}: {e}")
        return cancelled

    def get_held_symbols(self) -> set[str]:
        """Return set of symbols currently held."""
        positions = self.client.get_all_positions()
        return {p.symbol for p in positions}
