import logging
from dataclasses import dataclass
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

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

    def buy_notional(self, symbol: str, notional: float) -> str | None:
        """Place a market buy order for a dollar amount (fractional shares)."""
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

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol, or None if not held."""
        try:
            p = self.client.get_open_position(symbol)
            return Position(
                symbol=p.symbol,
                qty=p.qty,
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
            )
        except Exception:
            return None

    def get_held_symbols(self) -> set[str]:
        """Return set of symbols currently held."""
        positions = self.client.get_all_positions()
        return {p.symbol for p in positions}
