"""Shared fixtures and factories for the longinvest test suite."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.broker import Position
from src.data import StockData
from src.finscan import FinScanResult


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_stock(**overrides) -> StockData:
    """Create a StockData with sensible defaults that pass all guardrails."""
    defaults = dict(
        symbol="TEST",
        name="Test Corp",
        price=50.0,
        market_cap=500_000_000,
        sector="Technology",
        country="US",
        is_etf=False,
        is_actively_trading=True,
        de_ratio=0.5,
        roe=0.15,
        gross_margin=0.40,
        revenue_growth=0.12,
        free_cash_flow=50_000_000,
        free_cash_flow_growth=0.10,
        revenue=200_000_000,
        pe_ratio=15.0,
        peg_ratio=1.2,
        price_to_book=3.0,
        ev_to_ebitda=10.0,
        enterprise_value=600_000_000,
        fcf_margin=0.25,
        earnings_growth=0.10,
        eps_beat_count=3,
        quarterly_eps_values=[0.50, 0.55, 0.60, 0.65],
        earnings_growth_accelerating=True,
        revenue_growth_accelerating=True,
        next_earnings_date=None,
        days_since_last_earnings=30,
        avg_volume=1_000_000,
    )
    defaults.update(overrides)
    return StockData(**defaults)


def make_position(**overrides) -> Position:
    """Create a broker Position with sensible defaults."""
    defaults = dict(
        symbol="TEST",
        qty=Decimal("10"),
        market_value=5000.0,
        avg_entry_price=48.0,
        current_price=50.0,
        lastday_price=49.0,
        unrealized_pl=200.0,
        unrealized_plpc=0.04,
        change_today=0.02,
        unrealized_intraday_pl=100.0,
        unrealized_intraday_plpc=0.02,
    )
    defaults.update(overrides)
    return Position(**defaults)


def make_finscan_result(**overrides) -> FinScanResult:
    """Create a FinScanResult with LOW risk defaults."""
    defaults = dict(
        ticker="TEST",
        composite_score=25,
        risk_rating="LOW",
        piotroski_score=7,
        piotroski_signal="STRONG",
        beneish_signal="UNLIKELY_MANIPULATOR",
        altman_zone="SAFE",
        red_flags=[],
    )
    defaults.update(overrides)
    return FinScanResult(**defaults)


def make_prices_df(
    symbols: list[str],
    days: int = 300,
    base_price: float = 50.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic price DataFrame with deterministic randomness.

    Returns a DataFrame with DatetimeIndex (trading days), one column per symbol.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {}
    for i, symbol in enumerate(symbols):
        # Small daily returns so prices stay positive
        daily_returns = 1 + rng.normal(0.0003, 0.015, size=days)
        prices = base_price * np.cumprod(daily_returns)
        data[symbol] = prices
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# 12 Named Fixture Stocks
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_stocks() -> dict[str, StockData]:
    """12 named fixture stocks covering all guardrail/filter/scoring scenarios."""
    return {
        "GROW": make_stock(
            symbol="GROW",
            name="Growth Corp",
            price=60.0,
            roe=0.25,
            gross_margin=0.55,
            revenue_growth=0.30,
            free_cash_flow=80_000_000,
            fcf_margin=0.35,
            pe_ratio=12.0,
            peg_ratio=0.8,
            eps_beat_count=4,
            days_since_last_earnings=30,
        ),
        "VALU": make_stock(
            symbol="VALU",
            name="Value Inc",
            price=40.0,
            roe=0.10,
            gross_margin=0.35,
            revenue_growth=0.05,
            pe_ratio=8.0,
            peg_ratio=0.6,
            price_to_book=1.5,
            ev_to_ebitda=6.0,
        ),
        "JUNK": make_stock(
            symbol="JUNK",
            name="Junk Holdings",
            price=30.0,
            de_ratio=3.0,
            free_cash_flow=-10_000_000,
        ),
        "TINY": make_stock(
            symbol="TINY",
            name="Tiny Micro",
            price=10.0,
            revenue=5_000_000,
        ),
        "EARN": make_stock(
            symbol="EARN",
            name="Earnings Soon",
            price=55.0,
            next_earnings_date=None,  # set dynamically in tests
        ),
        "BEAT": make_stock(
            symbol="BEAT",
            name="Beat Earnings",
            price=52.0,
            eps_beat_count=3,
            days_since_last_earnings=3,
        ),
        "OVRB": make_stock(
            symbol="OVRB",
            name="Overbought Tech",
            price=70.0,
        ),
        "BSMA": make_stock(
            symbol="BSMA",
            name="Below SMA Corp",
            price=45.0,
        ),
        "RISK": make_stock(
            symbol="RISK",
            name="Risky Business",
            price=35.0,
        ),
        "ELEV": make_stock(
            symbol="ELEV",
            name="Elevated Risk",
            price=48.0,
        ),
        "MANI": make_stock(
            symbol="MANI",
            name="Manipulator Inc",
            price=25.0,
        ),
        "ILIQ": make_stock(
            symbol="ILIQ",
            name="Illiquid Penny",
            price=2.0,
            avg_volume=50_000,
        ),
    }


# ---------------------------------------------------------------------------
# Config override fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def config_defaults(monkeypatch):
    """Set dummy API keys and disable notifications for all tests.

    Returns an ``override(**attrs)`` helper so individual tests can
    tweak Config attributes on top of the defaults.
    """
    from config import Config

    monkeypatch.setattr(Config, "ALPACA_API_KEY", "test-key")
    monkeypatch.setattr(Config, "ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setattr(Config, "FMP_API_KEY", "test-fmp-key")
    monkeypatch.setattr(Config, "FINSCAN_API_KEY", "test-finscan-key")
    monkeypatch.setattr(Config, "ENABLE_NOTIFICATIONS", False)
    monkeypatch.setattr(Config, "DISCORD_WEBHOOK_URL", "")
    monkeypatch.setattr(Config, "ALPACA_PAPER", True)

    def override(**attrs):
        for key, value in attrs.items():
            monkeypatch.setattr(Config, key, value)

    return override
