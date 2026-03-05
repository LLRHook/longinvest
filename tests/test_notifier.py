"""Tests for src/notifier.py — embed formatting."""

import pytest

from src.notifier import (
    format_circuit_breaker_embed,
    format_dca_buy_embed,
    format_dca_summary_embed,
    format_performance_embed,
    format_screening_embed,
    format_sell_embed,
)
from tests.conftest import make_stock
from src.strategy import ScoredStock


# ===================================================================
# Helpers
# ===================================================================


def _assert_embed_structure(embed: dict):
    """Verify basic Discord embed structure."""
    assert "title" in embed
    assert "description" in embed
    assert "color" in embed
    assert isinstance(embed["color"], int)


# ===================================================================
# TestFormatScreeningEmbed
# ===================================================================


class TestFormatScreeningEmbed:
    def test_structure(self):
        recs = [ScoredStock(stock=make_stock(symbol="A"), score=80.0, reasons=[])]
        embed = format_screening_embed(recs)
        _assert_embed_structure(embed)
        assert "A" in embed["description"]

    def test_max_display(self):
        recs = [
            ScoredStock(stock=make_stock(symbol=f"S{i}"), score=80 - i, reasons=[])
            for i in range(10)
        ]
        embed = format_screening_embed(recs, max_display=3)
        assert "S0" in embed["description"]
        assert "S5" not in embed["description"]

    def test_empty(self):
        embed = format_screening_embed([])
        _assert_embed_structure(embed)
        assert "No recommendations" in embed["description"]


# ===================================================================
# TestFormatPerformanceEmbed
# ===================================================================


class TestFormatPerformanceEmbed:
    def _make_report(self, daily_pl_pct: float, bench_pct: float) -> dict:
        return {
            "date": "Mar 05, 2026",
            "portfolio": {
                "value": 100000,
                "cash": 50000,
                "invested": 50000,
                "daily_pl": daily_pl_pct * 50000,
                "daily_pl_pct": daily_pl_pct,
            },
            "benchmark": {
                "symbol": "SPY",
                "daily_change_pct": bench_pct,
            },
            "positions": [],
            "summary": {
                "outperformance": daily_pl_pct - bench_pct,
                "position_count": 5,
            },
            "risk_metrics": {"sharpe": 1.5, "sortino": 2.0, "max_drawdown": -5.0, "current_drawdown": -1.0, "win_rate": 60.0},
            "sector_exposure": {"Technology": 40.0, "Healthcare": 30.0},
        }

    def test_green_when_outperforms(self):
        embed = format_performance_embed(self._make_report(0.02, 0.01))
        assert embed["color"] == 0x00FF00

    def test_red_when_underperforms(self):
        embed = format_performance_embed(self._make_report(0.005, 0.02))
        assert embed["color"] == 0xFF0000

    def test_risk_fields_present(self):
        embed = format_performance_embed(self._make_report(0.01, 0.005))
        field_names = [f["name"] for f in embed.get("fields", [])]
        assert any("Risk" in n for n in field_names)

    def test_sector_fields_present(self):
        embed = format_performance_embed(self._make_report(0.01, 0.005))
        field_names = [f["name"] for f in embed.get("fields", [])]
        assert any("Sector" in n for n in field_names)


# ===================================================================
# TestFormatCircuitBreakerEmbed
# ===================================================================


class TestFormatCircuitBreakerEmbed:
    def test_structure(self):
        embed = format_circuit_breaker_embed("Portfolio down 9%", -0.09)
        _assert_embed_structure(embed)
        assert embed["color"] == 0xFF0000
        assert "Circuit Breaker" in embed["title"]


# ===================================================================
# TestFormatDcaBuyEmbed
# ===================================================================


class TestFormatDcaBuyEmbed:
    def test_structure(self):
        embed = format_dca_buy_embed(
            symbol="AAPL", name="Apple Inc", score=85.0,
            amount=1000.0, price=150.0, sector="Technology",
            reasons=["Good ROE", "Low PE"], position_count=5,
        )
        _assert_embed_structure(embed)
        assert embed["color"] == 0x22C55E
        assert "AAPL" in embed["title"]


# ===================================================================
# TestFormatDcaSummaryEmbed
# ===================================================================


class TestFormatDcaSummaryEmbed:
    def test_structure(self):
        buys = [
            {"symbol": "AAPL", "name": "Apple", "score": 85, "amount": 1000, "price": 150, "sector": "Tech", "vol": 25.0},
            {"symbol": "MSFT", "name": "Microsoft", "score": 80, "amount": 1000, "price": 400, "sector": "Tech", "vol": 20.0},
        ]
        embed = format_dca_summary_embed(buys, 2000.0)
        _assert_embed_structure(embed)
        assert "2" in embed["title"]  # 2 buys


# ===================================================================
# TestFormatSellEmbed
# ===================================================================


class TestFormatSellEmbed:
    def test_stop_loss_color(self):
        embed = format_sell_embed("AAPL", "trailing stop", 100.0, 90.0, -100.0)
        assert embed["color"] == 0xFF8C00  # Orange for stop

    def test_fundamental_color(self):
        embed = format_sell_embed("AAPL", "degraded fundamentals", 100.0, 90.0, -100.0)
        assert embed["color"] == 0xFF4444  # Red for fundamental


