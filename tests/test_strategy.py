"""Tests for src/strategy.py — guardrails, scoring, and DCA targeting."""

from unittest.mock import MagicMock

import pytest

from config import Config
from src.data import StockData
from src.strategy import MultiFactorStrategy, ScoredStock, score_universe
from tests.conftest import make_position, make_stock


# ===================================================================
# TestPassesGuardrails
# ===================================================================


class TestPassesGuardrails:
    """Test the passes_guardrails method on MultiFactorStrategy."""

    @pytest.fixture(autouse=True)
    def _setup(self, config_defaults):
        fmp = MagicMock()
        self.strategy = MultiFactorStrategy(fmp)

    def test_passes_with_defaults(self):
        stock = make_stock()
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is True
        assert failures == []

    def test_fails_low_revenue(self):
        stock = make_stock(revenue=5_000_000)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is False
        assert any("Revenue too low" in f for f in failures)

    def test_fails_negative_revenue_growth(self):
        stock = make_stock(revenue_growth=-0.20)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is False
        assert any("Revenue growth" in f for f in failures)

    def test_fails_none_revenue_growth(self):
        stock = make_stock(revenue_growth=None)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is False
        assert any("Revenue growth" in f for f in failures)

    def test_fails_negative_fcf(self):
        stock = make_stock(free_cash_flow=-1_000_000)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is False
        assert any("Negative FCF" in f for f in failures)

    def test_passes_negative_fcf_when_not_required(self, config_defaults):
        config_defaults(REQUIRE_POSITIVE_FCF=False)
        stock = make_stock(free_cash_flow=-1_000_000)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is True

    def test_passes_none_fcf(self):
        stock = make_stock(free_cash_flow=None)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is True

    def test_fails_high_de(self):
        stock = make_stock(de_ratio=3.0)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is False
        assert any("High D/E" in f for f in failures)

    def test_passes_none_de(self):
        stock = make_stock(de_ratio=None)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is True

    def test_fails_low_dollar_volume(self):
        stock = make_stock(avg_volume=50_000, price=2.0)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is False
        assert any("dollar volume" in f.lower() for f in failures)

    def test_passes_none_avg_volume(self):
        stock = make_stock(avg_volume=None)
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is True

    def test_multiple_failures_reported(self):
        stock = make_stock(
            revenue=1_000_000,
            revenue_growth=-0.50,
            free_cash_flow=-5_000_000,
            de_ratio=5.0,
        )
        passes, failures = self.strategy.passes_guardrails(stock)
        assert passes is False
        assert len(failures) >= 3


# ===================================================================
# TestScoreUniverse
# ===================================================================


class TestScoreUniverse:
    def test_empty_returns_empty(self):
        assert score_universe([]) == []

    def test_single_stock_gets_score(self):
        stock = make_stock()
        scored = score_universe([stock])
        assert len(scored) == 1
        assert scored[0].score > 0

    def test_higher_roe_scores_higher(self):
        low = make_stock(symbol="LOW", roe=0.05)
        high = make_stock(symbol="HIGH", roe=0.30)
        scored = score_universe([low, high])
        scores = {s.stock.symbol: s.score for s in scored}
        assert scores["HIGH"] > scores["LOW"]

    def test_momentum_signals_integrated(self):
        a = make_stock(symbol="A")
        b = make_stock(symbol="B")
        signals = {"A": 0.50, "B": -0.20}
        scored = score_universe([a, b], momentum_signals=signals)
        scores = {s.stock.symbol: s.score for s in scored}
        assert scores["A"] > scores["B"]

    def test_post_earnings_boost(self):
        """Stock with recent earnings beat gets +3 pts."""
        base = make_stock(
            symbol="BASE",
            eps_beat_count=0,
            days_since_last_earnings=30,
        )
        boosted = make_stock(
            symbol="BOOST",
            eps_beat_count=3,
            days_since_last_earnings=3,
        )
        scored = score_universe([base, boosted])
        scores = {s.stock.symbol: s.score for s in scored}
        # The boost stock should have +3 relative to base (approximately)
        assert scores["BOOST"] > scores["BASE"]


# ===================================================================
# TestGetDcaBuyTargets
# ===================================================================


class TestGetDcaBuyTargets:
    @pytest.fixture(autouse=True)
    def _setup(self, config_defaults):
        self.fmp = MagicMock()
        self.strategy = MultiFactorStrategy(self.fmp)

    def _make_scored_list(self, n: int, sector: str = "Technology") -> list[ScoredStock]:
        return [
            ScoredStock(
                stock=make_stock(
                    symbol=f"S{i}",
                    sector=sector,
                    price=50.0,
                    market_cap=500_000_000,
                ),
                score=80.0 - i,
                reasons=[f"reason_{i}"],
            )
            for i in range(n)
        ]

    def test_returns_up_to_dca_top_n(self, monkeypatch):
        monkeypatch.setattr(Config, "DCA_TOP_N", 3)
        scored = self._make_scored_list(10)
        self.strategy.get_buy_recommendations = MagicMock(return_value=scored)

        targets = self.strategy.get_dca_buy_targets(
            positions=[], portfolio_value=100_000.0
        )
        assert len(targets) <= 3

    def test_position_cap_respected(self, monkeypatch):
        monkeypatch.setattr(Config, "DCA_TOP_N", 5)
        monkeypatch.setattr(Config, "MAX_SINGLE_POSITION_PCT", 0.15)
        # One stock already over cap (adding per_pick_amount would exceed 15%)
        positions = [make_position(symbol="S0", market_value=14_500.0)]
        scored = self._make_scored_list(5)
        self.strategy.get_buy_recommendations = MagicMock(return_value=scored)

        targets = self.strategy.get_dca_buy_targets(
            positions=positions, portfolio_value=100_000.0
        )
        target_symbols = [t.stock.symbol for t in targets]
        assert "S0" not in target_symbols

    def test_sector_cap_respected(self, monkeypatch):
        monkeypatch.setattr(Config, "DCA_TOP_N", 5)
        monkeypatch.setattr(Config, "MAX_SECTOR_ALLOCATION", 0.35)
        # Heavy existing sector allocation
        positions = [
            make_position(symbol="EXISTING", market_value=34_000.0),
        ]
        # All candidates in same sector
        scored = self._make_scored_list(5, sector="Technology")
        # Add the existing position as a scored stock so strategy knows its sector
        scored.insert(0, ScoredStock(
            stock=make_stock(symbol="EXISTING", sector="Technology", price=50.0),
            score=90.0,
            reasons=[],
        ))
        self.strategy.get_buy_recommendations = MagicMock(return_value=scored)

        targets = self.strategy.get_dca_buy_targets(
            positions=positions, portfolio_value=100_000.0
        )
        # Should limit how many Tech stocks are added
        assert len(targets) <= 2

    def test_new_position_needs_premium_at_target(self, monkeypatch):
        monkeypatch.setattr(Config, "DCA_TOP_N", 3)
        monkeypatch.setattr(Config, "TARGET_POSITIONS", 2)
        monkeypatch.setattr(Config, "NEW_POSITION_SCORE_THRESHOLD", 0.20)

        # Already at TARGET_POSITIONS
        positions = [
            make_position(symbol="H1", market_value=5000.0),
            make_position(symbol="H2", market_value=5000.0),
        ]
        # Held stocks with known scores
        scored = [
            ScoredStock(stock=make_stock(symbol="H1"), score=60.0, reasons=[]),
            ScoredStock(stock=make_stock(symbol="H2"), score=50.0, reasons=[]),
            # New stock below threshold (worst is 50 * 1.2 = 60)
            ScoredStock(stock=make_stock(symbol="NEW_LOW"), score=55.0, reasons=[]),
            # New stock above threshold
            ScoredStock(stock=make_stock(symbol="NEW_HIGH"), score=65.0, reasons=[]),
        ]
        self.strategy.get_buy_recommendations = MagicMock(return_value=scored)

        targets = self.strategy.get_dca_buy_targets(
            positions=positions, portfolio_value=100_000.0
        )
        target_symbols = [t.stock.symbol for t in targets]
        assert "NEW_HIGH" in target_symbols
        assert "NEW_LOW" not in target_symbols

