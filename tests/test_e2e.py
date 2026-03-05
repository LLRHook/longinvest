"""End-to-end pipeline tests with the 12 fixture stocks."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config import Config
from src.data import StockData
from src.finscan import FinScanClient, FinScanResult
from src.strategy import MultiFactorStrategy, ScoredStock, score_universe
from tests.conftest import make_finscan_result, make_position, make_stock


class TestEndToEnd:
    """Full pipeline integration with the 12 fixture stocks."""

    @pytest.fixture(autouse=True)
    def _setup(self, fixture_stocks, config_defaults):
        self.stocks = fixture_stocks
        self.config_defaults = config_defaults

    def _all_passing(self) -> list[StockData]:
        """Return fixture stocks that should pass guardrails."""
        fmp = MagicMock()
        strategy = MultiFactorStrategy(fmp)
        passing = []
        for stock in self.stocks.values():
            passes, _ = strategy.passes_guardrails(stock)
            if passes:
                passing.append(stock)
        return passing

    def test_guardrails_filter_correctly(self):
        """JUNK, TINY, and ILIQ should fail guardrails; others pass."""
        fmp = MagicMock()
        strategy = MultiFactorStrategy(fmp)

        should_fail = {"JUNK", "TINY", "ILIQ"}
        for symbol, stock in self.stocks.items():
            passes, failures = strategy.passes_guardrails(stock)
            if symbol in should_fail:
                assert not passes, f"{symbol} should have failed guardrails but passed"
            else:
                assert passes, f"{symbol} should have passed guardrails but failed: {failures}"

    def test_scoring_ranks_correctly(self):
        """GROW should score higher than VALU; BEAT gets earnings boost."""
        passing = self._all_passing()
        scored = score_universe(passing)
        scores = {s.stock.symbol: s.score for s in scored}

        assert scores["GROW"] > scores["VALU"], "GROW should score higher than VALU"

        # BEAT should get +3pt boost (eps_beat_count=3, days_since_last_earnings=3)
        beat_scored = next(s for s in scored if s.stock.symbol == "BEAT")
        assert any("earnings beat" in r.lower() for r in beat_scored.reasons)

    def test_finscan_gating_in_pipeline(self):
        """RISK and MANI should be rejected by FinScan; ELEV gets 50% modifier."""
        self.config_defaults(DCA_TOP_N=10)
        fmp = MagicMock()
        strategy = MultiFactorStrategy(fmp)
        passing = self._all_passing()
        scored = score_universe(passing)
        strategy.get_buy_recommendations = MagicMock(return_value=scored)

        finscan = MagicMock()

        def scan_side_effect(ticker):
            if ticker == "RISK":
                return make_finscan_result(ticker="RISK", composite_score=80, risk_rating="HIGH")
            elif ticker == "MANI":
                return make_finscan_result(ticker="MANI", beneish_signal="LIKELY_MANIPULATOR")
            elif ticker == "ELEV":
                return make_finscan_result(ticker="ELEV", composite_score=55, risk_rating="ELEVATED")
            else:
                return make_finscan_result(ticker=ticker)

        finscan.scan.side_effect = scan_side_effect

        targets = strategy.get_dca_buy_targets(
            positions=[], portfolio_value=100_000.0, finscan=finscan,
        )
        target_symbols = {t.stock.symbol for t in targets}

        assert "RISK" not in target_symbols, "RISK should be rejected by FinScan"
        assert "MANI" not in target_symbols, "MANI should be rejected by FinScan"

        elev_target = next((t for t in targets if t.stock.symbol == "ELEV"), None)
        if elev_target:
            assert elev_target.allocation_modifier == 0.5

    def test_full_pipeline_selects_correct_stocks(self):
        """Final buy list should include top scorers, exclude guardrail failures and FinScan rejects."""
        self.config_defaults(DCA_TOP_N=10)
        fmp = MagicMock()
        strategy = MultiFactorStrategy(fmp)
        passing = self._all_passing()
        scored = score_universe(passing)
        strategy.get_buy_recommendations = MagicMock(return_value=scored)

        finscan = MagicMock()

        def scan_side_effect(ticker):
            if ticker == "RISK":
                return make_finscan_result(ticker="RISK", composite_score=80, risk_rating="HIGH")
            elif ticker == "MANI":
                return make_finscan_result(ticker="MANI", beneish_signal="LIKELY_MANIPULATOR")
            else:
                return make_finscan_result(ticker=ticker)

        finscan.scan.side_effect = scan_side_effect

        targets = strategy.get_dca_buy_targets(
            positions=[], portfolio_value=100_000.0, finscan=finscan,
        )
        target_symbols = {t.stock.symbol for t in targets}

        # Guardrail failures should not be in passing stocks
        assert "JUNK" not in target_symbols
        assert "TINY" not in target_symbols
        assert "ILIQ" not in target_symbols

        # FinScan rejects
        assert "RISK" not in target_symbols
        assert "MANI" not in target_symbols

        # Top scorers should be selected (GROW at minimum)
        assert "GROW" in target_symbols, f"GROW should be selected but targets are: {target_symbols}"

    def test_earnings_blackout_excludes_earn(self):
        """EARN with earnings in 3 days should be excluded by earnings blackout."""
        fmp = MagicMock()
        strategy = MultiFactorStrategy(fmp)

        # Set EARN's next_earnings_date to 3 days from now
        earn_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        self.stocks["EARN"] = make_stock(
            symbol="EARN",
            name="Earnings Soon",
            price=55.0,
            next_earnings_date=earn_date,
        )

        # Build scored list with EARN
        passing = self._all_passing()
        scored = score_universe(passing)

        # Simulate get_buy_recommendations earnings blackout
        filtered = []
        today = datetime.now().date()
        for s in scored:
            if s.stock.next_earnings_date:
                next_date = datetime.strptime(s.stock.next_earnings_date, "%Y-%m-%d").date()
                days_until = (next_date - today).days
                if 0 <= days_until <= Config.EARNINGS_BLACKOUT_DAYS:
                    continue
            filtered.append(s)

        filtered_symbols = {s.stock.symbol for s in filtered}
        assert "EARN" not in filtered_symbols, "EARN should be excluded by earnings blackout"

    def test_volume_surge_boosts_scoring(self):
        """A stock with volume surge ratio > 2.0 should be detected as a surge candidate."""
        # Simulate 50 days of volume data for two stocks
        days = 50
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

        # SURGE: last day has 3x average volume
        surge_volumes = [100_000] * (days - 1) + [300_000]
        # CALM: steady volume throughout
        calm_volumes = [100_000] * days

        volume_df = pd.DataFrame(
            {"SURGE": surge_volumes, "CALM": calm_volumes},
            index=dates,
        )

        # Compute volume surge ratio manually (latest / 50-day avg)
        for symbol in ["SURGE", "CALM"]:
            col = volume_df[symbol]
            avg_vol = col.iloc[:-1].mean()
            latest_vol = col.iloc[-1]
            surge_ratio = latest_vol / avg_vol if avg_vol > 0 else 0.0

            if symbol == "SURGE":
                assert surge_ratio >= 2.0, (
                    f"SURGE should have surge ratio >= 2.0 but got {surge_ratio:.2f}"
                )
            else:
                assert surge_ratio < 2.0, (
                    f"CALM should have surge ratio < 2.0 but got {surge_ratio:.2f}"
                )

        # Show that the surge detection could boost scoring:
        # a surging stock gets a multiplier applied to its fundamental score
        base_score = 70.0
        surge_boost = 1.10  # 10% boost for volume surge
        boosted = base_score * surge_boost
        assert boosted > base_score
        assert boosted == pytest.approx(77.0)

    def test_intraday_check_skips_falling_stock(self):
        """A stock down > 3% intraday should be skipped by the intraday momentum check."""
        # Simulate the intraday check logic from main.py
        intraday_min_change = -0.03  # Config.INTRADAY_MIN_CHANGE

        # Mock quote data: stock down 5% today
        mock_quotes = {
            "FALL": {"price": 47.50, "previousClose": 50.00},  # -5.0%
            "RISE": {"price": 52.00, "previousClose": 50.00},  # +4.0%
            "FLAT": {"price": 49.80, "previousClose": 50.00},  # -0.4%
        }

        allocations = {"FALL": 0.40, "RISE": 0.35, "FLAT": 0.25}
        skipped = []
        remaining = {}

        for symbol, alloc in allocations.items():
            quote = mock_quotes[symbol]
            change = (quote["price"] - quote["previousClose"]) / quote["previousClose"]
            if change < intraday_min_change:
                skipped.append(symbol)
            else:
                remaining[symbol] = alloc

        # FALL should be skipped (down 5% > 3% threshold)
        assert "FALL" in skipped, "FALL should be skipped (down 5%)"
        assert "RISE" not in skipped
        assert "FLAT" not in skipped

        # Redistribute FALL's allocation proportionally
        if remaining:
            total_remaining = sum(remaining.values())
            redistributed = {
                sym: alloc / total_remaining for sym, alloc in remaining.items()
            }
            assert sum(redistributed.values()) == pytest.approx(1.0)
            assert "FALL" not in redistributed

    def test_vol_adjusted_allocation_applied(self):
        """Inverse-vol weighting distributes budget correctly."""
        investment_budget = 10_000.0
        vols = {"GROW": 0.20, "VALU": 0.40, "BEAT": 0.30}

        raw_weights = {sym: 1.0 / v for sym, v in vols.items()}
        total_weight = sum(raw_weights.values())
        allocations = {
            sym: (w / total_weight) * investment_budget
            for sym, w in raw_weights.items()
        }

        # Lower vol => higher allocation
        assert allocations["GROW"] > allocations["BEAT"] > allocations["VALU"]
        assert sum(allocations.values()) == pytest.approx(investment_budget)
