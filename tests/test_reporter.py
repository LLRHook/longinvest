"""Tests for src/reporter.py — risk metrics, sector exposure."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.reporter import compute_portfolio_risk_metrics, compute_sector_exposure


# ===================================================================
# TestComputePortfolioRiskMetrics
# ===================================================================


def _make_equity_df(returns: list[float], start: float = 100000.0) -> pd.DataFrame:
    """Build a portfolio DataFrame from a list of daily returns."""
    dates = pd.bdate_range("2024-01-01", periods=len(returns) + 1)
    equity = [start]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    return pd.DataFrame({"equity": equity}, index=dates)


def _make_benchmark_df(returns: list[float], start: float = 450.0) -> pd.DataFrame:
    """Build a benchmark DataFrame from daily returns."""
    dates = pd.bdate_range("2024-01-01", periods=len(returns) + 1)
    close = [start]
    for r in returns:
        close.append(close[-1] * (1 + r))
    return pd.DataFrame({"close": close}, index=dates)


class TestComputePortfolioRiskMetrics:
    def test_positive_sharpe(self):
        # Use varied positive returns so std > 0
        np.random.seed(42)
        returns = list(np.random.normal(0.005, 0.01, 100))
        df = _make_equity_df(returns)
        metrics = compute_portfolio_risk_metrics(df)
        assert metrics["sharpe"] > 0

    def test_zero_for_flat(self):
        returns = [0.0] * 100
        df = _make_equity_df(returns)
        metrics = compute_portfolio_risk_metrics(df)
        assert metrics["sharpe"] == 0.0

    def test_sortino_downside_only(self):
        # All positive returns => no downside => sortino should be 0 (no downside std)
        returns = [0.01] * 100
        df = _make_equity_df(returns)
        metrics = compute_portfolio_risk_metrics(df)
        # With all positive returns, downside_std is 0, so sortino = 0
        assert metrics["sortino"] == 0.0

    def test_max_drawdown(self):
        # Up then down
        returns = [0.01] * 50 + [-0.02] * 50
        df = _make_equity_df(returns)
        metrics = compute_portfolio_risk_metrics(df)
        assert metrics["max_drawdown"] < 0

    def test_current_drawdown(self):
        returns = [0.01] * 50 + [-0.01] * 10
        df = _make_equity_df(returns)
        metrics = compute_portfolio_risk_metrics(df)
        assert metrics["current_drawdown"] <= 0

    def test_win_rate_vs_benchmark(self):
        port_returns = [0.01] * 50
        bench_returns = [0.005] * 50
        port_df = _make_equity_df(port_returns)
        bench_df = _make_benchmark_df(bench_returns)
        metrics = compute_portfolio_risk_metrics(port_df, bench_df)
        assert metrics["win_rate"] > 50.0  # Beats benchmark most days

    def test_win_rate_without_benchmark(self):
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 20
        df = _make_equity_df(returns)
        metrics = compute_portfolio_risk_metrics(df)
        assert 0 <= metrics["win_rate"] <= 100

    def test_alpha_beta(self):
        np.random.seed(42)
        bench_returns = list(np.random.normal(0.001, 0.01, 50))
        # Portfolio that tracks benchmark with some alpha
        port_returns = [r + 0.002 for r in bench_returns]
        port_df = _make_equity_df(port_returns)
        bench_df = _make_benchmark_df(bench_returns)
        metrics = compute_portfolio_risk_metrics(port_df, bench_df)
        assert "alpha" in metrics
        assert "beta" in metrics

    def test_insufficient_data(self):
        df = _make_equity_df([0.01])
        metrics = compute_portfolio_risk_metrics(df)
        assert metrics == {}

    def test_zero_std_edge_case(self):
        returns = [0.0] * 50
        df = _make_equity_df(returns)
        metrics = compute_portfolio_risk_metrics(df)
        assert metrics["sharpe"] == 0.0


# ===================================================================
# TestComputeSectorExposure
# ===================================================================


class TestComputeSectorExposure:
    def test_single_sector(self):
        fmp = MagicMock()
        fmp.get_profile.return_value = {"sector": "Technology"}
        positions = [{"symbol": "AAPL", "market_value": 10000.0}]
        exposure = compute_sector_exposure(positions, fmp)
        assert exposure == {"Technology": 100.0}

    def test_multiple_sectors(self):
        fmp = MagicMock()
        fmp.get_profile.side_effect = [
            {"sector": "Technology"},
            {"sector": "Healthcare"},
        ]
        positions = [
            {"symbol": "AAPL", "market_value": 7000.0},
            {"symbol": "JNJ", "market_value": 3000.0},
        ]
        exposure = compute_sector_exposure(positions, fmp)
        assert exposure["Technology"] == 70.0
        assert exposure["Healthcare"] == 30.0

    def test_empty_positions(self):
        fmp = MagicMock()
        exposure = compute_sector_exposure([], fmp)
        assert exposure == {}

    def test_api_fail_uses_unknown(self):
        fmp = MagicMock()
        fmp.get_profile.side_effect = Exception("API error")
        positions = [{"symbol": "AAPL", "market_value": 10000.0}]
        exposure = compute_sector_exposure(positions, fmp)
        assert "Unknown" in exposure
