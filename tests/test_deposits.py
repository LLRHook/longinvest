"""Tests for src/deposits.py — deposit tracking and SPY benchmark."""

import json
import os
from unittest.mock import patch

import pytest

from src.deposits import (
    DEPOSITS_FILE,
    compute_portfolio_vs_spy,
    compute_spy_benchmark,
    get_total_invested,
    load_deposits,
    record_deposit,
    save_deposits,
)


@pytest.fixture(autouse=True)
def _clean_deposits(tmp_path):
    """Use a temp file for deposits during tests."""
    tmp_file = str(tmp_path / "deposits.json")
    with patch("src.deposits.DEPOSITS_FILE", tmp_file):
        yield


class TestLoadSave:
    def test_load_empty(self):
        assert load_deposits() == []

    def test_save_and_load(self):
        data = [{"date": "2026-01-01", "amount": 25000, "spy_price": 500.0}]
        save_deposits(data)
        assert load_deposits() == data

    def test_round_trip_multiple(self):
        data = [
            {"date": "2026-01-01", "amount": 25000, "spy_price": 500.0},
            {"date": "2026-01-08", "amount": 25000, "spy_price": 505.0},
        ]
        save_deposits(data)
        loaded = load_deposits()
        assert len(loaded) == 2
        assert loaded[1]["spy_price"] == 505.0


class TestRecordDeposit:
    def test_record_first_deposit(self):
        record = record_deposit(25000.0, 500.0, date="2026-01-01")
        assert record["amount"] == 25000.0
        assert record["spy_price"] == 500.0
        assert record["date"] == "2026-01-01"
        assert len(load_deposits()) == 1

    def test_record_multiple_deposits(self):
        record_deposit(25000.0, 500.0, date="2026-01-01")
        record_deposit(25000.0, 510.0, date="2026-01-08")
        deposits = load_deposits()
        assert len(deposits) == 2

    def test_no_double_record_same_date(self):
        record_deposit(25000.0, 500.0, date="2026-01-01")
        record_deposit(25000.0, 510.0, date="2026-01-01")  # Same date
        deposits = load_deposits()
        assert len(deposits) == 1
        assert deposits[0]["spy_price"] == 500.0  # Kept original

    def test_defaults_to_today(self):
        from datetime import datetime

        record = record_deposit(25000.0, 500.0)
        assert record["date"] == datetime.now().strftime("%Y-%m-%d")


class TestGetTotalInvested:
    def test_empty(self):
        assert get_total_invested() == 0

    def test_with_deposits(self):
        record_deposit(25000.0, 500.0, date="2026-01-01")
        record_deposit(25000.0, 510.0, date="2026-01-08")
        assert get_total_invested() == 50000.0


class TestComputeSpyBenchmark:
    def test_empty_deposits(self):
        result = compute_spy_benchmark(500.0)
        assert result["total_invested"] == 0
        assert result["spy_value"] == 0
        assert result["num_deposits"] == 0

    def test_single_deposit_no_change(self):
        record_deposit(25000.0, 500.0, date="2026-01-01")
        result = compute_spy_benchmark(500.0)  # SPY unchanged
        assert result["total_invested"] == 25000.0
        assert result["spy_value"] == pytest.approx(25000.0)
        assert result["spy_return_pct"] == pytest.approx(0.0)
        assert result["spy_shares"] == pytest.approx(50.0)  # 25000/500

    def test_single_deposit_spy_up(self):
        record_deposit(25000.0, 500.0, date="2026-01-01")
        result = compute_spy_benchmark(550.0)  # SPY up 10%
        assert result["spy_value"] == pytest.approx(27500.0)  # 50 shares * 550
        assert result["spy_return_pct"] == pytest.approx(10.0)

    def test_multiple_deposits_dca(self):
        # Week 1: buy at 500, week 2: buy at 400 (dip)
        record_deposit(25000.0, 500.0, date="2026-01-01")
        record_deposit(25000.0, 400.0, date="2026-01-08")
        # Shares: 50 + 62.5 = 112.5
        result = compute_spy_benchmark(450.0)
        assert result["total_invested"] == 50000.0
        assert result["spy_shares"] == pytest.approx(112.5)
        assert result["spy_value"] == pytest.approx(112.5 * 450.0)
        assert result["num_deposits"] == 2


class TestComputePortfolioVsSpy:
    def test_empty(self):
        result = compute_portfolio_vs_spy(0, 500.0)
        assert result["total_invested"] == 0
        assert result["alpha_pct"] == 0

    def test_portfolio_beats_spy(self):
        record_deposit(25000.0, 500.0, date="2026-01-01")
        # Portfolio worth 27000 (+8%), SPY at 525 (+5%)
        result = compute_portfolio_vs_spy(27000.0, 525.0)
        assert result["portfolio_return_pct"] == pytest.approx(8.0)
        assert result["spy_return_pct"] == pytest.approx(5.0)
        assert result["alpha_pct"] == pytest.approx(3.0)
        assert result["num_weeks"] == 1

    def test_portfolio_lags_spy(self):
        record_deposit(25000.0, 500.0, date="2026-01-01")
        # Portfolio worth 24000 (-4%), SPY at 525 (+5%)
        result = compute_portfolio_vs_spy(24000.0, 525.0)
        assert result["portfolio_return_pct"] == pytest.approx(-4.0)
        assert result["spy_return_pct"] == pytest.approx(5.0)
        assert result["alpha_pct"] == pytest.approx(-9.0)
