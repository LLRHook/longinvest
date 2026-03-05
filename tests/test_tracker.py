"""Tests for src/tracker.py — trade recording, FIFO matching, stats."""

import json

import pytest

from src.tracker import TradeTracker


class TestTradeTracker:
    def test_record_buy(self, tmp_path):
        filepath = tmp_path / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        tracker.record_buy("AAPL", 150.0, 10.0, "screening")

        history = tracker.get_history()
        assert len(history) == 1
        assert history[0]["symbol"] == "AAPL"
        assert history[0]["action"] == "buy"
        assert history[0]["price"] == 150.0

    def test_record_sell(self, tmp_path):
        filepath = tmp_path / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        tracker.record_sell("AAPL", 160.0, 10.0, "stop-loss")

        history = tracker.get_history()
        assert len(history) == 1
        assert history[0]["action"] == "sell"

    def test_persistence_across_instances(self, tmp_path):
        filepath = tmp_path / "trades.json"
        t1 = TradeTracker(filepath=filepath)
        t1.record_buy("AAPL", 150.0, 10.0)

        t2 = TradeTracker(filepath=filepath)
        assert len(t2.get_history()) == 1

    def test_empty_stats(self, tmp_path):
        filepath = tmp_path / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        stats = tracker.compute_stats()
        assert stats["total_trades"] == 0
        assert stats["completed_round_trips"] == 0
        assert stats["win_rate"] == 0

    def test_win_stats(self, tmp_path):
        filepath = tmp_path / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        tracker.record_buy("AAPL", 100.0, 10.0)
        tracker.record_sell("AAPL", 120.0, 10.0)

        stats = tracker.compute_stats()
        assert stats["completed_round_trips"] == 1
        assert stats["win_rate"] == 100.0
        assert stats["avg_win"] > 0

    def test_loss_stats(self, tmp_path):
        filepath = tmp_path / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        tracker.record_buy("AAPL", 100.0, 10.0)
        tracker.record_sell("AAPL", 80.0, 10.0)

        stats = tracker.compute_stats()
        assert stats["completed_round_trips"] == 1
        assert stats["win_rate"] == 0
        assert stats["avg_loss"] < 0

    def test_mixed_stats(self, tmp_path):
        filepath = tmp_path / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        tracker.record_buy("AAPL", 100.0, 10.0)
        tracker.record_sell("AAPL", 120.0, 10.0)  # Win
        tracker.record_buy("MSFT", 200.0, 5.0)
        tracker.record_sell("MSFT", 180.0, 5.0)  # Loss

        stats = tracker.compute_stats()
        assert stats["completed_round_trips"] == 2
        assert stats["win_rate"] == 50.0

    def test_fifo_matching(self, tmp_path):
        filepath = tmp_path / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        tracker.record_buy("AAPL", 100.0, 10.0)
        tracker.record_buy("AAPL", 120.0, 10.0)
        tracker.record_sell("AAPL", 110.0, 10.0)  # Matches first buy at 100

        stats = tracker.compute_stats()
        assert stats["completed_round_trips"] == 1
        # P/L = (110/100) - 1 = 0.10 = 10%
        assert stats["avg_win"] == pytest.approx(10.0)

    def test_corrupt_file_handling(self, tmp_path):
        filepath = tmp_path / "trades.json"
        filepath.write_text("{corrupt!!!")
        tracker = TradeTracker(filepath=filepath)
        assert tracker.get_history() == []

    def test_nonexistent_file(self, tmp_path):
        filepath = tmp_path / "nonexistent" / "trades.json"
        tracker = TradeTracker(filepath=filepath)
        assert tracker.get_history() == []
