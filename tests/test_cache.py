"""Tests for src/cache.py — save/load/expiry, serialization helpers."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.cache import (
    CacheManager,
    dataframe_to_dict,
    dict_to_dataframe,
    dict_to_scored_stock,
    dict_to_stock_data,
    scored_stock_to_dict,
    stock_data_to_dict,
    symbols_hash,
)
from tests.conftest import make_stock


# ===================================================================
# TestCacheManager
# ===================================================================


class TestCacheManager:
    def test_save_and_load(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        cache.save("screener", "test_key", {"hello": "world"})
        result = cache.load("screener", "test_key")
        assert result == {"hello": "world"}

    def test_missing_key_returns_none(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        result = cache.load("screener", "nonexistent")
        assert result is None

    def test_expired_returns_none(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=1)
        cache.save("screener", "old_key", {"data": 1})

        # Manually backdate the created_at
        path = cache._get_path("screener", "old_key")
        with open(path, "r") as f:
            entry = json.load(f)
        entry["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
        with open(path, "w") as f:
            json.dump(entry, f)

        result = cache.load("screener", "old_key")
        assert result is None

    def test_wrong_version_returns_none(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        cache.save("screener", "v_key", {"data": 1})

        # Tamper with version
        path = cache._get_path("screener", "v_key")
        with open(path, "r") as f:
            entry = json.load(f)
        entry["version"] = 999
        with open(path, "w") as f:
            json.dump(entry, f)

        result = cache.load("screener", "v_key")
        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        cache.save("screener", "corrupt", {"data": 1})

        path = cache._get_path("screener", "corrupt")
        with open(path, "w") as f:
            f.write("{invalid json!!!")

        result = cache.load("screener", "corrupt")
        assert result is None

    def test_clear_category(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        cache.save("screener", "a", 1)
        cache.save("screener", "b", 2)
        cache.save("fundamentals", "c", 3)

        deleted = cache.clear("screener")
        assert deleted == 2
        assert cache.load("fundamentals", "c") == 3

    def test_clear_all(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        cache.save("screener", "a", 1)
        cache.save("fundamentals", "b", 2)

        deleted = cache.clear()
        assert deleted == 2

    def test_date_key_format(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        key = cache.get_date_key()
        # Should be YYYY-MM-DD
        datetime.strptime(key, "%Y-%m-%d")


# ===================================================================
# TestSerializationHelpers
# ===================================================================


class TestSerializationHelpers:
    def test_stock_data_roundtrip(self):
        stock = make_stock(symbol="ROUND", roe=0.18)
        d = stock_data_to_dict(stock)
        restored = dict_to_stock_data(d)
        assert restored.symbol == "ROUND"
        assert restored.roe == pytest.approx(0.18)

    def test_scored_stock_roundtrip(self):
        from src.strategy import ScoredStock
        stock = make_stock(symbol="SC")
        scored = ScoredStock(stock=stock, score=75.5, reasons=["good ROE"])
        d = scored_stock_to_dict(scored)
        restored = dict_to_scored_stock(d)
        assert restored.stock.symbol == "SC"
        assert restored.score == pytest.approx(75.5)
        assert restored.reasons == ["good ROE"]

    def test_dataframe_roundtrip(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 4.0, 3.0, 2.0, 1.0]}, index=dates)
        d = dataframe_to_dict(df)
        restored = dict_to_dataframe(d)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), restored.reset_index(drop=True))
        assert list(restored.columns) == ["A", "B"]

    def test_backward_compat_missing_fields(self):
        # Old cached dict missing newer optional fields
        d = {
            "symbol": "OLD",
            "name": "Old Corp",
            "price": 30.0,
            "market_cap": 500e6,
            "sector": "Tech",
            "country": "US",
            "is_etf": False,
            "is_actively_trading": True,
            "de_ratio": 0.5,
            "roe": 0.10,
            "gross_margin": 0.30,
            "revenue_growth": 0.05,
            "free_cash_flow": 10e6,
            "free_cash_flow_growth": 0.05,
        }
        stock = dict_to_stock_data(d)
        assert stock.symbol == "OLD"
        assert stock.pe_ratio is None
        assert stock.avg_volume is None

    def test_symbols_hash_deterministic(self):
        h1 = symbols_hash(["AAPL", "MSFT", "GOOG"])
        h2 = symbols_hash(["AAPL", "MSFT", "GOOG"])
        assert h1 == h2
        assert len(h1) == 8

    def test_symbols_hash_order_invariant(self):
        h1 = symbols_hash(["AAPL", "MSFT", "GOOG"])
        h2 = symbols_hash(["GOOG", "AAPL", "MSFT"])
        assert h1 == h2
