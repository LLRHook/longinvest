import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


class CacheManager:
    """Manages JSON-based caching for pipeline data."""

    def __init__(self, cache_dir: str | None = None, ttl_hours: int | None = None):
        self.cache_dir = Path(cache_dir or Config.CACHE_DIR)
        self.ttl_hours = ttl_hours or Config.CACHE_TTL_HOURS
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create cache directory structure."""
        categories = ["screener", "fundamentals", "scored", "prices", "optimization"]
        for category in categories:
            (self.cache_dir / category).mkdir(parents=True, exist_ok=True)

    def _get_path(self, category: str, key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / category / f"{key}.json"

    def _is_valid(self, data: dict[str, Any]) -> bool:
        """Check if cached data is still valid (not expired)."""
        if data.get("version") != CACHE_VERSION:
            return False

        created_at = data.get("created_at")
        ttl_hours = data.get("ttl_hours", self.ttl_hours)

        if not created_at:
            return False

        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_hours = (now - created).total_seconds() / 3600

        return age_hours < ttl_hours

    def save(self, category: str, key: str, data: Any) -> None:
        """Save data to cache with metadata.

        Args:
            category: Cache category (screener, fundamentals, scored, prices, optimization)
            key: Unique key for this cache entry
            data: Data to cache (must be JSON-serializable or a supported type)
        """
        cache_entry = {
            "version": CACHE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ttl_hours": self.ttl_hours,
            "data": data,
        }

        path = self._get_path(category, key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, default=str)

        logger.debug(f"Cached {category}/{key}")

    def load(self, category: str, key: str) -> Any | None:
        """Load data from cache if valid.

        Args:
            category: Cache category
            key: Cache key

        Returns:
            Cached data if valid, None otherwise
        """
        path = self._get_path(category, key)

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                cache_entry = json.load(f)

            if not self._is_valid(cache_entry):
                logger.debug(f"Cache expired: {category}/{key}")
                return None

            logger.debug(f"Cache hit: {category}/{key}")
            return cache_entry.get("data")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load cache {category}/{key}: {e}")
            return None

    def clear(self, category: str | None = None) -> int:
        """Delete cache files.

        Args:
            category: Specific category to clear, or None for all

        Returns:
            Number of files deleted
        """
        deleted = 0

        if category:
            cat_path = self.cache_dir / category
            if cat_path.exists():
                for f in cat_path.glob("*.json"):
                    f.unlink()
                    deleted += 1
        else:
            for cat_path in self.cache_dir.iterdir():
                if cat_path.is_dir():
                    for f in cat_path.glob("*.json"):
                        f.unlink()
                        deleted += 1

        logger.info(f"Cleared {deleted} cache files")
        return deleted

    def get_date_key(self) -> str:
        """Get today's date as a cache key component."""
        return datetime.now().strftime("%Y-%m-%d")


def symbols_hash(symbols: list[str]) -> str:
    """Create a short hash for a list of symbols."""
    sorted_symbols = sorted(symbols)
    content = ",".join(sorted_symbols)
    return hashlib.md5(content.encode()).hexdigest()[:8]


def stock_data_to_dict(stock: Any) -> dict[str, Any]:
    """Convert StockData dataclass to dict."""
    return asdict(stock)


def dict_to_stock_data(d: dict[str, Any]) -> Any:
    """Convert dict back to StockData."""
    from src.data import StockData
    return StockData(**d)


def scored_stock_to_dict(scored: Any) -> dict[str, Any]:
    """Convert ScoredStock to dict."""
    return {
        "stock": stock_data_to_dict(scored.stock),
        "score": scored.score,
        "reasons": scored.reasons,
    }


def dict_to_scored_stock(d: dict[str, Any]) -> Any:
    """Convert dict back to ScoredStock."""
    from src.strategy import ScoredStock
    return ScoredStock(
        stock=dict_to_stock_data(d["stock"]),
        score=d["score"],
        reasons=d["reasons"],
    )


def optimization_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert OptimizationResult to dict."""
    return asdict(result)


def dict_to_optimization_result(d: dict[str, Any]) -> Any:
    """Convert dict back to OptimizationResult."""
    from src.optimizer import OptimizationResult
    return OptimizationResult(**d)


def dataframe_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    """Convert DataFrame to JSON-serializable dict."""
    return {
        "index": df.index.strftime("%Y-%m-%d").tolist(),
        "columns": df.columns.tolist(),
        "data": df.values.tolist(),
    }


def dict_to_dataframe(d: dict[str, Any]) -> pd.DataFrame:
    """Convert dict back to DataFrame."""
    df = pd.DataFrame(d["data"], columns=d["columns"])
    df.index = pd.to_datetime(d["index"])
    return df
