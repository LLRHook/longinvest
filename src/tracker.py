import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "trade_history.json"


class TradeTracker:
    """Tracks trade history with JSON file persistence."""

    def __init__(self, filepath: Path = HISTORY_FILE):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.filepath.exists():
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load trade history: {e}")
            return []

    def _save(self) -> None:
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2, default=str)

    def record_buy(
        self, symbol: str, price: float, quantity: float, reason: str = "screening"
    ) -> None:
        self._history.append({
            "symbol": symbol,
            "action": "buy",
            "price": price,
            "quantity": quantity,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
        })
        self._save()

    def record_sell(
        self, symbol: str, price: float, quantity: float, reason: str = ""
    ) -> None:
        self._history.append({
            "symbol": symbol,
            "action": "sell",
            "price": price,
            "quantity": quantity,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
        })
        self._save()

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def compute_stats(self) -> dict[str, Any]:
        """Compute trade statistics from history.

        Returns dict with total_trades, buys, sells, win_rate, avg_win, avg_loss.
        """
        buys: dict[str, list[dict]] = {}
        completed_trades: list[float] = []

        for trade in self._history:
            sym = trade["symbol"]
            if trade["action"] == "buy":
                buys.setdefault(sym, []).append(trade)
            elif trade["action"] == "sell":
                # Match against earliest buy for this symbol
                if sym in buys and buys[sym]:
                    buy = buys[sym].pop(0)
                    pl_pct = (trade["price"] / buy["price"]) - 1 if buy["price"] > 0 else 0
                    completed_trades.append(pl_pct)

        total = len(completed_trades)
        if total == 0:
            return {
                "total_trades": len(self._history),
                "completed_round_trips": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
            }

        wins = [t for t in completed_trades if t > 0]
        losses = [t for t in completed_trades if t <= 0]

        return {
            "total_trades": len(self._history),
            "completed_round_trips": total,
            "win_rate": round(len(wins) / total * 100, 1),
            "avg_win": round(sum(wins) / len(wins) * 100, 1) if wins else 0,
            "avg_loss": round(sum(losses) / len(losses) * 100, 1) if losses else 0,
        }
