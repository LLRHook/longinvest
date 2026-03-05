"""Deposit tracker for DCA simulation.

Tracks weekly deposits and SPY prices to enable fair performance
comparison: "What if I had just bought SPY each week instead?"
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DEPOSITS_FILE = os.path.join(os.path.dirname(__file__), "..", "deposits.json")


def _ensure_dir():
    os.makedirs(os.path.dirname(DEPOSITS_FILE), exist_ok=True)


def load_deposits() -> list[dict]:
    """Load deposit history from JSON file."""
    if not os.path.exists(DEPOSITS_FILE):
        return []
    with open(DEPOSITS_FILE) as f:
        return json.load(f)


def save_deposits(deposits: list[dict]):
    """Save deposit history to JSON file."""
    _ensure_dir()
    with open(DEPOSITS_FILE, "w") as f:
        json.dump(deposits, f, indent=2)


def record_deposit(amount: float, spy_price: float, date: str | None = None) -> dict:
    """Record a weekly deposit with the SPY price at time of deposit.

    Args:
        amount: Dollar amount deposited (e.g. 25000.0)
        spy_price: SPY price at time of deposit
        date: ISO date string (defaults to today)

    Returns:
        The new deposit record.
    """
    deposits = load_deposits()
    date = date or datetime.now().strftime("%Y-%m-%d")

    # Don't double-record for the same date
    if deposits and deposits[-1]["date"] == date:
        logger.info(f"Deposit already recorded for {date}, skipping")
        return deposits[-1]

    record = {
        "date": date,
        "amount": amount,
        "spy_price": spy_price,
    }
    deposits.append(record)
    save_deposits(deposits)
    logger.info(f"Recorded deposit: ${amount:,.2f} on {date} (SPY @ ${spy_price:.2f})")
    return record


def get_total_invested() -> float:
    """Get total amount invested across all deposits."""
    return sum(d["amount"] for d in load_deposits())


def compute_spy_benchmark(current_spy_price: float) -> dict:
    """Compute what the portfolio would be worth if every deposit bought SPY.

    Returns:
        {
            "total_invested": float,
            "spy_value": float,
            "spy_return_pct": float,
            "num_deposits": int,
            "spy_shares": float,
        }
    """
    deposits = load_deposits()
    if not deposits:
        return {
            "total_invested": 0,
            "spy_value": 0,
            "spy_return_pct": 0,
            "num_deposits": 0,
            "spy_shares": 0,
        }

    total_invested = 0.0
    total_spy_shares = 0.0

    for d in deposits:
        total_invested += d["amount"]
        if d["spy_price"] > 0:
            total_spy_shares += d["amount"] / d["spy_price"]

    spy_value = total_spy_shares * current_spy_price
    spy_return_pct = (spy_value / total_invested - 1) * 100 if total_invested > 0 else 0

    return {
        "total_invested": total_invested,
        "spy_value": spy_value,
        "spy_return_pct": spy_return_pct,
        "num_deposits": len(deposits),
        "spy_shares": total_spy_shares,
    }


def compute_portfolio_vs_spy(portfolio_value: float, current_spy_price: float) -> dict:
    """Compare actual portfolio performance against SPY DCA benchmark.

    Args:
        portfolio_value: Current value of all held positions (excluding cash).
        current_spy_price: Current SPY price.

    Returns:
        {
            "total_invested": float,
            "portfolio_value": float,
            "portfolio_return_pct": float,
            "spy_value": float,
            "spy_return_pct": float,
            "alpha_pct": float,
            "num_weeks": int,
        }
    """
    bench = compute_spy_benchmark(current_spy_price)
    total_invested = bench["total_invested"]

    portfolio_return_pct = (portfolio_value / total_invested - 1) * 100 if total_invested > 0 else 0

    return {
        "total_invested": total_invested,
        "portfolio_value": portfolio_value,
        "portfolio_return_pct": portfolio_return_pct,
        "spy_value": bench["spy_value"],
        "spy_return_pct": bench["spy_return_pct"],
        "alpha_pct": portfolio_return_pct - bench["spy_return_pct"],
        "num_weeks": bench["num_deposits"],
    }
