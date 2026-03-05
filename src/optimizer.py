import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StockStats:
    symbol: str
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def compute_individual_stats(prices_df: pd.DataFrame) -> list[StockStats]:
    """Compute performance stats for each stock in the price DataFrame.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices

    Returns:
        List of StockStats sorted by Sharpe ratio (descending)
    """
    stats = []

    for symbol in prices_df.columns:
        prices = prices_df[symbol].dropna()

        if len(prices) < 2:
            continue

        daily_returns = prices.pct_change().dropna()
        adr = daily_returns.mean()
        annual_return = (1 + adr) ** 252 - 1
        annual_volatility = daily_returns.std(ddof=1) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 1e-10 else 0.0

        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        stats.append(
            StockStats(
                symbol=symbol,
                annual_return=annual_return,
                annual_volatility=annual_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
            )
        )

    stats.sort(key=lambda s: s.sharpe_ratio, reverse=True)
    return stats


def apply_momentum_tilt(
    allocations: dict[str, float],
    momentum_scores: dict[str, float],
    tilt_factor: float,
) -> dict[str, float]:
    """Tilt allocations toward momentum leaders.

    Args:
        allocations: Base allocations {symbol: dollar_amount}.
        momentum_scores: Momentum percentile scores {symbol: 0-100}.
        tilt_factor: How much to tilt (e.g. 0.20 = 20%).

    Returns:
        Adjusted allocations that preserve the original total budget.
    """
    if not allocations or not momentum_scores:
        return dict(allocations)

    original_total = sum(allocations.values())
    if original_total <= 0:
        return dict(allocations)

    adjusted = {}
    for sym, amount in allocations.items():
        pctl = momentum_scores.get(sym, 50.0)
        factor = 1.0 + tilt_factor * (pctl / 100.0 - 0.5)
        adjusted[sym] = amount * factor

    # Re-normalize to preserve total budget
    adjusted_total = sum(adjusted.values())
    if adjusted_total > 0:
        scale = original_total / adjusted_total
        adjusted = {sym: amt * scale for sym, amt in adjusted.items()}

    return adjusted
