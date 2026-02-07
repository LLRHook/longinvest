import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    allocations: dict[str, float]  # symbol -> percentage (only >0 allocations)
    sharpe_ratio: float
    expected_annual_return: float
    annual_volatility: float


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
        annual_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
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


def _neg_sharpe_ratio(allocs: np.ndarray, normed_prices: pd.DataFrame) -> float:
    """Compute negative Sharpe ratio for minimization.

    Args:
        allocs: Array of allocations (must sum to 1)
        normed_prices: DataFrame of normalized prices (first row = 1.0)

    Returns:
        Negative Sharpe ratio (for minimization)
    """
    port_val = (normed_prices * allocs).sum(axis=1)
    daily_returns = (port_val.iloc[1:].values / port_val.iloc[:-1].values) - 1

    adr = daily_returns.mean()
    sddr = daily_returns.std(ddof=1)

    if sddr < 1e-10:
        return 0.0

    return -(np.sqrt(252) * adr / sddr)


def _compute_portfolio_stats(
    prices_df: pd.DataFrame, allocs: np.ndarray
) -> tuple[float, float, float]:
    """Compute portfolio statistics.

    Returns:
        Tuple of (sharpe_ratio, annual_return, annual_volatility)
    """
    normed = prices_df / prices_df.iloc[0]
    port_val = (normed * allocs).sum(axis=1)
    daily_returns = port_val.pct_change().dropna()

    adr = daily_returns.mean()
    sddr = daily_returns.std()

    sharpe = np.sqrt(252) * adr / sddr if sddr > 1e-10 else 0.0
    annual_return = adr * 252
    annual_vol = sddr * np.sqrt(252)

    return sharpe, annual_return, annual_vol


def apply_minimum_threshold(allocs: np.ndarray, threshold: float) -> np.ndarray:
    """Set allocations below threshold to 0, redistribute remainder proportionally.

    Args:
        allocs: Array of allocations
        threshold: Minimum allocation threshold (e.g., 0.05 for 5%)

    Returns:
        Cleaned allocations array summing to 1.0
    """
    allocs = np.where(allocs < threshold, 0.0, allocs)
    total = allocs.sum()
    if total > 0:
        allocs = allocs / total
    return allocs


def _apply_position_cap(allocs: np.ndarray, max_pct: float) -> np.ndarray:
    """Clamp each allocation to max_pct, redistribute excess proportionally."""
    capped = np.minimum(allocs, max_pct)
    excess = allocs.sum() - capped.sum()

    if excess < 1e-6:
        return capped

    # Redistribute excess to uncapped positions
    uncapped_mask = capped < max_pct
    uncapped_total = capped[uncapped_mask].sum()

    if uncapped_total > 0:
        redistribution = excess * (capped[uncapped_mask] / uncapped_total)
        capped[uncapped_mask] += redistribution

    # Re-normalize
    total = capped.sum()
    if total > 0:
        capped = capped / total

    # Iteratively cap again if redistribution caused new breaches
    if np.any(capped > max_pct + 1e-6):
        return _apply_position_cap(capped, max_pct)

    return capped


def optimize_allocations(
    prices_df: pd.DataFrame,
    min_allocation: float = 0.05,
    max_position_pct: float | None = None,
    sector_map: dict[str, str] | None = None,
    max_sector_pct: float | None = None,
) -> OptimizationResult:
    """Optimize portfolio allocations to maximize Sharpe ratio.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices
        min_allocation: Minimum allocation threshold (allocations below this are zeroed)
        max_position_pct: Max allocation per single position (e.g. 0.15 for 15%)
        sector_map: Optional mapping of symbol -> sector name
        max_sector_pct: Max combined allocation per sector (e.g. 0.40 for 40%)

    Returns:
        OptimizationResult with allocations, Sharpe ratio, and other stats
    """
    symbols = list(prices_df.columns)
    n = len(symbols)

    if n == 0:
        logger.error("No symbols provided for optimization")
        return OptimizationResult(
            allocations={},
            sharpe_ratio=0.0,
            expected_annual_return=0.0,
            annual_volatility=0.0,
        )

    if n == 1:
        logger.warning("Only one symbol - using 100% allocation")
        sharpe, ann_ret, ann_vol = _compute_portfolio_stats(
            prices_df, np.array([1.0])
        )
        return OptimizationResult(
            allocations={symbols[0]: 1.0},
            sharpe_ratio=sharpe,
            expected_annual_return=ann_ret,
            annual_volatility=ann_vol,
        )

    normed_prices = prices_df / prices_df.iloc[0]
    init_allocs = np.ones(n) / n

    # Per-position upper bound: min of 1.0 and max_position_pct
    upper = max_position_pct if max_position_pct else 1.0
    bounds = tuple((0.0, upper) for _ in range(n))

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    # Sector diversification constraints
    if sector_map and max_sector_pct:
        sectors: dict[str, list[int]] = {}
        for i, sym in enumerate(symbols):
            sec = sector_map.get(sym, "Unknown")
            sectors.setdefault(sec, []).append(i)

        for sector_name, indices in sectors.items():
            if len(indices) < n:  # Only constrain if it doesn't cover all stocks
                idx = indices  # capture in closure
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x, _idx=idx: max_sector_pct - sum(x[i] for i in _idx),
                })

    try:
        result = minimize(
            _neg_sharpe_ratio,
            init_allocs,
            args=(normed_prices,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 1000},
        )

        allocs = result.x
        allocs = np.where(np.abs(allocs) < 1e-6, 0.0, allocs)
        allocs = np.clip(allocs, 0.0, 1.0)
        allocs = allocs / np.sum(allocs)

    except Exception as e:
        logger.error(f"Optimization failed: {e}. Using equal weights.")
        allocs = init_allocs

    allocs = apply_minimum_threshold(allocs, min_allocation)

    # Apply position cap as post-processing safety net
    if max_position_pct:
        allocs = _apply_position_cap(allocs, max_position_pct)

    if allocs.sum() < 1e-6:
        logger.warning("All allocations below threshold. Using equal weights.")
        allocs = np.ones(n) / n

    sharpe, ann_ret, ann_vol = _compute_portfolio_stats(prices_df, allocs)

    allocations = {
        symbol: float(alloc)
        for symbol, alloc in zip(symbols, allocs)
        if alloc > 0
    }

    logger.info(
        f"Optimization complete: {len(allocations)} positions, "
        f"Sharpe={sharpe:.2f}, Return={ann_ret:.1%}, Vol={ann_vol:.1%}"
    )

    return OptimizationResult(
        allocations=allocations,
        sharpe_ratio=sharpe,
        expected_annual_return=ann_ret,
        annual_volatility=ann_vol,
    )
