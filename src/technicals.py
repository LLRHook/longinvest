import logging

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def compute_sma(prices: pd.Series, period: int = 200) -> float | None:
    """Compute Simple Moving Average for the given period.

    Args:
        prices: Series of closing prices (oldest first).
        period: SMA lookback period in days.

    Returns:
        SMA value, or None if not enough data.
    """
    if len(prices) < period:
        return None
    return float(prices.iloc[-period:].mean())


def compute_rsi(prices: pd.Series, period: int = 14) -> float | None:
    """Compute Relative Strength Index.

    Args:
        prices: Series of closing prices (oldest first).
        period: RSI lookback period in days.

    Returns:
        RSI value (0-100), or None if not enough data.
    """
    if len(prices) < period + 1:
        return None

    deltas = prices.diff().dropna()
    gains = deltas.where(deltas > 0, 0.0)
    losses = (-deltas).where(deltas < 0, 0.0)

    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    # Use exponential smoothing after the initial window
    for i in range(period, len(gains)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gains.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + losses.iloc[i]) / period

    last_avg_gain = avg_gain.iloc[-1]
    last_avg_loss = avg_loss.iloc[-1]

    if last_avg_loss < 1e-10:
        return 100.0

    rs = last_avg_gain / last_avg_loss
    return float(100 - (100 / (1 + rs)))


def apply_technical_filters(
    prices_df: pd.DataFrame,
    sma_period: int | None = None,
    rsi_period: int = 14,
    rsi_overbought: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Filter stocks based on SMA trend and RSI overbought conditions.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.
        sma_period: Period for SMA trend filter. Defaults to Config.SMA_TREND_PERIOD.
        rsi_period: Period for RSI calculation.
        rsi_overbought: RSI threshold above which stock is considered overbought.
            Defaults to Config.RSI_OVERBOUGHT.

    Returns:
        Tuple of (filtered DataFrame, list of drop reason strings).
    """
    if sma_period is None:
        sma_period = Config.SMA_TREND_PERIOD
    if rsi_overbought is None:
        rsi_overbought = Config.RSI_OVERBOUGHT
    dropped = []
    keep_symbols = []

    for symbol in prices_df.columns:
        prices = prices_df[symbol].dropna()

        # SMA trend filter: price must be above SMA
        sma = compute_sma(prices, sma_period)
        if sma is not None and prices.iloc[-1] < sma:
            dropped.append(f"{symbol}: below SMA-{sma_period} (${prices.iloc[-1]:.2f} < ${sma:.2f})")
            continue

        # RSI filter: skip overbought stocks
        rsi = compute_rsi(prices, rsi_period)
        if rsi is not None and rsi > rsi_overbought:
            dropped.append(f"{symbol}: overbought RSI={rsi:.1f} (>{rsi_overbought})")
            continue

        keep_symbols.append(symbol)

    filtered_df = prices_df[keep_symbols] if keep_symbols else pd.DataFrame()
    return filtered_df, dropped


def compute_momentum_scores(
    prices_df: pd.DataFrame,
    lookback: int = 63,
) -> dict[str, float]:
    """Compute volatility-adjusted momentum scores for each symbol.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.
        lookback: Lookback period in trading days (default 63 = ~3 months).

    Returns:
        Dict of {symbol: momentum_score} where score is 0-100 percentile rank.
    """
    if prices_df.empty or len(prices_df) < lookback:
        return {}

    raw_scores: dict[str, float] = {}

    for symbol in prices_df.columns:
        prices = prices_df[symbol].dropna()
        if len(prices) < lookback:
            continue

        recent = prices.iloc[-lookback:]
        ret = (recent.iloc[-1] / recent.iloc[0]) - 1
        daily_returns = recent.pct_change().dropna()
        vol = daily_returns.std()

        # Volatility-adjusted momentum (mini Sharpe)
        raw_scores[symbol] = ret / vol if vol > 1e-10 else 0.0

    if not raw_scores:
        return {}

    # Percentile rank across candidates (0-100)
    sorted_symbols = sorted(raw_scores, key=lambda s: raw_scores[s])
    n = len(sorted_symbols)
    return {
        sym: (rank / (n - 1)) * 100 if n > 1 else 50.0
        for rank, sym in enumerate(sorted_symbols)
    }


def compute_volume_signals(
    prices_df: pd.DataFrame,
    avg_period: int = 50,
    surge_threshold: float = 2.0,
) -> dict[str, bool]:
    """Detect stocks with recent volume surges (institutional interest).

    Args:
        prices_df: DataFrame that may contain 'volume' data via MultiIndex or
            separate volume columns.
        avg_period: Period for average volume calculation.
        surge_threshold: Multiplier above average to flag as surge.

    Returns:
        Dict of {symbol: has_volume_surge}.
    """
    signals: dict[str, bool] = {}

    for symbol in prices_df.columns:
        if symbol == "volume":
            continue
        # Volume data may be in a separate column if returned alongside close
        # For now, we check if there is a volume column pattern
        signals[symbol] = False

    return signals


def compute_relative_strength(
    prices_df: pd.DataFrame,
    periods: list[int] | None = None,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Compute multi-timeframe relative strength scores.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.
        periods: Lookback periods in trading days (default [21, 63, 126]).
        weights: Weights for each period (default [0.4, 0.35, 0.25]).

    Returns:
        Dict of {symbol: composite_relative_strength_score} (0-100).
    """
    if periods is None:
        periods = [21, 63, 126]
    if weights is None:
        weights = [0.4, 0.35, 0.25]

    if prices_df.empty:
        return {}

    symbols = list(prices_df.columns)
    min_period = max(periods)
    if len(prices_df) < min_period:
        return {}

    # Compute returns for each period
    period_returns: dict[int, dict[str, float]] = {}
    for period in periods:
        returns: dict[str, float] = {}
        for symbol in symbols:
            prices = prices_df[symbol].dropna()
            if len(prices) >= period:
                returns[symbol] = (prices.iloc[-1] / prices.iloc[-period]) - 1
        period_returns[period] = returns

    # Percentile rank for each period
    period_ranks: dict[int, dict[str, float]] = {}
    for period in periods:
        returns = period_returns[period]
        if not returns:
            continue
        sorted_syms = sorted(returns, key=lambda s: returns[s])
        n = len(sorted_syms)
        period_ranks[period] = {
            sym: (rank / (n - 1)) * 100 if n > 1 else 50.0
            for rank, sym in enumerate(sorted_syms)
        }

    # Weighted composite score
    composite: dict[str, float] = {}
    for symbol in symbols:
        total_weight = 0.0
        weighted_rank = 0.0
        for period, weight in zip(periods, weights):
            ranks = period_ranks.get(period, {})
            if symbol in ranks:
                weighted_rank += weight * ranks[symbol]
                total_weight += weight
        if total_weight > 0:
            composite[symbol] = weighted_rank / total_weight

    return composite
