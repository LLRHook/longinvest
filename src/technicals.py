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


def compute_price_momentum_12_1(prices_df: pd.DataFrame) -> dict[str, float]:
    """Compute Jegadeesh-Titman 12-1 month momentum signal.

    Classic momentum factor: 252-day return skipping the most recent 21 days
    (excludes short-term reversal).

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.

    Returns:
        Dict of {symbol: raw_return} (not percentile-ranked).
    """
    min_days = 252 + 21  # Need 12 months + 1 month skip
    if prices_df.empty or len(prices_df) < min_days:
        return {}

    signals: dict[str, float] = {}

    for symbol in prices_df.columns:
        prices = prices_df[symbol].dropna()
        if len(prices) < min_days:
            continue

        # Price 12 months ago
        price_12m = prices.iloc[-(252 + 21)]
        # Price 1 month ago (skip recent month)
        price_1m = prices.iloc[-21]

        if price_12m > 0:
            signals[symbol] = (price_1m / price_12m) - 1

    return signals


def compute_momentum_percentiles(prices_df: pd.DataFrame) -> dict[str, float]:
    """Convert 12-1 momentum signals to percentile ranks across the universe.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.

    Returns:
        Dict of {symbol: percentile_rank} where rank is 0-100.
    """
    raw_signals = compute_price_momentum_12_1(prices_df)
    if not raw_signals:
        return {}

    series = pd.Series(raw_signals)
    # Use pandas rank() with percent=True for percentile ranking (0-1), scale to 0-100
    ranks = series.rank(pct=True) * 100
    return ranks.to_dict()


def compute_volume_signals(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute volume surge ratio for each symbol.

    The surge ratio is the latest volume divided by the VOLUME_SURGE_LOOKBACK-day
    average volume. A ratio > VOLUME_SURGE_THRESHOLD indicates institutional interest.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.
            Used only to determine the symbol universe if volume_df is provided.
        volume_df: DataFrame with Date index, columns = symbols, values = volume.
            Required for volume signal computation.

    Returns:
        Dict of {symbol: surge_ratio}.
    """
    if volume_df is None or volume_df.empty:
        return {}

    lookback = Config.VOLUME_SURGE_LOOKBACK
    signals: dict[str, float] = {}

    for symbol in volume_df.columns:
        vol = volume_df[symbol].dropna()
        if len(vol) < lookback + 1:
            continue

        avg_volume = vol.iloc[-(lookback + 1):-1].mean()
        if avg_volume < 1e-10:
            continue

        latest_volume = vol.iloc[-1]
        signals[symbol] = float(latest_volume / avg_volume)

    return signals


def compute_multi_tf_relative_strength(
    prices_df: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> dict[str, float]:
    """Compute multi-timeframe relative strength vs benchmark.

    For each timeframe in Config.RS_TIMEFRAMES, computes the stock return / benchmark
    return ratio, then creates a weighted composite score using Config.RS_WEIGHTS.
    The composite scores are percentile-ranked across the universe.

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.
        benchmark_prices: Series of benchmark (e.g. SPY) close prices with Date index.

    Returns:
        Dict of {symbol: composite_rs_percentile} where percentile is 0-100.
    """
    if prices_df.empty or benchmark_prices.empty:
        return {}

    timeframes = Config.RS_TIMEFRAMES
    weights = Config.RS_WEIGHTS
    composite_scores: dict[str, float] = {}

    for symbol in prices_df.columns:
        prices = prices_df[symbol].dropna()
        weighted_sum = 0.0
        total_weight = 0.0

        for tf, w in zip(timeframes, weights):
            if len(prices) < tf or len(benchmark_prices) < tf:
                continue

            stock_return = (prices.iloc[-1] / prices.iloc[-tf]) - 1
            bench_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[-tf]) - 1

            # Relative strength ratio: use difference to avoid division by zero
            if abs(bench_return) < 1e-10:
                rs = stock_return
            else:
                rs = stock_return / bench_return

            weighted_sum += w * rs
            total_weight += w

        if total_weight > 0:
            composite_scores[symbol] = weighted_sum / total_weight

    if not composite_scores:
        return {}

    # Percentile-rank the composite scores
    series = pd.Series(composite_scores)
    ranks = series.rank(pct=True) * 100
    return ranks.to_dict()


def compute_atr(prices_df: pd.DataFrame, period: int = 14) -> dict[str, float]:
    """Compute Average True Range using close-to-close approximation.

    Since we only have close prices (no high/low), the true range is approximated
    as the absolute daily change: abs(close[t] - close[t-1]).

    Args:
        prices_df: DataFrame with Date index, columns = symbols, values = prices.
        period: ATR lookback period in days.

    Returns:
        Dict of {symbol: atr_value}.
    """
    if prices_df.empty:
        return {}

    signals: dict[str, float] = {}

    for symbol in prices_df.columns:
        prices = prices_df[symbol].dropna()
        if len(prices) < period + 1:
            continue

        true_range = prices.diff().abs()
        atr = true_range.rolling(window=period, min_periods=period).mean()
        last_atr = atr.iloc[-1]

        if not np.isnan(last_atr):
            signals[symbol] = float(last_atr)

    return signals
