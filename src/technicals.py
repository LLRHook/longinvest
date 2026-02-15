import logging

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
