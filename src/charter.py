import logging
from datetime import datetime
from io import BytesIO

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from alpaca.trading.requests import GetPortfolioHistoryRequest

from src.broker import AlpacaBroker
from src.data import FMPClient

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def fetch_portfolio_history(broker: AlpacaBroker) -> pd.DataFrame | None:
    """Fetch daily portfolio equity history from Alpaca.

    Returns:
        DataFrame with DatetimeIndex and 'equity' column, or None on error.
    """
    try:
        request = GetPortfolioHistoryRequest(
            period="5A",
            timeframe="1D",
        )
        history = broker.client.get_portfolio_history(request)

        if not history or not history.timestamp:
            logger.warning("No portfolio history returned from Alpaca")
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(history.timestamp, unit="s", utc=True),
            "equity": [float(e) for e in history.equity],
        })
        df = df.set_index("date").sort_index()

        # Drop rows where equity is 0 (before any deposits)
        df = df[df["equity"] > 0]

        if df.empty:
            return None

        return df

    except Exception as e:
        logger.error(f"Error fetching portfolio history: {e}")
        return None


def fetch_benchmark_history(
    fmp: FMPClient, start_date: str, end_date: str
) -> pd.DataFrame | None:
    """Fetch SPY historical prices for the given date range.

    Returns:
        DataFrame with DatetimeIndex and 'close' column, or None on error.
    """
    return fmp.get_historical_prices("SPY", start_date, end_date)


def calculate_cumulative_returns(
    portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Calculate cumulative % returns for portfolio and benchmark.

    Both series are normalized to 0% at the first common date.

    Returns:
        DataFrame with 'portfolio' and 'spy' columns (cumulative % return),
        or None if insufficient data.
    """
    # Normalize both indices to date-only for joining
    portfolio_df = portfolio_df.copy()
    portfolio_df.index = portfolio_df.index.normalize()
    portfolio_df.index = portfolio_df.index.tz_localize(None)

    benchmark_df = benchmark_df.copy()
    benchmark_df.index = benchmark_df.index.normalize()

    # Join on common dates
    combined = portfolio_df[["equity"]].join(benchmark_df[["close"]], how="inner")

    if len(combined) < 2:
        logger.warning("Not enough overlapping dates for chart")
        return None

    initial_equity = combined["equity"].iloc[0]
    initial_spy = combined["close"].iloc[0]

    result = pd.DataFrame({
        "portfolio": (combined["equity"] / initial_equity - 1) * 100,
        "spy": (combined["close"] / initial_spy - 1) * 100,
    })

    return result


def generate_performance_chart(perf_data: pd.DataFrame) -> BytesIO:
    """Generate a cumulative performance chart as PNG.

    Args:
        perf_data: DataFrame with 'portfolio' and 'spy' columns (cumulative %).

    Returns:
        BytesIO containing the PNG image.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        perf_data.index, perf_data["portfolio"],
        color="#3B82F6", linewidth=2, label="Portfolio",
    )
    ax.plot(
        perf_data.index, perf_data["spy"],
        color="#F97316", linewidth=1.5, linestyle="--", label="SPY",
    )

    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Portfolio vs SPY")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    num_days = (perf_data.index[-1] - perf_data.index[0]).days
    if num_days < 30:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_performance_chart_image(
    broker: AlpacaBroker, fmp: FMPClient
) -> BytesIO | None:
    """Orchestrate chart generation: fetch data, compute returns, render chart.

    Returns:
        BytesIO containing PNG image, or None if generation fails.
    """
    try:
        portfolio_df = fetch_portfolio_history(broker)
        if portfolio_df is None or portfolio_df.empty:
            logger.warning("No portfolio history available for chart")
            return None

        start_date = portfolio_df.index[0].strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        benchmark_df = fetch_benchmark_history(fmp, start_date, end_date)
        if benchmark_df is None or benchmark_df.empty:
            logger.warning("No benchmark history available for chart")
            return None

        perf_data = calculate_cumulative_returns(portfolio_df, benchmark_df)
        if perf_data is None:
            return None

        return generate_performance_chart(perf_data)

    except Exception as e:
        logger.error(f"Error generating performance chart: {e}")
        return None
