import logging
from datetime import date, datetime
from io import BytesIO

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from alpaca.trading.requests import GetPortfolioHistoryRequest

from src.broker import AlpacaBroker
from src.data import FMPClient

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

COLORS = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "text": "#e6edf3",
    "text_secondary": "#8b949e",
    "grid": "#21262d",
    "blue": "#58a6ff",
    "gray": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "amber": "#d29922",
}


def _style_axis(ax):
    """Apply dark theme to an axis."""
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.grid(True, alpha=0.15, color=COLORS["grid"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(COLORS["grid"])


def fetch_portfolio_history(broker: AlpacaBroker) -> pd.DataFrame | None:
    """Fetch daily portfolio equity history from Alpaca.

    Returns:
        DataFrame with DatetimeIndex and 'equity' column, or None on error.
    """
    try:
        request = GetPortfolioHistoryRequest(
            period="5A",
            timeframe="1D",
            date_end=date.today(),
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

        logger.debug(f"Portfolio history: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")
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
    df = fmp.get_historical_prices("SPY", start_date, end_date)
    if df is not None and not df.empty:
        logger.debug(f"Benchmark history: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")
    return df


def calculate_cumulative_returns(
    portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Calculate cumulative % returns for portfolio and benchmark.

    Both series are normalized to 0% at the first common date.
    Uses total portfolio equity (including cash) since Alpaca does not provide
    historical cash breakdowns needed for invested-capital-only returns.

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

    logger.debug(f"Combined chart data: {combined.index[0]} to {combined.index[-1]} ({len(combined)} rows)")

    initial_equity = combined["equity"].iloc[0]
    initial_spy = combined["close"].iloc[0]

    result = pd.DataFrame({
        "portfolio": (combined["equity"] / initial_equity - 1) * 100,
        "spy": (combined["close"] / initial_spy - 1) * 100,
    })

    return result


def generate_performance_chart(
    perf_data: pd.DataFrame,
    report: dict | None = None,
    portfolio_df: pd.DataFrame | None = None,
    benchmark_df: pd.DataFrame | None = None,
) -> BytesIO:
    """Generate a dark-themed 3-panel performance chart as PNG.

    Panels:
        1. Cumulative returns (portfolio vs SPY) with stat box
        2. Drawdown (portfolio + SPY overlay)
        3. Rolling 60-day Sharpe ratio (if portfolio_df available with >=60 days)

    Args:
        perf_data: DataFrame with 'portfolio' and 'spy' columns (cumulative %).
        report: Daily report dict with 'risk_metrics' for stat box.
        portfolio_df: Raw portfolio equity DataFrame for rolling Sharpe.
        benchmark_df: Raw benchmark DataFrame for SPY drawdown.

    Returns:
        BytesIO containing the PNG image.
    """
    # Determine if we can show rolling Sharpe panel
    show_sharpe = False
    rolling_sharpe = None
    if portfolio_df is not None and len(portfolio_df) >= 60:
        daily_ret = portfolio_df["equity"].pct_change().dropna()
        if len(daily_ret) >= 60:
            roll_mean = daily_ret.rolling(60).mean()
            roll_std = daily_ret.rolling(60).std()
            rolling_sharpe = (roll_mean / roll_std) * np.sqrt(252)
            rolling_sharpe = rolling_sharpe.dropna()
            if len(rolling_sharpe) > 0:
                show_sharpe = True

    # Layout
    if show_sharpe:
        height_ratios = [3, 1.3, 1]
        n_panels = 3
    else:
        height_ratios = [3, 1]
        n_panels = 2

    fig = plt.figure(figsize=(13, 7.5), dpi=150, facecolor=COLORS["bg"])
    gs = fig.add_gridspec(n_panels, 1, height_ratios=height_ratios, hspace=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    ax_perf = axes[0]
    ax_dd = axes[1]
    ax_sharpe = axes[2] if show_sharpe else None
    bottom_ax = axes[-1]

    for ax in axes:
        _style_axis(ax)

    # ── Panel 1: Cumulative Returns ──
    ax_perf.plot(
        perf_data.index, perf_data["portfolio"],
        color=COLORS["blue"], linewidth=2, label="Portfolio",
    )
    ax_perf.plot(
        perf_data.index, perf_data["spy"],
        color=COLORS["gray"], linewidth=1.5, label="S&P 500",
    )

    # Green/red fill between
    ax_perf.fill_between(
        perf_data.index, perf_data["portfolio"], perf_data["spy"],
        where=perf_data["portfolio"] >= perf_data["spy"],
        interpolate=True, alpha=0.1, color=COLORS["green"],
    )
    ax_perf.fill_between(
        perf_data.index, perf_data["portfolio"], perf_data["spy"],
        where=perf_data["portfolio"] < perf_data["spy"],
        interpolate=True, alpha=0.1, color=COLORS["red"],
    )

    ax_perf.axhline(y=0, color=COLORS["gray"], linewidth=0.5, alpha=0.5)

    # End-of-line labels
    for series, label, color in [
        ("portfolio", "Portfolio", COLORS["blue"]),
        ("spy", "S&P 500", COLORS["gray"]),
    ]:
        final_val = perf_data[series].iloc[-1]
        sign = "+" if final_val >= 0 else ""
        ax_perf.annotate(
            f" {sign}{final_val:.1f}%", xy=(perf_data.index[-1], final_val),
            fontsize=8, color=color, va="center", fontweight="bold",
        )

    # Stat box
    stat_lines = []
    port_ret = perf_data["portfolio"].iloc[-1]
    spy_ret = perf_data["spy"].iloc[-1]
    stat_lines.append(f"Return  {port_ret:+.1f}%  vs  SPY {spy_ret:+.1f}%")

    risk = report.get("risk_metrics", {}) if report else {}
    if risk:
        if "sharpe" in risk:
            stat_lines.append(f"Sharpe  {risk['sharpe']:.2f}")
        if "sortino" in risk:
            stat_lines.append(f"Sortino {risk['sortino']:.2f}")
        if "alpha" in risk:
            stat_lines.append(f"Alpha   {risk['alpha']:+.1f}%")
        if "beta" in risk:
            stat_lines.append(f"Beta    {risk['beta']:.2f}")
        if "win_rate" in risk:
            stat_lines.append(f"Win Rate {risk['win_rate']:.0f}%")

    if stat_lines:
        stat_text = "\n".join(stat_lines)
        ax_perf.text(
            0.02, 0.97, stat_text, transform=ax_perf.transAxes,
            fontsize=7.5, fontfamily="monospace", color=COLORS["text"],
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor=COLORS["bg"],
                edgecolor=COLORS["grid"], alpha=0.85,
            ),
        )

    ax_perf.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.55),
        fontsize=8, framealpha=0.6,
        facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )
    ax_perf.set_ylabel("Cumulative Return", color=COLORS["text_secondary"], fontsize=9)
    ax_perf.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%" if x != 0 else "0%")
    )

    # ── Panel 2: Drawdown ──
    cumulative = 1 + perf_data["portfolio"] / 100
    port_dd = (cumulative / cumulative.cummax() - 1) * 100

    ax_dd.fill_between(perf_data.index, port_dd, 0, color=COLORS["red"], alpha=0.3)
    ax_dd.plot(perf_data.index, port_dd, color=COLORS["red"], linewidth=1, label="Portfolio")

    # SPY drawdown overlay
    spy_cum = 1 + perf_data["spy"] / 100
    spy_dd = (spy_cum / spy_cum.cummax() - 1) * 100
    ax_dd.plot(perf_data.index, spy_dd, color=COLORS["gray"], linewidth=1, alpha=0.7, label="S&P 500")

    # Max drawdown annotation
    max_dd_idx = port_dd.idxmin()
    max_dd_val = port_dd.loc[max_dd_idx]
    ax_dd.annotate(
        f" {max_dd_val:.1f}%", xy=(max_dd_idx, max_dd_val),
        xytext=(max_dd_idx, max_dd_val * 0.6),
        fontsize=7.5, color=COLORS["red"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=0.8),
    )

    ax_dd.set_ylim(top=0)
    ax_dd.set_ylabel("Drawdown", color=COLORS["text_secondary"], fontsize=9)
    ax_dd.legend(
        loc="lower left", fontsize=7, framealpha=0.6,
        facecolor=COLORS["panel"], edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
    )

    # ── Panel 3: Rolling Sharpe ──
    if ax_sharpe is not None and rolling_sharpe is not None:
        ax_sharpe.plot(
            rolling_sharpe.index, rolling_sharpe.values,
            color=COLORS["amber"], linewidth=1.2,
        )
        ax_sharpe.axhline(y=0, color=COLORS["gray"], linewidth=0.5, linestyle="--", alpha=0.5)
        ax_sharpe.set_ylabel("Rolling Sharpe (60d)", color=COLORS["text_secondary"], fontsize=9)

    # ── Shared x-axis formatting ──
    # Hide x-tick labels on all panels except bottom
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    num_days = (perf_data.index[-1] - perf_data.index[0]).days
    if num_days < 60:
        bottom_ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    elif num_days < 365:
        bottom_ax.xaxis.set_major_locator(mdates.MonthLocator())
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.setp(bottom_ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "Portfolio vs S&P 500", color=COLORS["text"],
        fontsize=14, fontweight="bold", y=0.98,
    )

    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=150,
        facecolor=COLORS["bg"], bbox_inches="tight", pad_inches=0.3,
    )
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_performance_chart_image(
    broker: AlpacaBroker, fmp: FMPClient,
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

        return generate_performance_chart(
            perf_data,
            portfolio_df=portfolio_df,
            benchmark_df=benchmark_df,
        )

    except Exception as e:
        logger.error(f"Error generating performance chart: {e}")
        return None
