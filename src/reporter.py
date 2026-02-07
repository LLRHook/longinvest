import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.broker import AlpacaBroker, AccountStatus
from src.data import FMPClient

logger = logging.getLogger(__name__)


def compute_portfolio_risk_metrics(
    portfolio_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    risk_free_rate: float = 0.05,
) -> dict[str, Any]:
    """Compute risk-adjusted portfolio metrics from equity history.

    Args:
        portfolio_df: DataFrame with DatetimeIndex and 'equity' column.
        benchmark_df: DataFrame with DatetimeIndex and 'close' column (SPY). Optional.
        risk_free_rate: Annual risk-free rate for alpha/beta calculation.

    Returns:
        Dict with sharpe, sortino, max_drawdown, current_drawdown, win_rate,
        and optionally alpha, beta.
    """
    equity = portfolio_df["equity"]
    daily_returns = equity.pct_change().dropna()

    if len(daily_returns) < 2:
        return {}

    # Sharpe ratio (annualized)
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 1e-10 else 0.0

    # Sortino ratio (downside deviation only)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std > 1e-10 else 0.0

    # Drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative / rolling_max) - 1
    max_drawdown = float(drawdown.min())
    current_drawdown = float(drawdown.iloc[-1])

    # Win rate
    win_rate = float((daily_returns > 0).sum() / len(daily_returns))

    metrics = {
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown": round(max_drawdown * 100, 1),
        "current_drawdown": round(current_drawdown * 100, 1),
        "win_rate": round(win_rate * 100, 1),
    }

    # Alpha & Beta vs benchmark
    if benchmark_df is not None and not benchmark_df.empty:
        bench_returns = benchmark_df["close"].pct_change().dropna()

        # Align dates
        port_norm = portfolio_df.copy()
        port_norm.index = port_norm.index.normalize()
        if port_norm.index.tz is not None:
            port_norm.index = port_norm.index.tz_localize(None)

        bench_norm = benchmark_df.copy()
        bench_norm.index = bench_norm.index.normalize()

        port_daily = port_norm["equity"].pct_change().dropna()
        bench_daily = bench_norm["close"].pct_change().dropna()

        common = port_daily.index.intersection(bench_daily.index)
        if len(common) > 10:
            p = port_daily.loc[common]
            b = bench_daily.loc[common]

            cov = np.cov(p, b)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else 0.0

            portfolio_annual = float(p.mean() * 252)
            benchmark_annual = float(b.mean() * 252)
            alpha = portfolio_annual - (risk_free_rate + beta * (benchmark_annual - risk_free_rate))

            metrics["beta"] = round(beta, 2)
            metrics["alpha"] = round(alpha * 100, 1)

    return metrics


def compute_sector_exposure(
    positions: list[dict[str, Any]], fmp: FMPClient
) -> dict[str, float]:
    """Compute sector allocation percentages from held positions.

    Args:
        positions: Position details with 'symbol' and 'market_value'.
        fmp: FMP client for fetching sector info.

    Returns:
        Dict of sector -> percentage (sorted descending).
    """
    total_value = sum(p["market_value"] for p in positions)
    if total_value <= 0:
        return {}

    sector_values: dict[str, float] = {}
    for p in positions:
        try:
            profile = fmp.get_profile(p["symbol"])
            sector = profile.get("sector", "Unknown") if profile else "Unknown"
        except Exception:
            sector = "Unknown"
        sector_values[sector] = sector_values.get(sector, 0) + p["market_value"]

    exposure = {
        sector: round(value / total_value * 100, 1)
        for sector, value in sector_values.items()
    }
    return dict(sorted(exposure.items(), key=lambda x: -x[1]))


def get_benchmark_daily_return(fmp: FMPClient, symbol: str = "SPY") -> float | None:
    """Get the daily return for a benchmark symbol.

    Args:
        fmp: FMP client instance
        symbol: Benchmark symbol (default SPY for S&P 500)

    Returns:
        Daily return as decimal (e.g., 0.01 for 1%), or None on error
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        df = fmp.get_historical_prices(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        if df is None or len(df) < 2:
            logger.warning(f"Not enough historical data for {symbol}")
            return None

        recent_prices = df["close"].tail(2)
        prev_close = recent_prices.iloc[0]
        curr_close = recent_prices.iloc[1]

        daily_return = (curr_close - prev_close) / prev_close
        logger.debug(f"{symbol} daily return: {daily_return:.4f}")
        return float(daily_return)

    except Exception as e:
        logger.error(f"Error fetching benchmark data for {symbol}: {e}")
        return None


def generate_daily_report(
    broker: AlpacaBroker | None = None,
    fmp: FMPClient | None = None,
    portfolio_df: pd.DataFrame | None = None,
    benchmark_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate a daily performance report using Alpaca data.

    Args:
        broker: AlpacaBroker instance (created if None)
        fmp: FMPClient instance (created if None)
        portfolio_df: Pre-fetched portfolio history (avoids duplicate API call)
        benchmark_df: Pre-fetched benchmark history (avoids duplicate API call)

    Returns:
        Dict containing report data
    """
    if broker is None:
        broker = AlpacaBroker()
    if fmp is None:
        fmp = FMPClient()

    status = broker.get_account_status()

    # Calculate daily P/L from Alpaca's last_equity
    daily_pl = status.portfolio_value - status.last_equity
    daily_pl_pct = daily_pl / status.last_equity if status.last_equity > 0 else 0.0

    # Get benchmark return
    benchmark_return = get_benchmark_daily_return(fmp, "SPY")
    if benchmark_return is None:
        benchmark_return = 0.0
        logger.warning("Could not fetch SPY return, using 0.0")

    # Build position details using Alpaca's intraday data
    position_details = []
    for p in status.positions:
        position_details.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": p.market_value,
            "avg_entry_price": p.avg_entry_price,
            "current_price": p.current_price,
            "lastday_price": p.lastday_price,
            # Total P/L (from entry)
            "unrealized_pl": p.unrealized_pl,
            "unrealized_plpc": p.unrealized_plpc,
            # Today's P/L
            "change_today": p.change_today,  # Stock's daily % change
            "intraday_pl": p.unrealized_intraday_pl,
            "intraday_plpc": p.unrealized_intraday_plpc,
        })

    # Calculate total intraday P/L from positions (should match daily_pl)
    total_intraday_pl = sum(p.unrealized_intraday_pl for p in status.positions)

    # Compute risk metrics from portfolio history
    risk_metrics = {}
    if portfolio_df is not None and not portfolio_df.empty:
        risk_metrics = compute_portfolio_risk_metrics(
            portfolio_df, benchmark_df=benchmark_df
        )

    # Compute sector exposure
    sector_exposure = {}
    if position_details:
        sector_exposure = compute_sector_exposure(position_details, fmp)

    report = {
        "date": datetime.now().strftime("%b %d, %Y"),
        "portfolio": {
            "value": status.portfolio_value,
            "cash": status.cash,
            "last_equity": status.last_equity,
            "daily_pl": daily_pl,
            "daily_pl_pct": daily_pl_pct,
            "positions_pl": total_intraday_pl,
        },
        "benchmark": {
            "symbol": "SPY",
            "daily_change_pct": benchmark_return,
        },
        "positions": position_details,
        "summary": {
            "outperformance": daily_pl_pct - benchmark_return,
            "position_count": len(status.positions),
        },
        "risk_metrics": risk_metrics,
        "sector_exposure": sector_exposure,
    }

    return report


def format_console_report(report: dict[str, Any]) -> str:
    """Format report for console output.

    Args:
        report: Report data from generate_daily_report()

    Returns:
        Formatted string for console display
    """
    portfolio = report["portfolio"]
    benchmark = report["benchmark"]
    positions = report["positions"]
    summary = report["summary"]

    # Format daily P/L
    pl_sign = "+" if portfolio["daily_pl"] >= 0 else ""
    pct_sign = "+" if portfolio["daily_pl_pct"] >= 0 else ""

    lines = [
        "",
        f"=== Daily Portfolio Report - {report['date']} ===",
        "",
        f"Portfolio Value: ${portfolio['value']:,.2f}",
        f"  Yesterday:     ${portfolio['last_equity']:,.2f}",
        f"  Today's P/L:   {pl_sign}${portfolio['daily_pl']:,.2f} ({pct_sign}{portfolio['daily_pl_pct']:.2%})",
        f"  Cash:          ${portfolio['cash']:,.2f}",
        "",
        f"S&P 500 (SPY):   {'+' if benchmark['daily_change_pct'] >= 0 else ''}{benchmark['daily_change_pct']:.2%}",
        f"vs Benchmark:    {'+' if summary['outperformance'] >= 0 else ''}{summary['outperformance']:.2%} {'[OK]' if summary['outperformance'] >= 0 else '[BEHIND]'}",
        "",
    ]

    # Risk metrics
    risk = report.get("risk_metrics", {})
    if risk:
        lines.append("=== Risk Metrics ===")
        lines.append(f"  Sharpe Ratio:     {risk.get('sharpe', 'N/A')}")
        lines.append(f"  Sortino Ratio:    {risk.get('sortino', 'N/A')}")
        lines.append(f"  Max Drawdown:     {risk.get('max_drawdown', 'N/A')}%")
        lines.append(f"  Current Drawdown: {risk.get('current_drawdown', 'N/A')}%")
        lines.append(f"  Win Rate:         {risk.get('win_rate', 'N/A')}%")
        if "alpha" in risk:
            lines.append(f"  Alpha (ann.):     {'+' if risk['alpha'] >= 0 else ''}{risk['alpha']}%")
        if "beta" in risk:
            lines.append(f"  Beta:             {risk['beta']}")
        lines.append("")

    # Sector exposure
    sector = report.get("sector_exposure", {})
    if sector:
        lines.append("=== Sector Exposure ===")
        for name, pct in sector.items():
            lines.append(f"  {name}: {pct}%")
        lines.append("")

    if positions:
        lines.append(f"=== Positions ({len(positions)}) ===")
        lines.append(f"{'Symbol':<8} {'Value':>12} {'Today':>10} {'Total P/L':>12}")
        lines.append("-" * 46)

        # Sort by today's P/L
        sorted_positions = sorted(
            positions, key=lambda p: p["intraday_pl"], reverse=True
        )
        for p in sorted_positions:
            today_str = f"{'+' if p['intraday_pl'] >= 0 else ''}${p['intraday_pl']:,.2f}"
            total_str = f"{'+' if p['unrealized_pl'] >= 0 else ''}${p['unrealized_pl']:,.2f}"
            lines.append(
                f"{p['symbol']:<8} ${p['market_value']:>10,.2f} {today_str:>10} {total_str:>12}"
            )
    else:
        lines.append("No positions held.")

    lines.append("")
    return "\n".join(lines)
