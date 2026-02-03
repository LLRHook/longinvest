import logging
from datetime import datetime, timedelta
from typing import Any

from src.broker import AlpacaBroker, AccountStatus
from src.data import FMPClient

logger = logging.getLogger(__name__)


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
) -> dict[str, Any]:
    """Generate a daily performance report using Alpaca data.

    Args:
        broker: AlpacaBroker instance (created if None)
        fmp: FMPClient instance (created if None)

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
