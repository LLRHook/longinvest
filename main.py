#!/usr/bin/env python3
import argparse
import logging
import sys
from datetime import datetime, timedelta

from config import Config
from src.broker import AlpacaBroker
from src.cache import (
    CacheManager,
    dataframe_to_dict,
    dict_to_dataframe,
    dict_to_optimization_result,
    dict_to_scored_stock,
    optimization_result_to_dict,
    scored_stock_to_dict,
    symbols_hash,
)
from src.data import FMPClient
from src.charter import generate_performance_chart_image
from src.notifier import (
    format_performance_embed,
    send_discord_notification,
    send_discord_notification_with_chart,
)
from src.optimizer import compute_individual_stats, optimize_allocations
from src.reporter import format_console_report, generate_daily_report
from src.strategy import GrowthStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_clear_cache() -> int:
    """Clear all cached data."""
    cache = CacheManager()
    deleted = cache.clear()
    print(f"Cleared {deleted} cache files.")
    return 0


def cmd_report(dry_run: bool = False) -> int:
    """Generate and send daily performance report."""
    print("\n=== Generating Daily Report ===")

    broker = AlpacaBroker()
    fmp = FMPClient()

    report = generate_daily_report(broker=broker, fmp=fmp)

    # Always show console output
    print(format_console_report(report))

    # Generate performance chart
    print("Generating performance chart...")
    chart_image = generate_performance_chart_image(broker, fmp)
    if chart_image:
        print("Chart generated successfully.")
    else:
        print("Chart generation skipped (not enough history).")

    if dry_run:
        print("[DRY RUN - Discord notification not sent]")
        print("\nDiscord embed preview:")
        embed = format_performance_embed(report)
        # Strip emojis for console display (Windows compatibility)
        title = embed['title'].encode('ascii', 'ignore').decode('ascii').strip()
        desc = embed['description'].encode('ascii', 'ignore').decode('ascii')
        print(f"  Title: {title}")
        print(f"  Color: {'Green' if embed['color'] == 0x00FF00 else 'Red'}")
        print(f"  Description:\n    {desc.replace(chr(10), chr(10) + '    ')}")
        if chart_image:
            print("  Chart: [attached]")
        return 0

    # Send Discord notification
    if Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
        embed = format_performance_embed(report)
        if chart_image:
            success = send_discord_notification_with_chart(
                Config.DISCORD_WEBHOOK_URL, embed, chart_image
            )
        else:
            success = send_discord_notification(Config.DISCORD_WEBHOOK_URL, embed)
        if success:
            print("Discord notification sent.")
        else:
            print("Failed to send Discord notification.")
            return 1
    elif not Config.DISCORD_WEBHOOK_URL:
        print("Discord webhook URL not configured. Set DISCORD_WEBHOOK_URL in .env")
    elif not Config.ENABLE_NOTIFICATIONS:
        print("Notifications disabled. Set ENABLE_NOTIFICATIONS=true in .env")

    print(f"\nTotal API calls: {fmp.get_api_call_count()}")
    return 0


def cmd_status() -> int:
    """Show portfolio status."""
    broker = AlpacaBroker()
    status = broker.get_account_status()

    print("\n=== Account Status ===")
    print(f"Portfolio Value: ${status.portfolio_value:,.2f}")
    print(f"Cash: ${status.cash:,.2f}")
    print(f"Buying Power: ${status.buying_power:,.2f}")

    if status.positions:
        print(f"\n=== Positions ({len(status.positions)}/{Config.MAX_POSITIONS}) ===")
        for p in status.positions:
            pl_sign = "+" if p.unrealized_pl >= 0 else ""
            print(
                f"  {p.symbol}: {p.qty} shares @ ${p.avg_entry_price:.2f} "
                f"| Value: ${p.market_value:,.2f} "
                f"| P/L: {pl_sign}${p.unrealized_pl:,.2f} ({pl_sign}{p.unrealized_plpc:.2%})"
            )
    else:
        print("\nNo positions held.")

    return 0


def cmd_screen(force_refresh: bool = False) -> int:
    """Run screener and display results."""
    fmp = FMPClient()
    cache = CacheManager()
    strategy = GrowthStrategy(fmp, cache=cache, force_refresh=force_refresh)

    print("\n=== Running Growth Screener ===")
    if force_refresh:
        print("(forcing fresh data)")
    scored = strategy.screen()

    print(f"\n=== Top 20 Candidates ===")
    for i, s in enumerate(scored[:20], 1):
        print(f"\n{i}. {s.stock.symbol} - {s.stock.name}")
        print(f"   Score: {s.score:.1f}")
        print(f"   Market Cap: ${s.stock.market_cap / 1e9:.1f}B")
        print(f"   Price: ${s.stock.price:.2f}")
        for reason in s.reasons:
            print(f"   - {reason}")

    print(f"\nTotal API calls: {fmp.get_api_call_count()}")
    return 0


def cmd_dry_run(force_refresh: bool = False) -> int:
    """Full cycle without executing trades."""
    broker = AlpacaBroker()
    fmp = FMPClient()
    cache = CacheManager()
    strategy = GrowthStrategy(fmp, cache=cache, force_refresh=force_refresh)

    status = broker.get_account_status()
    held_symbols = {p.symbol for p in status.positions}

    print("\n=== Dry Run Mode ===")
    if force_refresh:
        print("(forcing fresh data)")
    print(f"Current positions: {len(held_symbols)}/{Config.MAX_POSITIONS}")
    print(f"Investment budget: ${Config.INVESTMENT_BUDGET:,.2f}")

    # Step 1: Fundamental screening (uses cache)
    print("\n=== Step 1: Fundamental Screening ===")
    recommendations = strategy.get_buy_recommendations(
        existing_symbols=held_symbols,
        max_picks=Config.OPTIMIZER_CANDIDATES,
    )

    if not recommendations:
        print("No buy recommendations at this time.")
        return 0

    symbols = [r.stock.symbol for r in recommendations]
    print(f"Top {len(symbols)} candidates: {', '.join(symbols)}")

    # Step 2: Fetch historical prices (with caching)
    print("\n=== Step 2: Fetching Historical Prices ===")
    date_key = cache.get_date_key()
    sym_hash = symbols_hash(symbols)
    prices_cache_key = f"prices_{date_key}_{sym_hash}"

    prices_df = None
    if not force_refresh:
        cached_prices = cache.load("prices", prices_cache_key)
        if cached_prices:
            prices_df = dict_to_dataframe(cached_prices)
            print(f"  Using cached price data ({len(prices_df)} days, {len(prices_df.columns)} symbols)")

    if prices_df is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=Config.HISTORICAL_DAYS)
        from_str = start_date.strftime("%Y-%m-%d")
        to_str = end_date.strftime("%Y-%m-%d")

        prices_df = fmp.get_historical_prices_batch(
            symbols, from_str, to_str, Config.MIN_HISTORICAL_DAYS
        )

        if not prices_df.empty:
            cache.save("prices", prices_cache_key, dataframe_to_dict(prices_df))

    if prices_df.empty:
        print("Error: No historical price data available.")
        return 1

    print(f"Got {len(prices_df)} days of data for {len(prices_df.columns)} symbols")

    # Display per-stock historical performance
    stock_stats = compute_individual_stats(prices_df)
    print(f"\n{'Symbol':<8} {'1Y Return':>10} {'Volatility':>11} {'Sharpe':>7} {'Max DD':>8}")
    print("-" * 48)
    for s in stock_stats:
        ret_str = f"{s.annual_return:+.1%}"
        vol_str = f"{s.annual_volatility:.1%}"
        sharpe_str = f"{s.sharpe_ratio:.2f}"
        dd_str = f"{s.max_drawdown:.1%}"
        print(f"{s.symbol:<8} {ret_str:>10} {vol_str:>11} {sharpe_str:>7} {dd_str:>8}")

    # Step 3: Optimize allocations (with caching)
    print("\n=== Step 3: Portfolio Optimization ===")
    opt_cache_key = f"result_{date_key}_{sym_hash}"

    result = None
    if not force_refresh:
        cached_result = cache.load("optimization", opt_cache_key)
        if cached_result:
            result = dict_to_optimization_result(cached_result)
            print("  Using cached optimization result")

    if result is None:
        result = optimize_allocations(prices_df, Config.MIN_ALLOCATION_THRESHOLD)
        if result.allocations:
            cache.save("optimization", opt_cache_key, optimization_result_to_dict(result))

    if not result.allocations:
        print("Error: Optimization produced no allocations.")
        return 1

    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Expected Annual Return: {result.expected_annual_return:.1%}")
    print(f"Annual Volatility: {result.annual_volatility:.1%}")

    # Step 4: Display allocations
    print(f"\n=== Optimized Portfolio ({len(result.allocations)} positions) ===")
    for symbol, pct in sorted(result.allocations.items(), key=lambda x: -x[1]):
        amount = Config.INVESTMENT_BUDGET * pct
        stock = next((r.stock for r in recommendations if r.stock.symbol == symbol), None)
        name = stock.name if stock else ""
        print(f"  {symbol}: {pct:.1%} = ${amount:,.2f}  ({name})")

    total_pct = sum(result.allocations.values())
    print(f"\nTotal allocation: {total_pct:.1%}")
    print(f"Total investment: ${Config.INVESTMENT_BUDGET * total_pct:,.2f}")

    print(f"\nTotal API calls: {fmp.get_api_call_count()}")
    print("\n[DRY RUN - No trades executed]")
    return 0


def cmd_execute(force_refresh: bool = False) -> int:
    """Execute the full trading cycle with portfolio optimization."""
    broker = AlpacaBroker()
    fmp = FMPClient()
    cache = CacheManager()
    strategy = GrowthStrategy(fmp, cache=cache, force_refresh=force_refresh)

    status = broker.get_account_status()
    held_symbols = {p.symbol for p in status.positions}

    print("\n=== Executing Trading Cycle ===")
    if force_refresh:
        print("(forcing fresh data)")
    print(f"Current positions: {len(held_symbols)}/{Config.MAX_POSITIONS}")
    print(f"Investment budget: ${Config.INVESTMENT_BUDGET:,.2f}")

    # Step 1: Fundamental screening (uses cache)
    print("\n=== Step 1: Fundamental Screening ===")
    recommendations = strategy.get_buy_recommendations(
        existing_symbols=held_symbols,
        max_picks=Config.OPTIMIZER_CANDIDATES,
    )

    if not recommendations:
        print("No buy recommendations at this time.")
        return 0

    symbols = [r.stock.symbol for r in recommendations]
    print(f"Top {len(symbols)} candidates: {', '.join(symbols)}")

    # Step 2: Fetch historical prices (with caching)
    print("\n=== Step 2: Fetching Historical Prices ===")
    date_key = cache.get_date_key()
    sym_hash = symbols_hash(symbols)
    prices_cache_key = f"prices_{date_key}_{sym_hash}"

    prices_df = None
    if not force_refresh:
        cached_prices = cache.load("prices", prices_cache_key)
        if cached_prices:
            prices_df = dict_to_dataframe(cached_prices)
            print(f"  Using cached price data ({len(prices_df)} days, {len(prices_df.columns)} symbols)")

    if prices_df is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=Config.HISTORICAL_DAYS)
        from_str = start_date.strftime("%Y-%m-%d")
        to_str = end_date.strftime("%Y-%m-%d")

        prices_df = fmp.get_historical_prices_batch(
            symbols, from_str, to_str, Config.MIN_HISTORICAL_DAYS
        )

        if not prices_df.empty:
            cache.save("prices", prices_cache_key, dataframe_to_dict(prices_df))

    if prices_df.empty:
        print("Error: No historical price data available.")
        return 1

    print(f"Got {len(prices_df)} days of data for {len(prices_df.columns)} symbols")

    # Step 3: Optimize allocations (with caching)
    print("\n=== Step 3: Portfolio Optimization ===")
    opt_cache_key = f"result_{date_key}_{sym_hash}"

    result = None
    if not force_refresh:
        cached_result = cache.load("optimization", opt_cache_key)
        if cached_result:
            result = dict_to_optimization_result(cached_result)
            print("  Using cached optimization result")

    if result is None:
        result = optimize_allocations(prices_df, Config.MIN_ALLOCATION_THRESHOLD)
        if result.allocations:
            cache.save("optimization", opt_cache_key, optimization_result_to_dict(result))

    if not result.allocations:
        print("Error: Optimization produced no allocations.")
        return 1

    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Expected Annual Return: {result.expected_annual_return:.1%}")
    print(f"Annual Volatility: {result.annual_volatility:.1%}")

    # Step 4: Execute trades
    print(f"\n=== Executing Buys ({len(result.allocations)} positions) ===")
    orders_placed = 0

    for symbol, pct in sorted(result.allocations.items(), key=lambda x: -x[1]):
        amount = Config.INVESTMENT_BUDGET * pct
        print(f"\nBuying {symbol}: {pct:.1%} = ${amount:,.2f}...")
        order_id = broker.buy_notional(symbol, amount)
        if order_id:
            print(f"  Order placed: {order_id}")
            orders_placed += 1
        else:
            print(f"  Order FAILED")

    print(f"\nOrders placed: {orders_placed}/{len(result.allocations)}")
    print(f"Total API calls: {fmp.get_api_call_count()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Long-term growth investment bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show portfolio status",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Run screener only (no trades)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Full cycle without executing trades",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate and send daily performance report",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and fetch fresh data",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached files and exit",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle clear-cache before validation (no API keys needed)
    if args.clear_cache:
        return cmd_clear_cache()

    # Validate configuration
    missing = Config.validate()
    if missing:
        # Allow certain commands with partial config
        if not (args.status or args.screen or args.dry_run or args.report):
            print(f"Error: Missing required configuration: {', '.join(missing)}")
            print("Copy .env.example to .env and fill in your API keys.")
            return 1
        elif args.status and "ALPACA_API_KEY" in missing:
            print("Error: Alpaca keys required for --status")
            return 1
        elif args.screen and "FMP_API_KEY" in missing:
            print("Error: FMP key required for --screen")
            return 1
        elif args.report and ("ALPACA_API_KEY" in missing or "FMP_API_KEY" in missing):
            print("Error: Alpaca and FMP keys required for --report")
            return 1

    try:
        if args.status:
            return cmd_status()
        elif args.screen:
            return cmd_screen(force_refresh=args.force_refresh)
        elif args.dry_run and not args.report:
            return cmd_dry_run(force_refresh=args.force_refresh)
        elif args.report:
            return cmd_report(dry_run=args.dry_run)
        else:
            return cmd_execute(force_refresh=args.force_refresh)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
