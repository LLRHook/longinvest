#!/usr/bin/env python3
import argparse
import logging
import sys
import time
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
from src.charter import (
    fetch_benchmark_history,
    fetch_portfolio_history,
    calculate_cumulative_returns,
    generate_performance_chart,
)
from src.notifier import (
    format_circuit_breaker_embed,
    format_performance_embed,
    format_rebalance_embed,
    format_screening_embed,
    format_sell_embed,
    send_discord_chart_message,
    send_discord_notification,
    send_discord_notification_with_chart,
)
from src.optimizer import compute_individual_stats, optimize_allocations
from src.reporter import format_console_report, generate_daily_report
from src.strategy import GrowthStrategy
from src.technicals import apply_technical_filters, compute_momentum_scores, compute_relative_strength
from src.tracker import TradeTracker

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

    # Get current account status for cash balance
    status = broker.get_account_status()
    cash_balance = status.cash

    # Fetch portfolio + benchmark history once (shared by reporter and charter)
    portfolio_df = fetch_portfolio_history(broker)
    benchmark_df = None
    if portfolio_df is not None and not portfolio_df.empty:
        start_date = portfolio_df.index[0].strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        benchmark_df = fetch_benchmark_history(fmp, start_date, end_date)

    report = generate_daily_report(
        broker=broker, fmp=fmp,
        portfolio_df=portfolio_df, benchmark_df=benchmark_df,
    )

    # Always show console output
    print(format_console_report(report))

    # Generate performance chart from pre-fetched data (invested capital only)
    print("Generating performance chart...")
    chart_image = None
    if portfolio_df is not None and benchmark_df is not None:
        perf_data = calculate_cumulative_returns(
            portfolio_df, benchmark_df, cash_balance=cash_balance,
        )
        if perf_data is not None:
            chart_image = generate_performance_chart(perf_data)
    if chart_image:
        print("Chart generated successfully.")
    else:
        print("Chart generation skipped (not enough history).")

    if dry_run:
        print("[DRY RUN - Discord notifications not sent]")
        embed = format_performance_embed(report)
        # Strip emojis for console display (Windows compatibility)
        title = embed['title'].encode('ascii', 'ignore').decode('ascii').strip()
        desc = embed['description'].encode('ascii', 'ignore').decode('ascii')
        print("\nMessage 1 - Daily Summary:")
        print(f"  Title: {title}")
        print(f"  Color: {'Green' if embed['color'] == 0x00FF00 else 'Red'}")
        print(f"  Description:\n    {desc.replace(chr(10), chr(10) + '    ')}")
        print("\nMessage 2 - Cumulative Growth Chart:")
        if chart_image:
            print("  Chart: [attached]")
        else:
            print("  Chart: [skipped - not enough history]")
        return 0

    # Send Discord notifications (two messages)
    if Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
        embed = format_performance_embed(report)

        # Message 1: Daily summary embed
        success = send_discord_notification(Config.DISCORD_WEBHOOK_URL, embed)
        if success:
            print("Discord message 1 (summary) sent.")
        else:
            print("Failed to send Discord summary.")
            return 1

        # Message 2: Cumulative growth chart
        if chart_image:
            time.sleep(2)
            chart_success = send_discord_chart_message(
                Config.DISCORD_WEBHOOK_URL, chart_image
            )
            if chart_success:
                print("Discord message 2 (chart) sent.")
            else:
                print("Failed to send Discord chart.")
        else:
            print("Chart skipped (not enough history).")
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
        print(f"\n=== Positions ({len(status.positions)}) ===")
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
    print(f"Current positions: {len(held_symbols)}")
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

    # Step 2b: Technical filters (SMA + RSI)
    print("\n=== Step 2b: Technical Filters ===")
    prices_df, dropped_reasons = apply_technical_filters(prices_df)
    for reason in dropped_reasons:
        print(f"  Dropped: {reason}")
    if prices_df.empty:
        print("Error: All stocks filtered out by technical analysis.")
        return 1
    print(f"Passed filters: {len(prices_df.columns)} symbols")

    # Compute momentum scores for optimizer tilt
    momentum_scores = compute_momentum_scores(prices_df)

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
    sym_hash_filtered = symbols_hash(list(prices_df.columns))
    opt_cache_key = f"result_{date_key}_{sym_hash_filtered}"

    # Build sector mapping for optimizer constraints
    sector_map = {}
    for r in recommendations:
        if r.stock.symbol in prices_df.columns:
            sector_map[r.stock.symbol] = r.stock.sector or "Unknown"

    result = None
    if not force_refresh:
        cached_result = cache.load("optimization", opt_cache_key)
        if cached_result:
            result = dict_to_optimization_result(cached_result)
            print("  Using cached optimization result")

    if result is None:
        result = optimize_allocations(
            prices_df,
            Config.MIN_ALLOCATION_THRESHOLD,
            max_position_pct=Config.MAX_SINGLE_POSITION_PCT,
            sector_map=sector_map,
            max_sector_pct=Config.MAX_SECTOR_ALLOCATION,
            momentum_scores=momentum_scores,
        )
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
    tracker = TradeTracker()

    status = broker.get_account_status()
    held_symbols = {p.symbol for p in status.positions}

    print("\n=== Executing Trading Cycle ===")
    if force_refresh:
        print("(forcing fresh data)")
    print(f"Current positions: {len(held_symbols)}")
    print(f"Investment budget: ${Config.INVESTMENT_BUDGET:,.2f}")

    # Circuit breaker: halt if portfolio or market is crashing
    portfolio_change = 0.0
    if status.last_equity > 0:
        portfolio_change = (status.portfolio_value - status.last_equity) / status.last_equity
    if portfolio_change < Config.CIRCUIT_BREAKER_PCT:
        msg = f"Circuit breaker triggered: portfolio down {portfolio_change:.2%} today (threshold: {Config.CIRCUIT_BREAKER_PCT:.0%})"
        print(f"\n{msg}")
        logger.warning(msg)
        if Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
            embed = format_circuit_breaker_embed(msg, portfolio_change)
            send_discord_notification(Config.DISCORD_WEBHOOK_URL, embed)
        return 0

    spy_quote = fmp.get_quote("SPY")
    if spy_quote:
        spy_prev = spy_quote.get("previousClose", 0)
        spy_price = spy_quote.get("price", 0)
        if spy_prev > 0:
            spy_change = (spy_price - spy_prev) / spy_prev
            if spy_change < Config.MARKET_CIRCUIT_BREAKER_PCT:
                msg = f"Market circuit breaker: SPY down {spy_change:.2%} today (threshold: {Config.MARKET_CIRCUIT_BREAKER_PCT:.0%})"
                print(f"\n{msg}")
                logger.warning(msg)
                if Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
                    embed = format_circuit_breaker_embed(msg, spy_change)
                    send_discord_notification(Config.DISCORD_WEBHOOK_URL, embed)
                return 0

    # Step 0: Review existing positions for fundamental degradation
    if held_symbols:
        print("\n=== Step 0: Position Review ===")
        sell_recs = strategy.get_sell_recommendations(held_symbols)
        for symbol, reason in sell_recs:
            print(f"  Selling {symbol}: {reason}")
            position = broker.get_position(symbol)
            order_id = broker.sell_all(symbol)
            if order_id:
                print(f"    Order placed: {order_id}")
                if position:
                    tracker.record_sell(
                        symbol=symbol,
                        price=position.current_price,
                        quantity=float(position.qty),
                        reason=reason,
                    )
                # Send sell notification
                if Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
                    embed = format_sell_embed(
                        symbol=symbol,
                        reason=reason,
                        entry_price=position.avg_entry_price if position else 0,
                        exit_price=position.current_price if position else 0,
                        pl=position.unrealized_pl if position else 0,
                        hold_days=None,
                    )
                    send_discord_notification(Config.DISCORD_WEBHOOK_URL, embed)
                held_symbols.discard(symbol)
            else:
                print(f"    Sell FAILED")
        if not sell_recs:
            print("  All positions pass fundamental review.")

    # Step 0b: Rebalance overweight positions
    if Config.ENABLE_REBALANCING and held_symbols:
        print("\n=== Step 0b: Rebalancing ===")
        rebalance_status = broker.get_account_status()
        portfolio_value = rebalance_status.portfolio_value
        if portfolio_value > 0:
            target_pct = 1.0 / max(len(rebalance_status.positions), 1)
            for pos in rebalance_status.positions:
                actual_pct = pos.market_value / portfolio_value
                drift = actual_pct - target_pct
                if drift > Config.REBALANCE_THRESHOLD:
                    trim_amount = drift * portfolio_value
                    print(f"  {pos.symbol}: {actual_pct:.1%} -> target {target_pct:.1%}, trimming ${trim_amount:,.2f}")
                    order_id = broker.sell_notional(pos.symbol, trim_amount)
                    if order_id:
                        print(f"    Trim order: {order_id}")
                        tracker.record_sell(
                            symbol=pos.symbol,
                            price=pos.current_price,
                            quantity=trim_amount / pos.current_price if pos.current_price > 0 else 0,
                            reason=f"Rebalance: {actual_pct:.1%} -> {target_pct:.1%}",
                        )
                    else:
                        print(f"    Trim FAILED for {pos.symbol}")

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

    # Step 2b: Technical filters (SMA + RSI)
    print("\n=== Step 2b: Technical Filters ===")
    prices_df, dropped_reasons = apply_technical_filters(prices_df)
    for reason in dropped_reasons:
        print(f"  Dropped: {reason}")
    if prices_df.empty:
        print("Error: All stocks filtered out by technical analysis.")
        return 1
    print(f"Passed filters: {len(prices_df.columns)} symbols")

    # Compute momentum scores for optimizer tilt
    momentum_scores = compute_momentum_scores(prices_df)

    # Step 3: Optimize allocations (with caching)
    print("\n=== Step 3: Portfolio Optimization ===")
    sym_hash_filtered = symbols_hash(list(prices_df.columns))
    opt_cache_key = f"result_{date_key}_{sym_hash_filtered}"

    # Build sector mapping for optimizer constraints
    sector_map = {}
    for r in recommendations:
        if r.stock.symbol in prices_df.columns:
            sector_map[r.stock.symbol] = r.stock.sector or "Unknown"

    result = None
    if not force_refresh:
        cached_result = cache.load("optimization", opt_cache_key)
        if cached_result:
            result = dict_to_optimization_result(cached_result)
            print("  Using cached optimization result")

    if result is None:
        result = optimize_allocations(
            prices_df,
            Config.MIN_ALLOCATION_THRESHOLD,
            max_position_pct=Config.MAX_SINGLE_POSITION_PCT,
            sector_map=sector_map,
            max_sector_pct=Config.MAX_SECTOR_ALLOCATION,
            momentum_scores=momentum_scores,
        )
        if result.allocations:
            cache.save("optimization", opt_cache_key, optimization_result_to_dict(result))

    if not result.allocations:
        print("Error: Optimization produced no allocations.")
        return 1

    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Expected Annual Return: {result.expected_annual_return:.1%}")
    print(f"Annual Volatility: {result.annual_volatility:.1%}")

    # Step 3b: Intraday momentum check -- skip stocks crashing today
    if Config.ENABLE_INTRADAY_CHECK:
        print("\n=== Step 3b: Intraday Momentum Check ===")
        skipped = []
        for symbol in list(result.allocations.keys()):
            quote = fmp.get_quote(symbol)
            if quote:
                prev_close = quote.get("previousClose", 0)
                current_price = quote.get("price", 0)
                if prev_close > 0:
                    intraday_change = (current_price - prev_close) / prev_close
                    if intraday_change < Config.INTRADAY_MIN_CHANGE:
                        print(f"  Skipping {symbol}: down {intraday_change:.2%} today (threshold: {Config.INTRADAY_MIN_CHANGE:.0%})")
                        skipped.append(symbol)
                        del result.allocations[symbol]

        if skipped and result.allocations:
            # Redistribute skipped allocations proportionally
            total_remaining = sum(result.allocations.values())
            if total_remaining > 0:
                for sym in result.allocations:
                    result.allocations[sym] /= total_remaining
                print(f"  Redistributed allocations among {len(result.allocations)} remaining stocks")
        elif not result.allocations:
            print("  All stocks skipped by intraday check. No trades today.")
            return 0

    # Step 4: Execute trades + trailing stops
    print(f"\n=== Executing Buys ({len(result.allocations)} positions) ===")
    orders_placed = 0

    for symbol, pct in sorted(result.allocations.items(), key=lambda x: -x[1]):
        amount = Config.INVESTMENT_BUDGET * pct
        stock = next((r.stock for r in recommendations if r.stock.symbol == symbol), None)
        price = stock.price if stock else 0

        # Check if asset supports fractional shares
        fractionable = broker.is_fractionable(symbol)

        if fractionable:
            print(f"\nBuying {symbol}: {pct:.1%} = ${amount:,.2f}...")
            order_id = broker.buy_notional(symbol, amount)
        else:
            # Non-fractionable: buy whole shares only
            if price <= 0:
                print(f"\nSkipping {symbol}: no price available for whole-share calculation")
                continue
            whole_qty = int(amount // price)
            if whole_qty < 1:
                print(f"\nSkipping {symbol}: ${amount:,.2f} < 1 share at ${price:.2f}")
                continue
            actual_amount = whole_qty * price
            print(f"\nBuying {symbol}: {whole_qty} shares (~${actual_amount:,.2f}, non-fractionable)...")
            order_id = broker.buy_qty(symbol, whole_qty)

        if order_id:
            print(f"  Order placed: {order_id}")
            orders_placed += 1

            # Record buy
            qty_est = amount / price if price > 0 else 0
            tracker.record_buy(symbol=symbol, price=price, quantity=qty_est)

            # Place trailing stop using actual position qty
            if Config.TRAILING_STOP_PCT > 0:
                time.sleep(1)  # Brief pause for order fill
                position = broker.get_position(symbol)
                if position and float(position.qty) > 0:
                    stop_id = broker.place_trailing_stop(
                        symbol, float(position.qty), Config.TRAILING_STOP_PCT * 100,
                    )
                    if stop_id:
                        print(f"  Trailing stop ({Config.TRAILING_STOP_PCT:.0%}): {stop_id}")
                    else:
                        print(f"  Warning: Trailing stop failed for {symbol}")
                else:
                    print(f"  Warning: Could not get position for trailing stop")
        else:
            print(f"  Order FAILED")

    print(f"\nOrders placed: {orders_placed}/{len(result.allocations)}")

    # Step 5: Tighten trailing stops on winning positions
    if Config.TRAILING_STOP_TIGHT_PCT > 0:
        print("\n=== Step 5: Reviewing Trailing Stops ===")
        refreshed_status = broker.get_account_status()
        for pos in refreshed_status.positions:
            if pos.unrealized_plpc >= Config.PROFIT_TARGET_TIGHTEN_PCT:
                print(f"  {pos.symbol}: up {pos.unrealized_plpc:.1%}, tightening stop to {Config.TRAILING_STOP_TIGHT_PCT:.0%}")
                cancelled = broker.cancel_open_orders(pos.symbol)
                if cancelled:
                    print(f"    Cancelled {cancelled} existing order(s)")
                stop_id = broker.place_trailing_stop(
                    pos.symbol,
                    float(pos.qty),
                    Config.TRAILING_STOP_TIGHT_PCT * 100,
                )
                if stop_id:
                    print(f"    Tight trailing stop: {stop_id}")
                else:
                    print(f"    Warning: Tight trailing stop failed for {pos.symbol}")

    print(f"\nTotal API calls: {fmp.get_api_call_count()}")
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
