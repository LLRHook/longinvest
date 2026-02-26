#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from config import Config
from src.broker import AlpacaBroker
from src.cache import (
    CacheManager,
    dataframe_to_dict,
    dict_to_dataframe,
    symbols_hash,
)
from src.data import FMPClient
from src.charter import (
    fetch_benchmark_history,
    fetch_portfolio_history,
    calculate_cumulative_returns,
    generate_performance_chart,
)
from src.finscan import FinScanClient
from src.notifier import (
    format_circuit_breaker_embed,
    format_dca_buy_embed,
    format_dca_summary_embed,
    format_finscan_alert_embed,
    format_finscan_reject_embed,
    format_performance_embed,
    format_screening_embed,
    send_discord_chart_message,
    send_discord_notification,
    send_discord_notification_with_chart,
)
from src.optimizer import compute_individual_stats
from src.reporter import format_console_report, generate_daily_report
from src.strategy import MultiFactorStrategy
from src.technicals import apply_technical_filters, compute_price_momentum_12_1
from src.tracker import TradeTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _is_monday() -> bool:
    """Check if today is Monday (weekday 0)."""
    return datetime.now().weekday() == 0


def _get_investment_budget() -> tuple[float, int]:
    """Get investment budget and target number of stocks.

    On Monday: use WEEKLY_INVESTMENT spread across WEEKLY_TOP_N stocks.
    Otherwise: use DAILY_INVESTMENT spread across DCA_TOP_N stocks.

    Returns: (budget, num_stocks)
    """
    if _is_monday():
        return Config.WEEKLY_INVESTMENT, Config.WEEKLY_TOP_N
    else:
        return Config.DAILY_INVESTMENT, Config.DCA_TOP_N


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

    # Generate performance chart from pre-fetched data
    print("Generating performance chart...")
    chart_image = None
    if portfolio_df is not None and benchmark_df is not None:
        perf_data = calculate_cumulative_returns(portfolio_df, benchmark_df)
        if perf_data is not None:
            chart_image = generate_performance_chart(
                perf_data,
                report=report,
                portfolio_df=portfolio_df,
                benchmark_df=benchmark_df,
            )
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


def _fetch_momentum_signals(fmp: FMPClient, symbols: list[str], cache: CacheManager, force_refresh: bool) -> tuple[dict[str, float], pd.DataFrame]:
    """Fetch historical prices and compute 12-1 month momentum signals."""
    if not symbols:
        return {}, pd.DataFrame()

    date_key = cache.get_date_key()
    sym_hash = symbols_hash(symbols)
    prices_cache_key = f"prices_{date_key}_{sym_hash}"

    prices_df = None
    if not force_refresh:
        cached_prices = cache.load("prices", prices_cache_key)
        if cached_prices:
            prices_df = dict_to_dataframe(cached_prices)
            print(f"  Using cached prices ({len(prices_df)} days, {len(prices_df.columns)} symbols)")

    if prices_df is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=Config.HISTORICAL_DAYS)
        from_str = start_date.strftime("%Y-%m-%d")
        to_str = end_date.strftime("%Y-%m-%d")

        print(f"  Fetching prices for {len(symbols)} symbols ({from_str} to {to_str})...")
        prices_df = fmp.get_historical_prices_batch(
            symbols, from_str, to_str, Config.MIN_HISTORICAL_DAYS
        )

        if not prices_df.empty:
            cache.save("prices", prices_cache_key, dataframe_to_dict(prices_df))
            print(f"  Got {len(prices_df)} days for {len(prices_df.columns)} symbols")
        else:
            print("  No price data returned")

    if prices_df.empty:
        return {}, pd.DataFrame()

    signals = compute_price_momentum_12_1(prices_df)
    if not signals:
        print(f"  Momentum: need 273 trading days, have {len(prices_df)} — insufficient history")
    return signals, prices_df


def _compute_candidate_volatilities(prices_df: pd.DataFrame) -> dict[str, float]:
    """Compute annualized volatility for each symbol from price history."""
    if prices_df.empty:
        return {}
    vols = {}
    for symbol in prices_df.columns:
        prices = prices_df[symbol].dropna()
        if len(prices) < 20:
            continue
        daily_returns = prices.pct_change().dropna()
        annual_vol = float(daily_returns.std() * (252 ** 0.5))
        if annual_vol > 0:
            vols[symbol] = annual_vol
    return vols


def _init_finscan(cache: CacheManager | None = None) -> FinScanClient | None:
    """Initialize FinScan client if API key is configured."""
    if Config.FINSCAN_API_KEY:
        return FinScanClient(Config.FINSCAN_API_KEY, Config.FINSCAN_BASE_URL, cache=cache)
    return None


def cmd_screen(force_refresh: bool = False) -> int:
    """Run screener and display multi-factor results."""
    fmp = FMPClient()
    cache = CacheManager()
    strategy = MultiFactorStrategy(fmp, cache=cache, force_refresh=force_refresh)

    print("\n=== Running Multi-Factor Screener ===")
    if force_refresh:
        print("(forcing fresh data)")

    # Get passing stocks, compute momentum, then score
    passing_stocks = strategy._get_passing_stocks()
    symbols = [s.symbol for s in passing_stocks[:Config.OPTIMIZER_CANDIDATES]]
    momentum, prices_df = _fetch_momentum_signals(fmp, symbols, cache, force_refresh)
    if momentum:
        print(f"  Computed 12-1 momentum for {len(momentum)} symbols")

    scored = strategy.screen(momentum_signals=momentum)

    print(f"\n=== Top 20 Candidates ===")
    for i, s in enumerate(scored[:20], 1):
        print(f"\n{i}. {s.stock.symbol} - {s.stock.name}")
        print(f"   Score: {s.score:.1f}/100")
        print(f"   Market Cap: ${s.stock.market_cap / 1e9:.1f}B")
        print(f"   Price: ${s.stock.price:.2f}")
        print(f"   Sector: {s.stock.sector}")
        for reason in s.reasons:
            print(f"   - {reason}")

    print(f"\nTotal API calls: {fmp.get_api_call_count()}")
    return 0


def cmd_dry_run(force_refresh: bool = False) -> int:
    """DCA dry run — show what would be bought today."""
    broker = AlpacaBroker()
    fmp = FMPClient()
    cache = CacheManager()
    strategy = MultiFactorStrategy(fmp, cache=cache, force_refresh=force_refresh)
    finscan = _init_finscan(cache)
    if finscan:
        print("FinScan risk screening: enabled")

    status = broker.get_account_status()
    investment_budget, num_targets = _get_investment_budget()
    is_weekly = _is_monday()

    print("\n=== DCA Dry Run ===")
    if force_refresh:
        print("(forcing fresh data)")
    print(f"Cash: ${status.cash:,.2f}")
    if is_weekly:
        print(f"Weekly investment: ${investment_budget:,.2f} (Monday execution)")
    else:
        print(f"Daily investment: ${investment_budget:,.2f}")
    print(f"Target stocks: {num_targets}")
    print(f"Current positions: {len(status.positions)}")

    if status.cash < investment_budget:
        print(f"\nInsufficient cash (${status.cash:,.2f} < ${investment_budget:,.2f})")
        return 0

    # Circuit breaker checks
    portfolio_change = 0.0
    if status.last_equity > 0:
        portfolio_change = (status.portfolio_value - status.last_equity) / status.last_equity
    if portfolio_change < Config.CIRCUIT_BREAKER_PCT:
        print(f"\nCircuit breaker: portfolio down {portfolio_change:.2%} (threshold: {Config.CIRCUIT_BREAKER_PCT:.0%})")
        return 0

    spy_quote = fmp.get_quote("SPY")
    if spy_quote:
        spy_prev = spy_quote.get("previousClose", 0)
        spy_price = spy_quote.get("price", 0)
        if spy_prev > 0:
            spy_change = (spy_price - spy_prev) / spy_prev
            if spy_change < Config.MARKET_CIRCUIT_BREAKER_PCT:
                print(f"\nMarket circuit breaker: SPY down {spy_change:.2%} (threshold: {Config.MARKET_CIRCUIT_BREAKER_PCT:.0%})")
                return 0

    # Step 1: Get passing stocks (fundamentals — cached after first run)
    print("\n=== Screening Universe ===")
    passing_stocks = strategy._get_passing_stocks()
    symbols = [s.symbol for s in passing_stocks[:Config.OPTIMIZER_CANDIDATES]]

    # Step 2: Compute momentum from top candidates
    print("\n=== Fetching Price History ===")
    momentum, prices_df = _fetch_momentum_signals(fmp, symbols, cache, force_refresh)
    if momentum:
        print(f"  Computed 12-1 momentum for {len(momentum)} symbols")

    # Step 3: Score with momentum and pick DCA targets
    print("\n=== Selecting DCA Targets ===")
    targets = strategy.get_dca_buy_targets(
        positions=status.positions,
        portfolio_value=status.portfolio_value,
        momentum_signals=momentum,
        finscan=finscan,
    )

    if not targets:
        print("No valid DCA targets found today.")
        return 0

    # Compute vol-adjusted allocations
    vols = _compute_candidate_volatilities(prices_df)
    raw_weights = {}
    for t in targets:
        vol = vols.get(t.stock.symbol)
        if vol and vol > 0:
            raw_weights[t.stock.symbol] = 1.0 / vol
        else:
            median_vol = sorted(vols.values())[len(vols) // 2] if vols else 0.5
            raw_weights[t.stock.symbol] = 1.0 / median_vol

    total_weight = sum(raw_weights.values())
    allocations = {sym: (w / total_weight) * investment_budget for sym, w in raw_weights.items()}

    # Apply FinScan allocation modifiers (ELEVATED risk -> 50%)
    for target in targets:
        sym = target.stock.symbol
        if target.allocation_modifier < 1.0 and sym in allocations:
            allocations[sym] *= target.allocation_modifier

    weekly_text = " (WEEKLY - Monday)" if is_weekly else ""
    print(f"\nBuying {len(targets)} stocks (vol-adjusted, DRY RUN{weekly_text}):")
    for target in targets:
        symbol = target.stock.symbol
        amount = allocations[symbol]
        vol_pct = vols.get(symbol, 0) * 100
        print(f"\n  {symbol} ({target.stock.name})")
        print(f"    Score: {target.score:.1f}/100 | Vol: {vol_pct:.0f}% | Amount: ${amount:,.2f}")
        print(f"    Price: ${target.stock.price:.2f}")
        print(f"    Sector: {target.stock.sector}")
        if target.finscan_result:
            fs = target.finscan_result
            pio_str = f" | Piotroski: {fs.piotroski_score}/9" if fs.piotroski_score is not None else ""
            print(f"    FinScan: {fs.composite_score}/100 ({fs.risk_rating}){pio_str}")
        for reason in target.reasons:
            print(f"    - {reason}")

    print(f"\nTotal API calls: {fmp.get_api_call_count()}")
    print("\n[DRY RUN - No trades executed]")
    return 0


def cmd_execute(force_refresh: bool = False) -> int:
    """Execute the DCA buy (daily or weekly on Monday)."""
    broker = AlpacaBroker()
    fmp = FMPClient()
    cache = CacheManager()
    strategy = MultiFactorStrategy(fmp, cache=cache, force_refresh=force_refresh)
    tracker = TradeTracker()
    finscan = _init_finscan(cache)
    if finscan:
        print("FinScan risk screening: enabled")

    status = broker.get_account_status()
    investment_budget, num_targets = _get_investment_budget()
    is_weekly = _is_monday()

    print("\n=== Executing DCA Buy ===")
    if force_refresh:
        print("(forcing fresh data)")
    print(f"Cash: ${status.cash:,.2f}")
    if is_weekly:
        print(f"Weekly investment: ${investment_budget:,.2f} (Monday execution)")
    else:
        print(f"Daily investment: ${investment_budget:,.2f}")
    print(f"Target stocks: {num_targets}")
    print(f"Current positions: {len(status.positions)}")

    # Check cash
    if status.cash < investment_budget:
        print(f"\nInsufficient cash (${status.cash:,.2f} < ${investment_budget:,.2f})")
        return 0

    # Circuit breaker: portfolio
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

    # Circuit breaker: market (SPY)
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

    # Step 1: Get passing stocks (fundamentals — cached after first run)
    print("\n=== Screening Universe ===")
    passing_stocks = strategy._get_passing_stocks()
    symbols = [s.symbol for s in passing_stocks[:Config.OPTIMIZER_CANDIDATES]]

    # Step 2: Compute momentum from top candidates
    momentum, prices_df = _fetch_momentum_signals(fmp, symbols, cache, force_refresh)
    if momentum:
        print(f"  Computed 12-1 momentum for {len(momentum)} symbols")

    # Step 3: Score with momentum and pick DCA targets
    print("\n=== Selecting DCA Targets ===")
    targets = strategy.get_dca_buy_targets(
        positions=status.positions,
        portfolio_value=status.portfolio_value,
        momentum_signals=momentum,
        finscan=finscan,
    )

    if not targets:
        print("No valid DCA targets found today.")
        return 0

    # Compute vol-adjusted allocations
    vols = _compute_candidate_volatilities(prices_df)
    raw_weights = {}
    for t in targets:
        vol = vols.get(t.stock.symbol)
        if vol and vol > 0:
            raw_weights[t.stock.symbol] = 1.0 / vol
        else:
            # Fallback: use median vol of available candidates
            median_vol = sorted(vols.values())[len(vols) // 2] if vols else 0.5
            raw_weights[t.stock.symbol] = 1.0 / median_vol

    total_weight = sum(raw_weights.values())
    allocations = {sym: (w / total_weight) * investment_budget for sym, w in raw_weights.items()}

    # Apply FinScan allocation modifiers (ELEVATED risk -> 50%)
    for target in targets:
        sym = target.stock.symbol
        if target.allocation_modifier < 1.0 and sym in allocations:
            original = allocations[sym]
            allocations[sym] = original * target.allocation_modifier
            logger.info(f"FinScan reduced {sym} allocation: ${original:,.2f} -> ${allocations[sym]:,.2f}")

    weekly_text = " (WEEKLY - Monday)" if is_weekly else ""
    print(f"\nBuying {len(targets)} stocks (vol-adjusted{weekly_text}):")
    bought_count = 0
    for target in targets:
        symbol = target.stock.symbol
        price = target.stock.price
        amount = allocations[symbol]
        vol_pct = vols.get(symbol, 0) * 100

        print(f"\n  {symbol} ({target.stock.name})")
        print(f"    Score: {target.score:.1f}/100 | Vol: {vol_pct:.0f}% | Amount: ${amount:,.2f}")

        if price <= 0:
            print(f"    Skipping: no price available")
            continue

        # Determine order type and place order
        order_id = None
        order_type_str = "market"

        fractionable = broker.is_fractionable(symbol)

        # Try limit order first if enabled
        if Config.USE_LIMIT_ORDERS:
            limit_price = price * (1 + Config.LIMIT_ORDER_SPREAD_PCT)
            order_type_str = f"limit@${limit_price:.2f}"

            if fractionable:
                order_id = broker.buy_limit_notional(symbol, amount, limit_price)
            else:
                whole_qty = int(amount // price)
                if whole_qty >= 1:
                    order_id = broker.buy_limit_qty(symbol, whole_qty, limit_price)

            # Fall back to market order if limit order failed and configured
            if not order_id and Config.LIMIT_ORDER_FALLBACK_TO_MARKET:
                logger.info(f"Limit order failed for {symbol}, falling back to market order")
                order_type_str = "market (fallback)"

        # Fall back to market if limit not enabled or failed
        if not order_id:
            if fractionable:
                order_id = broker.buy_notional(symbol, amount)
            else:
                whole_qty = int(amount // price)
                if whole_qty < 1:
                    print(f"    Skipping: ${amount:,.2f} < 1 share at ${price:.2f}")
                    continue
                print(f"    Non-fractionable: buying {whole_qty} shares (~${whole_qty * price:,.2f})")
                order_id = broker.buy_qty(symbol, whole_qty)

        if order_id:
            print(f"    Order placed ({order_type_str}): {order_id}")
            qty_est = amount / price if price > 0 else 0
            tracker.record_buy(symbol=symbol, price=price, quantity=qty_est)
            bought_count += 1

            if Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
                embed = format_dca_buy_embed(
                    symbol=symbol,
                    name=target.stock.name,
                    score=target.score,
                    amount=amount,
                    price=price,
                    sector=target.stock.sector or "Unknown",
                    reasons=target.reasons,
                    position_count=len(status.positions) + bought_count,
                )
                send_discord_notification(Config.DISCORD_WEBHOOK_URL, embed)
        else:
            print(f"    Order FAILED for {symbol}")

    print(f"\nBought {bought_count}/{len(targets)} stocks successfully.")

    # Send DCA summary notification
    if bought_count > 1 and Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
        buy_summaries = []
        for target in targets:
            sym = target.stock.symbol
            if sym in allocations:
                buy_summaries.append({
                    "symbol": sym,
                    "name": target.stock.name,
                    "score": target.score,
                    "amount": allocations[sym],
                    "price": target.stock.price,
                    "sector": target.stock.sector or "Unknown",
                    "vol": vols.get(sym, 0) * 100,
                })
        summary_embed = format_dca_summary_embed(buy_summaries, Config.DAILY_INVESTMENT)
        time.sleep(1)
        send_discord_notification(Config.DISCORD_WEBHOOK_URL, summary_embed)

    print(f"Total API calls: {fmp.get_api_call_count()}")
    return 0 if bought_count > 0 else 1


def cmd_monitor() -> int:
    """Scan all held positions through FinScan for risk monitoring."""
    if not Config.FINSCAN_API_KEY:
        print("FinScan API key not configured. Set FINSCAN_API_KEY.")
        return 1

    broker = AlpacaBroker()
    cache = CacheManager()
    finscan = FinScanClient(Config.FINSCAN_API_KEY, Config.FINSCAN_BASE_URL, cache=cache)

    status = broker.get_account_status()
    print(f"\n=== FinScan Portfolio Monitor ===")
    print(f"Scanning {len(status.positions)} held positions...\n")

    alerts = []
    for p in status.positions:
        result = finscan.scan(p.symbol)
        if result:
            flag_str = f" | {len(result.red_flags)} flags" if result.red_flags else ""
            pio_str = f" | F-Score: {result.piotroski_score}/9" if result.piotroski_score is not None else ""
            print(f"  {p.symbol:8} Risk: {result.composite_score:3}/100 ({result.risk_rating:9}){pio_str}{flag_str}")

            if result.risk_rating in ("HIGH", "ELEVATED"):
                alerts.append((p.symbol, result))
        else:
            print(f"  {p.symbol:8} [scan failed]")

    if alerts:
        print(f"\n=== {len(alerts)} Alert(s) ===")
        for symbol, result in alerts:
            print(f"  {symbol}: {result.risk_rating} ({result.composite_score}/100)")
            for flag in result.red_flags[:5]:
                print(f"    [{flag.get('severity', '?')}] {flag.get('message', 'N/A')}")

            # Send Discord alert
            if Config.ENABLE_NOTIFICATIONS and Config.DISCORD_WEBHOOK_URL:
                embed = format_finscan_alert_embed(
                    symbol=symbol,
                    composite_score=result.composite_score,
                    risk_rating=result.risk_rating,
                    red_flags=result.red_flags,
                )
                send_discord_notification(Config.DISCORD_WEBHOOK_URL, embed)
    else:
        print("\nNo alerts — all positions within risk tolerance.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-Factor DCA investment bot",
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
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Scan held positions through FinScan for risk alerts",
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
        if not (args.status or args.screen or args.dry_run or args.report or args.monitor):
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
        elif args.monitor:
            return cmd_monitor()
        else:
            return cmd_execute(force_refresh=args.force_refresh)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
