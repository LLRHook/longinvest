import json
import logging
from io import BytesIO
from typing import Any

import requests

logger = logging.getLogger(__name__)


def send_discord_notification(webhook_url: str, embed: dict[str, Any]) -> bool:
    """Send a Discord notification with an embed.

    Args:
        webhook_url: Discord webhook URL
        embed: Discord embed object

    Returns:
        True if sent successfully, False otherwise
    """
    if not webhook_url:
        logger.warning("Discord webhook URL not configured")
        return False

    payload = {"embeds": [embed]}

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10,
        )

        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 5)
            logger.warning(f"Discord rate limited, retry after {retry_after}s")
            return False

        response.raise_for_status()
        logger.info("Discord notification sent successfully")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to send Discord notification: {e}")
        return False


def send_discord_notification_with_chart(
    webhook_url: str, embed: dict[str, Any], chart_image: BytesIO
) -> bool:
    """Send a Discord notification with an embed and attached chart image.

    Uses multipart/form-data to attach the chart PNG and reference it in the embed.

    Args:
        webhook_url: Discord webhook URL
        embed: Discord embed object
        chart_image: PNG image as BytesIO

    Returns:
        True if sent successfully, False otherwise
    """
    if not webhook_url:
        logger.warning("Discord webhook URL not configured")
        return False

    embed["image"] = {"url": "attachment://performance.png"}
    payload = {"embeds": [embed]}

    try:
        response = requests.post(
            webhook_url,
            files={"file": ("performance.png", chart_image, "image/png")},
            data={"payload_json": json.dumps(payload)},
            timeout=15,
        )

        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 5)
            logger.warning(f"Discord rate limited, retry after {retry_after}s")
            return False

        response.raise_for_status()
        logger.info("Discord notification with chart sent successfully")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to send Discord notification with chart: {e}")
        return False


def send_discord_chart_message(
    webhook_url: str,
    chart_image: BytesIO,
    title: str = "Cumulative Growth: Portfolio vs SPY",
) -> bool:
    """Send a chart image as a standalone Discord message.

    Args:
        webhook_url: Discord webhook URL
        chart_image: PNG image as BytesIO
        title: Embed title for the chart message

    Returns:
        True if sent successfully, False otherwise
    """
    if not webhook_url:
        logger.warning("Discord webhook URL not configured")
        return False

    embed = {
        "title": title,
        "image": {"url": "attachment://performance.png"},
        "color": 0x3B82F6,
    }
    payload = {"embeds": [embed]}

    try:
        response = requests.post(
            webhook_url,
            files={"file": ("performance.png", chart_image, "image/png")},
            data={"payload_json": json.dumps(payload)},
            timeout=15,
        )

        if response.status_code == 429:
            retry_after = response.json().get("retry_after", 5)
            logger.warning(f"Discord rate limited, retry after {retry_after}s")
            return False

        response.raise_for_status()
        logger.info("Discord chart message sent successfully")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to send Discord chart message: {e}")
        return False


def format_screening_embed(
    recommendations: list,
    max_display: int = 5,
) -> dict[str, Any]:
    """Format top screening picks as a Discord embed.

    Args:
        recommendations: List of ScoredStock objects.
        max_display: Number of top picks to show.

    Returns:
        Discord embed dict.
    """
    lines = []
    for i, rec in enumerate(recommendations[:max_display], 1):
        lines.append(
            f"**{i}. {rec.stock.symbol}** - Score: {rec.score:.1f} "
            f"| ${rec.stock.price:.2f} | MCap: ${rec.stock.market_cap / 1e9:.1f}B"
        )
    description = "\n".join(lines) if lines else "No recommendations."
    return {
        "title": f"Screening Results - Top {min(max_display, len(recommendations))} Picks",
        "description": description,
        "color": 0x3B82F6,  # Blue
        "footer": {"text": "Long-term Growth Bot"},
    }


def format_rebalance_embed(
    trimmed: list[tuple[str, float, float]],
) -> dict[str, Any]:
    """Format rebalancing actions as a Discord embed.

    Args:
        trimmed: List of (symbol, old_pct, new_pct) tuples.

    Returns:
        Discord embed dict.
    """
    lines = []
    for symbol, old_pct, trim_amount in trimmed:
        lines.append(f"**{symbol}**: {old_pct:.1%} -> trimmed ${trim_amount:,.2f}")
    description = "\n".join(lines) if lines else "No rebalancing needed."
    return {
        "title": "Portfolio Rebalanced",
        "description": description,
        "color": 0xFF8C00,  # Orange
        "footer": {"text": "Long-term Growth Bot"},
    }


def format_circuit_breaker_embed(
    reason: str,
    change_pct: float,
) -> dict[str, Any]:
    """Format a circuit breaker alert as a Discord embed."""
    return {
        "title": "Circuit Breaker Triggered",
        "description": f"**Reason:** {reason}\n**Change:** {change_pct:.2%}\n\nTrading halted for the day.",
        "color": 0xFF0000,
        "footer": {"text": "Long-term Growth Bot"},
    }


def format_sell_embed(
    symbol: str,
    reason: str,
    entry_price: float,
    exit_price: float,
    pl: float,
    hold_days: int | None = None,
) -> dict[str, Any]:
    """Format a sell notification as a Discord embed.

    Args:
        symbol: Stock symbol sold
        reason: Why the position was sold
        entry_price: Average entry price
        exit_price: Price at exit
        pl: Realized P/L in dollars
        hold_days: Days held (None if unknown)

    Returns:
        Discord embed dict
    """
    # Orange for stop-loss, red for fundamental degradation
    is_stop = "stop" in reason.lower() or "trailing" in reason.lower()
    color = 0xFF8C00 if is_stop else 0xFF4444

    pl_sign = "+" if pl >= 0 else ""
    pl_pct = ((exit_price / entry_price) - 1) if entry_price > 0 else 0
    pct_sign = "+" if pl_pct >= 0 else ""

    description_lines = [
        f"**Symbol:** {symbol}",
        f"**Reason:** {reason}",
        f"**Entry:** ${entry_price:.2f} -> **Exit:** ${exit_price:.2f}",
        f"**P/L:** {pl_sign}${pl:,.2f} ({pct_sign}{pl_pct:.2%})",
    ]
    if hold_days is not None:
        description_lines.append(f"**Held:** {hold_days} days")

    return {
        "title": f"{'🛑' if is_stop else '🔴'} Position Sold - {symbol}",
        "description": "\n".join(description_lines),
        "color": color,
        "footer": {"text": "Long-term Growth Bot"},
    }


def format_dca_buy_embed(
    symbol: str,
    name: str,
    score: float,
    amount: float,
    price: float,
    sector: str,
    reasons: list[str],
    position_count: int,
) -> dict[str, Any]:
    """Format a DCA daily buy notification as a Discord embed.

    Args:
        symbol: Stock symbol bought
        name: Company name
        score: Multi-factor score (0-100)
        amount: Dollar amount invested
        price: Price at purchase
        sector: Stock sector
        reasons: List of scoring reasons
        position_count: Total positions after buy

    Returns:
        Discord embed dict
    """
    reason_lines = "\n".join(f"- {r}" for r in reasons[:6]) if reasons else "N/A"

    description_lines = [
        f"**Symbol:** {symbol} ({name})",
        f"**Score:** {score:.1f}/100",
        f"**Amount:** ${amount:,.2f} @ ${price:.2f}",
        f"**Sector:** {sector}",
        f"**Positions:** {position_count}",
        "",
        f"**Scoring Factors:**\n{reason_lines}",
    ]

    return {
        "title": f"DCA Buy - {symbol}",
        "description": "\n".join(description_lines),
        "color": 0x22C55E,  # Green
        "footer": {"text": "Multi-Factor DCA Bot"},
    }


def format_dca_summary_embed(
    buys: list[dict[str, Any]],
    total_amount: float,
) -> dict[str, Any]:
    """Format a summary of all DCA buys for the day as a Discord embed.

    Args:
        buys: List of dicts with keys: symbol, name, score, amount, price, sector, vol
        total_amount: Total dollar amount invested

    Returns:
        Discord embed dict
    """
    lines = []
    for b in buys:
        vol_str = f" | Vol: {b['vol']:.0f}%" if b.get('vol') else ""
        lines.append(
            f"**{b['symbol']}** ({b['name'][:20]}) — "
            f"${b['amount']:,.0f} @ ${b['price']:.2f} | "
            f"Score: {b['score']:.0f}{vol_str}"
        )

    description_lines = [
        f"**Total invested:** ${total_amount:,.2f}",
        f"**Stocks bought:** {len(buys)}",
        "",
    ] + lines

    return {
        "title": f"DCA Summary — {len(buys)} Buys",
        "description": "\n".join(description_lines),
        "color": 0x22C55E,
        "footer": {"text": "Multi-Factor DCA Bot"},
    }


def format_performance_embed(report_data: dict[str, Any]) -> dict[str, Any]:
    """Format report data into a Discord embed.

    Args:
        report_data: Report data from generate_daily_report()

    Returns:
        Discord embed dict
    """
    portfolio = report_data["portfolio"]
    benchmark = report_data["benchmark"]
    positions = report_data["positions"]

    # Determine color based on outperformance vs benchmark (not absolute P/L)
    daily_pl = portfolio["daily_pl"]
    daily_pl_pct = portfolio["daily_pl_pct"]
    outperformance = daily_pl_pct - benchmark["daily_change_pct"]

    if outperformance >= 0:
        color = 0x00FF00  # Green — beat benchmark
        change_emoji = "📈"
    else:
        color = 0xFF0000  # Red — behind benchmark
        change_emoji = "📉"

    # Format portfolio change
    pl_sign = "+" if daily_pl >= 0 else ""
    pct_sign = "+" if daily_pl_pct >= 0 else ""

    # Format benchmark
    bench_sign = "+" if benchmark["daily_change_pct"] >= 0 else ""

    out_sign = "+" if outperformance >= 0 else ""
    out_emoji = "✅" if outperformance >= 0 else "❌"

    # Build description
    invested = portfolio.get("invested", portfolio["value"])
    description_lines = [
        f"**Invested:** ${invested:,.2f} ({pl_sign}${daily_pl:,.2f} | {pct_sign}{daily_pl_pct:.2%})",
        f"**Cash:** ${portfolio['cash']:,.2f} | **Total:** ${portfolio['value']:,.2f}",
        f"**S&P 500 (SPY):** {bench_sign}{benchmark['daily_change_pct']:.2%}",
        f"**vs Benchmark:** {out_sign}{outperformance:.2%} {out_emoji}",
    ]
    description = "\n".join(description_lines)

    # Sort positions by today's P/L
    sorted_positions = sorted(
        positions, key=lambda p: p["intraday_pl"], reverse=True
    )

    # Build position fields
    fields = []

    # Compact positions table
    if sorted_positions:
        header = f"{'Sym':<6} {'Value':>8} {'Today':>8} {'Total':>8}"
        sep = "-" * len(header)
        rows = []
        for p in sorted_positions:
            sym = p["symbol"][:6]
            val = f"${p['market_value']:,.0f}"
            today_sign = "+" if p["intraday_pl"] >= 0 else ""
            today = f"{today_sign}${p['intraday_pl']:,.0f}"
            total_sign = "+" if p["unrealized_pl"] >= 0 else ""
            total = f"{total_sign}${p['unrealized_pl']:,.0f}"
            rows.append(f"{sym:<6} {val:>8} {today:>8} {total:>8}")
        table = f"```\n{header}\n{sep}\n" + "\n".join(rows) + "\n```"
        fields.append({
            "name": f"📊 Positions ({len(sorted_positions)})",
            "value": table,
            "inline": False,
        })

    # Risk metrics
    risk = report_data.get("risk_metrics", {})
    if risk:
        risk_lines = []
        risk_lines.append(f"Sharpe: **{risk.get('sharpe', 'N/A')}** | Sortino: **{risk.get('sortino', 'N/A')}**")
        risk_lines.append(f"Max DD: **{risk.get('max_drawdown', 'N/A')}%** | Current DD: **{risk.get('current_drawdown', 'N/A')}%**")
        risk_lines.append(f"Win Rate: **{risk.get('win_rate', 'N/A')}%**")
        if "alpha" in risk and "beta" in risk:
            alpha_sign = "+" if risk["alpha"] >= 0 else ""
            risk_lines.append(f"Alpha: **{alpha_sign}{risk['alpha']}%** | Beta: **{risk['beta']}**")
        fields.append({
            "name": "📐 Risk Metrics",
            "value": "\n".join(risk_lines),
            "inline": False,
        })

    # Sector exposure
    sector = report_data.get("sector_exposure", {})
    if sector:
        sector_lines = [f"`{name[:16]:16}` {pct}%" for name, pct in sector.items()]
        fields.append({
            "name": "🏭 Sector Exposure",
            "value": "\n".join(sector_lines[:8]),
            "inline": False,
        })

    embed = {
        "title": f"{change_emoji} Daily Portfolio Report - {report_data['date']}",
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {"text": "Long-term Growth Bot"},
    }

    return embed


def format_finscan_reject_embed(
    symbol: str,
    reason: str,
    score: int,
    rating: str,
) -> dict[str, Any]:
    """Format a FinScan rejection as a Discord embed.

    Args:
        symbol: Stock symbol that was rejected
        reason: Reason for rejection
        score: Risk score (0-100)
        rating: Risk rating label

    Returns:
        Discord embed dict
    """
    description_lines = [
        f"**Reason:** {reason}",
        f"**Risk Score:** {score}/100 ({rating})",
    ]

    return {
        "title": f"FinScan Blocked - {symbol}",
        "description": "\n".join(description_lines),
        "color": 0xFF4444,
        "footer": {"text": "Multi-Factor DCA Bot"},
    }


def format_finscan_alert_embed(
    symbol: str,
    composite_score: int,
    risk_rating: str,
    red_flags: list[dict],
) -> dict[str, Any]:
    """Format a FinScan alert as a Discord embed.

    Args:
        symbol: Stock symbol with elevated risk
        composite_score: Composite risk score (0-100)
        risk_rating: Risk rating label (e.g. "ELEVATED", "HIGH")
        red_flags: List of dicts with 'severity' and 'message' keys

    Returns:
        Discord embed dict
    """
    description_lines = [
        f"**Risk Score:** {composite_score}/100 ({risk_rating})",
        "**Red Flags:**",
    ]
    for flag in red_flags[:5]:
        description_lines.append(f"- [{flag['severity']}] {flag['message']}")

    color = 0xFF8C00 if risk_rating == "ELEVATED" else 0xFF0000

    return {
        "title": f"FinScan Alert - {symbol}",
        "description": "\n".join(description_lines),
        "color": color,
        "footer": {"text": "Multi-Factor DCA Bot"},
    }
