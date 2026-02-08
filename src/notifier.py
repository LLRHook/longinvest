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

    # Determine color based on portfolio performance
    daily_pl = portfolio["daily_pl"]
    daily_pl_pct = portfolio["daily_pl_pct"]

    if daily_pl >= 0:
        color = 0x00FF00  # Green
        change_emoji = "📈"
    else:
        color = 0xFF0000  # Red
        change_emoji = "📉"

    # Format portfolio change
    pl_sign = "+" if daily_pl >= 0 else ""
    pct_sign = "+" if daily_pl_pct >= 0 else ""

    # Format benchmark
    bench_sign = "+" if benchmark["daily_change_pct"] >= 0 else ""

    # Calculate outperformance
    outperformance = daily_pl_pct - benchmark["daily_change_pct"]
    out_sign = "+" if outperformance >= 0 else ""
    out_emoji = "✅" if outperformance >= 0 else "❌"

    # Build description
    description_lines = [
        f"**Portfolio Value:** ${portfolio['value']:,.2f} ({pl_sign}${daily_pl:,.2f} | {pct_sign}{daily_pl_pct:.2%})",
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

    # Top performers (up to 3) - show positions with positive intraday P/L
    top_performers = [p for p in sorted_positions if p["intraday_pl"] > 0][:3]
    if top_performers:
        top_lines = [
            f"`{p['symbol']:6}` +${p['intraday_pl']:.2f} ({'+' if p['change_today'] >= 0 else ''}{p['change_today']:.2%})"
            for p in top_performers
        ]
        fields.append({
            "name": "🚀 Top Performers",
            "value": "\n".join(top_lines),
            "inline": True,
        })

    # Laggards (up to 3) - show positions with negative intraday P/L
    laggards = [p for p in reversed(sorted_positions) if p["intraday_pl"] < 0][:3]
    if laggards:
        lag_lines = [
            f"`{p['symbol']:6}` ${p['intraday_pl']:.2f} ({p['change_today']:.2%})"
            for p in laggards
        ]
        fields.append({
            "name": "📉 Laggards",
            "value": "\n".join(lag_lines),
            "inline": True,
        })

    # Add all positions summary if more than 6
    if len(positions) > 6:
        all_lines = [
            f"`{p['symbol']:6}` {'+' if p['intraday_pl'] >= 0 else ''}${p['intraday_pl']:.2f}"
            for p in sorted_positions
        ]
        fields.append({
            "name": f"📊 All Positions ({len(positions)})",
            "value": "\n".join(all_lines),
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
