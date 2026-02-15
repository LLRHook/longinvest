import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from config import Config
from src.cache import (
    CacheManager,
    dict_to_scored_stock,
    dict_to_stock_data,
    scored_stock_to_dict,
    stock_data_to_dict,
)
from src.data import FMPClient, StockData

logger = logging.getLogger(__name__)


@dataclass
class ScoredStock:
    stock: StockData
    score: float
    reasons: list[str]


# Factor weights (must sum to 1.0)
QUALITY_WEIGHT = 0.40
VALUE_WEIGHT = 0.35
MOMENTUM_WEIGHT = 0.25


def _collect_metrics(stock: StockData) -> dict[str, float | None]:
    """Extract raw metric values from a StockData, inverting "lower is better" metrics.

    Returns dict of {metric_name: value} where higher is always better.
    """
    metrics: dict[str, float | None] = {}

    # Quality factors (higher is better)
    metrics["roe"] = stock.roe
    metrics["gross_margin"] = stock.gross_margin
    metrics["fcf_margin"] = stock.fcf_margin
    metrics["revenue_growth"] = stock.revenue_growth
    metrics["eps_beat_count"] = float(stock.eps_beat_count) if stock.eps_beat_count is not None else None

    # Value factors (lower is better — invert so higher = cheaper)
    metrics["neg_pe"] = -stock.pe_ratio if stock.pe_ratio is not None and stock.pe_ratio > 0 else None
    metrics["neg_peg"] = -stock.peg_ratio if stock.peg_ratio is not None and stock.peg_ratio > 0 else None
    metrics["neg_pb"] = -stock.price_to_book if stock.price_to_book is not None and stock.price_to_book > 0 else None
    metrics["neg_ev_ebitda"] = -stock.ev_to_ebitda if stock.ev_to_ebitda is not None and stock.ev_to_ebitda > 0 else None

    # Momentum factor (higher is better) — placeholder, filled by caller if available
    metrics["momentum_12_1"] = None

    return metrics


# Which factor group each metric belongs to, and its max point allocation within the group
METRIC_FACTOR_MAP: dict[str, tuple[str, float]] = {
    # Quality (40 pts total)
    "roe":              ("quality", 10.0),
    "gross_margin":     ("quality", 10.0),
    "fcf_margin":       ("quality", 8.0),
    "revenue_growth":   ("quality", 7.0),
    "eps_beat_count":   ("quality", 5.0),
    # Value (35 pts total)
    "neg_pe":           ("value", 10.0),
    "neg_peg":          ("value", 10.0),
    "neg_pb":           ("value", 8.0),
    "neg_ev_ebitda":    ("value", 7.0),
    # Momentum (25 pts total)
    "momentum_12_1":    ("momentum", 25.0),
}


def score_universe(
    stocks: list[StockData],
    momentum_signals: dict[str, float] | None = None,
) -> list[ScoredStock]:
    """Score a universe of stocks using cross-sectional z-score normalization.

    For each metric, z = (value - mean) / std, capped at +/-3.
    Converted to points: metric_score = max_pts * (z + 3) / 6.

    Args:
        stocks: List of StockData that passed guardrails.
        momentum_signals: Optional {symbol: raw_return} from 12-1 momentum.

    Returns:
        List of ScoredStock sorted by score descending.
    """
    if not stocks:
        return []

    momentum_signals = momentum_signals or {}

    # Step 1: Collect raw metrics for each stock
    raw_data: list[tuple[StockData, dict[str, float | None]]] = []
    for stock in stocks:
        m = _collect_metrics(stock)
        m["momentum_12_1"] = momentum_signals.get(stock.symbol)
        raw_data.append((stock, m))

    # Step 2: For each metric, compute mean/std across non-None values
    metric_names = list(METRIC_FACTOR_MAP.keys())
    metric_values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _, metrics in raw_data:
        for name in metric_names:
            val = metrics.get(name)
            if val is not None:
                metric_values[name].append(val)

    metric_stats: dict[str, tuple[float, float]] = {}
    for name in metric_names:
        vals = metric_values[name]
        if len(vals) >= 2:
            metric_stats[name] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
        elif len(vals) == 1:
            metric_stats[name] = (vals[0], 1.0)  # single value gets z=0

    # Step 3: Score each stock
    scored: list[ScoredStock] = []
    for stock, metrics in raw_data:
        total_score = 0.0
        reasons = []

        for metric_name, (factor, max_pts) in METRIC_FACTOR_MAP.items():
            val = metrics.get(metric_name)
            if val is None or metric_name not in metric_stats:
                continue

            mean, std = metric_stats[metric_name]
            if std < 1e-10:
                z = 0.0
            else:
                z = (val - mean) / std

            # Cap at +/- 3
            z = max(-3.0, min(3.0, z))

            # Convert z-score to points: z=-3 -> 0 pts, z=+3 -> max_pts
            pts = max_pts * (z + 3.0) / 6.0
            total_score += pts

            # Build reason string for significant contributors
            if pts > max_pts * 0.6:  # Above-average contributor
                display_name = metric_name.replace("neg_", "").replace("_", " ").upper()
                if metric_name.startswith("neg_"):
                    # Show the original positive value for readability
                    orig_val = -val
                    reasons.append(f"{display_name}: {orig_val:.2f} (z={z:+.1f}, {pts:.1f}/{max_pts:.0f}pts)")
                else:
                    if abs(val) < 1:
                        reasons.append(f"{display_name}: {val:.2%} (z={z:+.1f}, {pts:.1f}/{max_pts:.0f}pts)")
                    else:
                        reasons.append(f"{display_name}: {val:.2f} (z={z:+.1f}, {pts:.1f}/{max_pts:.0f}pts)")

        # Post-earnings boost
        if (stock.days_since_last_earnings is not None
                and stock.days_since_last_earnings <= Config.EARNINGS_BOOST_DAYS
                and stock.eps_beat_count is not None
                and stock.eps_beat_count > 0):
            total_score += 3.0
            reasons.append(f"Recent earnings beat ({stock.days_since_last_earnings}d ago, +3pts)")

        scored.append(ScoredStock(stock=stock, score=total_score, reasons=reasons))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


class MultiFactorStrategy:
    def __init__(
        self,
        fmp_client: FMPClient,
        cache: CacheManager | None = None,
        force_refresh: bool = False,
    ):
        self.fmp = fmp_client
        self.cache = cache
        self.force_refresh = force_refresh

    def get_screener_candidates(self, limit: int = 2000) -> list[dict]:
        """Get stock universe from FMP screener, sorted by market cap."""
        stocks = self.fmp.get_screener_stocks(
            min_market_cap=Config.MIN_MARKET_CAP,
            max_market_cap=Config.MAX_MARKET_CAP,
            limit=limit,
        )
        stocks.sort(key=lambda x: x.get("marketCap", 0), reverse=True)
        logger.info(f"FMP screener returned {len(stocks)} US stocks")
        return stocks

    def passes_guardrails(self, stock: StockData) -> tuple[bool, list[str]]:
        """Check if stock passes basic quality guardrails.

        Intentionally lenient — scoring does the real ranking work.
        """
        failures = []

        if stock.revenue is not None and stock.revenue < Config.MIN_REVENUE:
            failures.append(f"Revenue too low: ${stock.revenue:,.0f}")

        if stock.revenue_growth is None or stock.revenue_growth < Config.MIN_REVENUE_GROWTH:
            growth_str = f"{stock.revenue_growth:.1%}" if stock.revenue_growth is not None else "N/A"
            failures.append(f"Revenue growth below threshold: {growth_str}")

        if Config.REQUIRE_POSITIVE_FCF:
            if stock.free_cash_flow is not None and stock.free_cash_flow <= 0:
                failures.append(f"Negative FCF: ${stock.free_cash_flow:,.0f}")

        if stock.de_ratio is not None and stock.de_ratio > Config.MAX_DE_RATIO:
            failures.append(f"High D/E: {stock.de_ratio:.2f}")

        return len(failures) == 0, failures

    def screen(
        self,
        existing_symbols: set[str] | None = None,
        momentum_signals: dict[str, float] | None = None,
    ) -> list[ScoredStock]:
        """Run the full screening process with z-score batch scoring.

        Args:
            existing_symbols: Symbols to exclude (already held).
            momentum_signals: Optional {symbol: raw_return} from 12-1 momentum.
        """
        existing = existing_symbols or set()

        # Try to load cached scored results first
        if self.cache and not self.force_refresh:
            date_key = self.cache.get_date_key()
            cached_scored = self.cache.load("scored", f"scored_{date_key}")
            if cached_scored:
                scored_stocks = [dict_to_scored_stock(s) for s in cached_scored]
                scored_stocks = [s for s in scored_stocks if s.stock.symbol not in existing]
                print(f"  Using cached screening results ({len(scored_stocks)} stocks)")
                logger.info(f"Loaded {len(scored_stocks)} scored stocks from cache")
                return scored_stocks

        # Get screener candidates (with caching)
        candidates = self._get_cached_screener_candidates()
        candidates = [c for c in candidates if c["symbol"] not in existing]
        logger.info(f"After filtering existing positions: {len(candidates)} candidates")

        # Fetch fundamentals for all candidates
        passing_stocks: list[StockData] = []
        for i, candidate in enumerate(candidates):
            symbol = candidate["symbol"]
            print(f"  [{i + 1}/{len(candidates)}] {symbol}...", end=" ", flush=True)

            stock_data = self._get_cached_stock_data(symbol)
            if not stock_data:
                print("no data")
                continue

            passes, failures = self.passes_guardrails(stock_data)
            if not passes:
                print(f"failed ({failures[0]})")
                continue

            passing_stocks.append(stock_data)
            print("pass")

        # Batch z-score scoring across all passing stocks
        scored_stocks = score_universe(passing_stocks, momentum_signals)
        logger.info(f"Screening complete: {len(scored_stocks)} stocks scored")

        for s in scored_stocks[:10]:
            print(f"  {s.stock.symbol}: {s.score:.1f}")

        # Cache the scored results
        if self.cache:
            date_key = self.cache.get_date_key()
            cached_data = [scored_stock_to_dict(s) for s in scored_stocks]
            self.cache.save("scored", f"scored_{date_key}", cached_data)

        return scored_stocks

    def get_buy_recommendations(
        self,
        existing_symbols: set[str] | None = None,
        max_picks: int | None = None,
        momentum_signals: dict[str, float] | None = None,
    ) -> list[ScoredStock]:
        """Get top stocks to buy, filtering out earnings blackout.

        Args:
            existing_symbols: Already-held symbols to exclude.
            max_picks: Maximum number of recommendations.
            momentum_signals: Optional momentum data.
        """
        existing = existing_symbols or set()
        scored = self.screen(existing_symbols=existing, momentum_signals=momentum_signals)

        # Earnings blackout filter
        filtered = []
        today = datetime.now().date()
        for s in scored:
            if s.stock.next_earnings_date:
                try:
                    next_date = datetime.strptime(s.stock.next_earnings_date, "%Y-%m-%d").date()
                    days_until = (next_date - today).days
                    if 0 <= days_until <= Config.EARNINGS_BLACKOUT_DAYS:
                        logger.info(f"Skipping {s.stock.symbol}: earnings in {days_until} days")
                        continue
                except ValueError:
                    pass
            filtered.append(s)

        recommendations = filtered[:max_picks] if max_picks else filtered
        logger.info(f"Recommending {len(recommendations)} stocks for purchase")
        return recommendations

    def get_dca_buy_target(
        self,
        positions: list,
        portfolio_value: float,
        momentum_signals: dict[str, float] | None = None,
    ) -> ScoredStock | None:
        """Pick the single best stock for today's DCA buy.

        Logic:
        1. Re-screen + re-score the universe (including existing holdings).
        2. Pick highest-scoring stock subject to:
           - Position cap: no position > MAX_SINGLE_POSITION_PCT of portfolio
           - Sector cap: no sector > MAX_SECTOR_ALLOCATION of portfolio
        3. New positions require NEW_POSITION_SCORE_THRESHOLD premium over worst holding
           when at TARGET_POSITIONS count.

        Args:
            positions: List of broker position objects (need .symbol, .market_value, .sector or lookup).
            portfolio_value: Current total portfolio value.
            momentum_signals: Optional {symbol: raw_return}.

        Returns:
            Best ScoredStock to buy, or None if no valid target.
        """
        held_symbols = {p.symbol for p in positions}
        position_values = {p.symbol: p.market_value for p in positions}

        # Sector tracking for held positions
        sector_values: dict[str, float] = {}
        held_sectors: dict[str, str] = {}
        for p in positions:
            # Sector may not be on position object; we'll look it up during scoring
            sector_values[p.symbol] = p.market_value

        # Get scored universe (DON'T exclude held positions — we want to add to them)
        scored = self.get_buy_recommendations(
            existing_symbols=None,
            max_picks=Config.OPTIMIZER_CANDIDATES,
            momentum_signals=momentum_signals,
        )

        if not scored:
            logger.info("No scored stocks available for DCA")
            return None

        # Build sector map from scored stocks
        for s in scored:
            if s.stock.symbol in held_symbols:
                held_sectors[s.stock.symbol] = s.stock.sector or "Unknown"

        # Compute sector totals from held positions
        sector_totals: dict[str, float] = {}
        for sym, val in position_values.items():
            sector = held_sectors.get(sym, "Unknown")
            sector_totals[sector] = sector_totals.get(sector, 0) + val

        # Find worst held score (for new position threshold)
        held_scores: dict[str, float] = {}
        for s in scored:
            if s.stock.symbol in held_symbols:
                held_scores[s.stock.symbol] = s.score
        worst_held_score = min(held_scores.values()) if held_scores else 0.0

        # Pick best valid target
        for candidate in scored:
            symbol = candidate.stock.symbol
            sector = candidate.stock.sector or "Unknown"

            # Position cap check
            current_value = position_values.get(symbol, 0)
            projected_value = current_value + Config.DAILY_INVESTMENT
            if portfolio_value > 0 and projected_value / portfolio_value > Config.MAX_SINGLE_POSITION_PCT:
                logger.info(f"Skipping {symbol}: would exceed position cap ({projected_value / portfolio_value:.1%})")
                continue

            # Sector cap check
            current_sector_value = sector_totals.get(sector, 0)
            projected_sector = current_sector_value + Config.DAILY_INVESTMENT
            if portfolio_value > 0 and projected_sector / portfolio_value > Config.MAX_SECTOR_ALLOCATION:
                logger.info(f"Skipping {symbol}: would exceed sector cap for {sector}")
                continue

            # New position threshold: require premium over worst holding when at target
            if symbol not in held_symbols and len(held_symbols) >= Config.TARGET_POSITIONS:
                required_score = worst_held_score * (1 + Config.NEW_POSITION_SCORE_THRESHOLD)
                if candidate.score < required_score:
                    logger.info(
                        f"Skipping {symbol}: score {candidate.score:.1f} < "
                        f"required {required_score:.1f} (worst held + {Config.NEW_POSITION_SCORE_THRESHOLD:.0%})"
                    )
                    continue

            return candidate

        logger.info("No valid DCA target found after applying caps")
        return None

    def _get_cached_screener_candidates(self) -> list[dict]:
        """Get screener candidates with caching."""
        if self.cache and not self.force_refresh:
            date_key = self.cache.get_date_key()
            cached = self.cache.load("screener", f"candidates_{date_key}")
            if cached:
                logger.info(f"Loaded {len(cached)} candidates from cache")
                return cached

        candidates = self.get_screener_candidates()

        if self.cache:
            date_key = self.cache.get_date_key()
            self.cache.save("screener", f"candidates_{date_key}", candidates)

        return candidates

    def _get_cached_stock_data(self, symbol: str) -> StockData | None:
        """Get stock fundamental data with caching."""
        if self.cache and not self.force_refresh:
            date_key = self.cache.get_date_key()
            cached = self.cache.load("fundamentals", f"{symbol}_{date_key}")
            if cached:
                logger.debug(f"Loaded {symbol} fundamentals from cache")
                return dict_to_stock_data(cached)

        stock_data = self.fmp.get_stock_data(symbol)

        if self.cache and stock_data:
            date_key = self.cache.get_date_key()
            self.cache.save("fundamentals", f"{symbol}_{date_key}", stock_data_to_dict(stock_data))

        return stock_data
