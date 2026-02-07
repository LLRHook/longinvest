import logging
from dataclasses import dataclass

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


class GrowthStrategy:
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
        """Get stock universe from FMP screener, sorted by market cap.

        Returns list of dicts with symbol and marketCap.
        """
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
        These just eliminate clearly broken companies.
        """
        failures = []

        # Revenue growth must exist and be positive
        if stock.revenue_growth is None or stock.revenue_growth <= 0:
            growth_str = f"{stock.revenue_growth:.1%}" if stock.revenue_growth is not None else "N/A"
            failures.append(f"No revenue growth: {growth_str}")

        # Free cash flow must be positive (if data available)
        if stock.free_cash_flow is not None and stock.free_cash_flow <= 0:
            failures.append(f"Negative FCF: ${stock.free_cash_flow:,.0f}")

        # D/E ratio must be < 1.5 (if data available)
        if stock.de_ratio is not None and stock.de_ratio > 1.5:
            failures.append(f"High D/E: {stock.de_ratio:.2f}")

        return len(failures) == 0, failures

    def score_stock(self, stock: StockData) -> ScoredStock:
        """Score a stock based on small cap growth metrics. Max 100 points.

        Scoring breakdown:
            Revenue growth:        30 pts
            FCF yield:             20 pts
            ROE:                   15 pts
            Gross margin:          10 pts
            FCF growth bonus:       5 pts
            Earnings momentum:     15 pts (EPS beats)
            Earnings growth:        5 pts
        """
        score = 0.0
        reasons = []

        # Revenue growth (30 pts) — top signal for small caps
        if stock.revenue_growth is not None and stock.revenue_growth > 0:
            growth_score = min(stock.revenue_growth * 100, 30)
            score += growth_score
            reasons.append(f"Revenue growth: {stock.revenue_growth:.1%}")

        # FCF yield (20 pts) — proves growth is sustainable
        if stock.free_cash_flow is not None and stock.free_cash_flow > 0:
            if stock.market_cap and stock.market_cap > 0:
                fcf_yield = stock.free_cash_flow / stock.market_cap
                fcf_score = min(fcf_yield * 400, 20)
                score += fcf_score
                reasons.append(f"FCF yield: {fcf_yield:.1%}")
            else:
                score += 12
                reasons.append("FCF positive (yield N/A)")

        # ROE (15 pts) — capital efficiency
        if stock.roe is not None and stock.roe > 0:
            roe_score = min(stock.roe * 75, 15)
            score += roe_score
            reasons.append(f"ROE: {stock.roe:.1%}")

        # Gross margin (10 pts) — pricing power, scored above 30%
        if stock.gross_margin is not None and stock.gross_margin > 0.3:
            margin_score = min((stock.gross_margin - 0.3) * 50, 10)
            score += margin_score
            reasons.append(f"Gross margin: {stock.gross_margin:.1%}")

        # FCF growth bonus (5 pts) — improving cash generation
        if stock.free_cash_flow_growth is not None and stock.free_cash_flow_growth > 0.10:
            score += 5
            reasons.append(f"FCF growth: {stock.free_cash_flow_growth:.1%}")

        # Earnings momentum (15 pts) — EPS beat frequency
        if stock.eps_beat_count is not None and stock.eps_beat_count > 0:
            # 3.75 pts per beat, max 15 for 4/4 beats
            earnings_momentum_score = stock.eps_beat_count * 3.75
            score += earnings_momentum_score
            reasons.append(f"EPS beats: {stock.eps_beat_count}/4 quarters")
            if stock.earnings_growth_accelerating:
                reasons.append("Earnings accelerating")

        # Earnings growth (5 pts) — net income / EPS growth
        if stock.earnings_growth is not None and stock.earnings_growth > 0:
            eg_score = min(stock.earnings_growth * 25, 5)
            score += eg_score
            reasons.append(f"Earnings growth: {stock.earnings_growth:.1%}")

        return ScoredStock(stock=stock, score=score, reasons=reasons)

    def screen(
        self,
        existing_symbols: set[str] | None = None,
    ) -> list[ScoredStock]:
        """Run the full screening process and return ranked stocks.

        Args:
            existing_symbols: Symbols to exclude (already held)
        """
        existing = existing_symbols or set()

        # Try to load cached scored results first
        if self.cache and not self.force_refresh:
            date_key = self.cache.get_date_key()
            cached_scored = self.cache.load("scored", f"scored_{date_key}")
            if cached_scored:
                scored_stocks = [dict_to_scored_stock(s) for s in cached_scored]
                # Filter out existing positions
                scored_stocks = [s for s in scored_stocks if s.stock.symbol not in existing]
                print(f"  Using cached screening results ({len(scored_stocks)} stocks)")
                logger.info(f"Loaded {len(scored_stocks)} scored stocks from cache")
                return scored_stocks

        # Get screener candidates (with caching)
        candidates = self._get_cached_screener_candidates()

        # Filter out already-held positions
        candidates = [c for c in candidates if c["symbol"] not in existing]
        logger.info(f"After filtering existing positions: {len(candidates)} candidates")

        logger.info(f"Analyzing {len(candidates)} candidates")

        scored_stocks: list[ScoredStock] = []

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

            scored = self.score_stock(stock_data)
            if scored.score > 0:
                scored_stocks.append(scored)
                print(f"score: {scored.score:.1f}")
            else:
                print("score: 0")

        # Sort by score descending
        scored_stocks.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Screening complete: {len(scored_stocks)} stocks passed")

        # Cache the scored results
        if self.cache:
            date_key = self.cache.get_date_key()
            cached_data = [scored_stock_to_dict(s) for s in scored_stocks]
            self.cache.save("scored", f"scored_{date_key}", cached_data)

        return scored_stocks

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

    def get_buy_recommendations(
        self,
        existing_symbols: set[str] | None = None,
        max_picks: int | None = None,
    ) -> list[ScoredStock]:
        """Get top stocks to buy."""
        existing = existing_symbols or set()

        scored = self.screen(existing_symbols=existing)
        recommendations = scored[:max_picks] if max_picks else scored

        logger.info(f"Recommending {len(recommendations)} stocks for purchase")
        return recommendations

    def get_sell_recommendations(
        self, held_symbols: set[str]
    ) -> list[tuple[str, str]]:
        """Review held positions and recommend sells for degraded fundamentals.

        Returns list of (symbol, reason) tuples for positions that should be sold.
        """
        sell_list: list[tuple[str, str]] = []

        for symbol in held_symbols:
            stock_data = self.fmp.get_stock_data(symbol)
            if not stock_data:
                sell_list.append((symbol, "Unable to fetch fundamentals"))
                continue

            passes, failures = self.passes_guardrails(stock_data)
            if not passes:
                sell_list.append((symbol, f"Failed guardrails: {', '.join(failures)}"))
                continue

            # Additional degradation checks beyond guardrails
            if stock_data.revenue_growth is not None and stock_data.revenue_growth < 0:
                sell_list.append((symbol, f"Revenue declining: {stock_data.revenue_growth:.1%}"))
            elif stock_data.free_cash_flow is not None and stock_data.free_cash_flow < 0:
                sell_list.append((symbol, f"FCF turned negative: ${stock_data.free_cash_flow:,.0f}"))

        return sell_list
