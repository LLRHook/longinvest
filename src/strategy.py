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

    def get_screener_candidates(self, limit: int = 500) -> list[dict]:
        """Get stock universe from FMP screener, sorted by market cap.

        Returns list of dicts with symbol and marketCap.
        """
        stocks = self.fmp.get_screener_stocks(
            min_market_cap=Config.MIN_MARKET_CAP,
            limit=limit,
        )
        stocks.sort(key=lambda x: x.get("marketCap", 0), reverse=True)
        logger.info(f"FMP screener returned {len(stocks)} US stocks")
        return stocks

    def passes_guardrails(self, stock: StockData) -> tuple[bool, list[str]]:
        """Check if stock passes basic quality guardrails."""
        failures = []

        # P/E check: must be positive and reasonable
        if stock.pe_ratio is not None:
            if stock.pe_ratio <= 0:
                failures.append(f"Negative P/E: {stock.pe_ratio:.1f}")
            elif stock.pe_ratio > 100:
                failures.append(f"P/E too high: {stock.pe_ratio:.1f}")

        # D/E check: debt-to-equity should not be excessive
        if stock.de_ratio is not None and stock.de_ratio > 2.0:
            failures.append(f"High D/E: {stock.de_ratio:.2f}")

        # Current ratio: should have adequate liquidity
        if stock.current_ratio is not None and stock.current_ratio < 1.0:
            failures.append(f"Low current ratio: {stock.current_ratio:.2f}")

        return len(failures) == 0, failures

    def score_stock(self, stock: StockData) -> ScoredStock:
        """Score a stock based on growth metrics."""
        score = 0.0
        reasons = []

        # Revenue growth (weight: 30)
        if stock.revenue_growth is not None and stock.revenue_growth > 0:
            growth_score = min(stock.revenue_growth * 100, 30)
            score += growth_score
            reasons.append(f"Revenue growth: {stock.revenue_growth:.1%}")

        # EPS growth (weight: 30)
        if stock.eps_growth is not None and stock.eps_growth > 0:
            eps_score = min(stock.eps_growth * 100, 30)
            score += eps_score
            reasons.append(f"EPS growth: {stock.eps_growth:.1%}")

        # ROE (weight: 20)
        if stock.roe is not None and stock.roe > 0:
            roe_score = min(stock.roe * 100, 20)
            score += roe_score
            reasons.append(f"ROE: {stock.roe:.1%}")

        # Gross margin (weight: 10)
        if stock.gross_margin is not None and stock.gross_margin > 0.3:
            margin_score = min((stock.gross_margin - 0.3) * 50, 10)
            score += margin_score
            reasons.append(f"Gross margin: {stock.gross_margin:.1%}")

        # Operating margin (weight: 10)
        if stock.operating_margin is not None and stock.operating_margin > 0.1:
            op_margin_score = min((stock.operating_margin - 0.1) * 50, 10)
            score += op_margin_score
            reasons.append(f"Operating margin: {stock.operating_margin:.1%}")

        return ScoredStock(stock=stock, score=score, reasons=reasons)

    def screen(
        self,
        existing_symbols: set[str] | None = None,
        max_fundamental_analysis: int = 100,
    ) -> list[ScoredStock]:
        """Run the full screening process and return ranked stocks.

        Args:
            existing_symbols: Symbols to exclude (already held)
            max_fundamental_analysis: Limit detailed API calls to top N by market cap
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

        # Limit fundamental analysis to top N by market cap (saves API calls)
        candidates = candidates[:max_fundamental_analysis]
        logger.info(f"Analyzing top {len(candidates)} by market cap")

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
        """Get top stocks to buy, respecting max positions."""
        existing = existing_symbols or set()
        current_count = len(existing)
        slots_available = Config.MAX_POSITIONS - current_count

        if slots_available <= 0:
            logger.info(f"Already at max positions ({Config.MAX_POSITIONS})")
            return []

        picks = max_picks if max_picks else slots_available
        picks = min(picks, slots_available)

        scored = self.screen(existing_symbols=existing)
        recommendations = scored[:picks]

        logger.info(f"Recommending {len(recommendations)} stocks for purchase")
        return recommendations
