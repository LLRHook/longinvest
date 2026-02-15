import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Alpaca (Paper Trading)
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_PAPER: bool = True

    # Discord Notifications
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "true").lower() == "true"

    # FMP
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
    FMP_BASE_URL: str = "https://financialmodelingprep.com/stable"
    FMP_RATE_LIMIT_MS: int = 200  # Starter tier: 300 calls/min
    FMP_MAX_CONCURRENT: int = 20

    # Strategy
    MIN_MARKET_CAP: float = 300_000_000  # $300M (small cap)
    MAX_MARKET_CAP: float = 2_000_000_000  # $2B (small cap ceiling)

    # Guardrails
    MAX_DE_RATIO: float = 3.0
    MIN_REVENUE_GROWTH: float = -0.10  # Allow 10% decline
    REQUIRE_POSITIVE_FCF: bool = False
    MIN_REVENUE: float = 10_000_000  # $10M minimum revenue

    # DCA (Dollar-Cost Averaging)
    DAILY_INVESTMENT: float = 50.0  # $50/day into best stock
    TARGET_POSITIONS: int = 15
    NEW_POSITION_SCORE_THRESHOLD: float = 0.20  # 20% premium to open new position

    # Portfolio Limits
    OPTIMIZER_CANDIDATES: int = 30
    HISTORICAL_DAYS: int = 365
    MIN_HISTORICAL_DAYS: int = 100
    MAX_SINGLE_POSITION_PCT: float = 0.15  # 15% max per stock
    MAX_SECTOR_ALLOCATION: float = 0.35  # 35% max per sector

    # Technical Filters
    SMA_TREND_PERIOD: int = 50
    RSI_OVERBOUGHT: float = 75.0

    # Circuit Breaker
    CIRCUIT_BREAKER_PCT: float = -0.08  # Halt if portfolio down > 8% today
    MARKET_CIRCUIT_BREAKER_PCT: float = -0.04  # Halt if SPY down > 4% today

    # Earnings Calendar
    EARNINGS_BLACKOUT_DAYS: int = 5
    EARNINGS_BOOST_DAYS: int = 10

    # Caching
    CACHE_DIR: str = "cache"
    CACHE_TTL_HOURS: int = 24  # 1 day TTL for all cached data

    @classmethod
    def validate(cls) -> list[str]:
        """Return list of missing required config values."""
        missing = []
        if not cls.ALPACA_API_KEY:
            missing.append("ALPACA_API_KEY")
        if not cls.ALPACA_SECRET_KEY:
            missing.append("ALPACA_SECRET_KEY")
        if not cls.FMP_API_KEY:
            missing.append("FMP_API_KEY")
        return missing
