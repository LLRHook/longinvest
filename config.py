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

    # Strategy
    MAX_POSITIONS: int = 10
    MIN_MARKET_CAP: float = 300_000_000  # $300M (small cap)
    MAX_MARKET_CAP: float = 2_000_000_000  # $2B (small cap ceiling)
    MAX_FUNDAMENTAL_ANALYSIS: int = 100

    # Portfolio Optimization
    INVESTMENT_BUDGET: float = 5000.0
    OPTIMIZER_CANDIDATES: int = 15
    MIN_ALLOCATION_THRESHOLD: float = 0.05  # 5%
    HISTORICAL_DAYS: int = 365
    MIN_HISTORICAL_DAYS: int = 200

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
