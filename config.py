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
    MAX_DE_RATIO: float = 1.5
    MIN_REVENUE_GROWTH: float = -0.10  # Allow 10% decline
    REQUIRE_POSITIVE_FCF: bool = True
    MIN_REVENUE: float = 10_000_000  # $10M minimum revenue
    MIN_AVG_DAILY_DOLLAR_VOLUME: float = 500_000

    # DCA (Dollar-Cost Averaging)
    DAILY_INVESTMENT: float = 5_000.0  # $5,000/day into best stock (paper trading) — legacy
    WEEKLY_INVESTMENT: float = 25_000.0  # $25,000/week spread across top 10 (Monday-only execution)
    TARGET_POSITIONS: int = 15
    NEW_POSITION_SCORE_THRESHOLD: float = 0.20  # 20% premium to open new position
    DCA_TOP_N: int = 5
    WEEKLY_TOP_N: int = 10  # Spread across top 10 candidates on Monday

    # Portfolio Limits
    OPTIMIZER_CANDIDATES: int = 30
    HISTORICAL_DAYS: int = 400  # ~273 trading days needed for 12-1 month momentum
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

    # Order Execution Strategy
    USE_LIMIT_ORDERS: bool = True  # Use limit orders instead of market orders
    LIMIT_ORDER_SPREAD_PCT: float = 0.005  # 0.5% above bid price as limit
    LIMIT_ORDER_FALLBACK_TO_MARKET: bool = True  # Fall back to market if limit not filled

    # Caching
    CACHE_DIR: str = "cache"
    CACHE_TTL_HOURS: int = 24  # 1 day TTL for all cached data

    # --- Multi-Timeframe Relative Strength (Task 9) ---
    RS_TIMEFRAMES: list = [63, 126, 252]             # 3mo, 6mo, 12mo lookback days
    RS_WEIGHTS: list = [0.25, 0.35, 0.40]            # Weight per timeframe

    # --- Volume Signals (Task 4) ---
    VOLUME_SURGE_THRESHOLD: float = 2.0              # 2x avg volume = surge
    VOLUME_SURGE_LOOKBACK: int = 20                  # 20-day avg volume baseline

    # --- Trailing Stops (Phase 3c / Task 5) ---
    TRAILING_STOP_ENABLED: bool = True
    TRAILING_STOP_ATR_MULTIPLIER: float = 2.5        # Stop distance = ATR * multiplier
    TRAILING_STOP_ATR_PERIOD: int = 14               # ATR lookback
    TRAILING_STOP_TIGHT_MULTIPLIER: float = 1.5      # Tighter stop after profit target
    TRAILING_STOP_PROFIT_THRESHOLD: float = 0.15     # 15% gain triggers tightening
    TRAILING_STOP_MIN_PCT: float = 0.03              # Floor: never tighter than 3%
    TRAILING_STOP_MAX_PCT: float = 0.15              # Ceiling: never wider than 15%

    # --- Intraday Momentum Check (Task 6) ---
    INTRADAY_CHECK_ENABLED: bool = True
    INTRADAY_MIN_CHANGE: float = -0.03               # Skip buy if stock down >3% intraday

    # --- Momentum Tilt (Task 14) ---
    MOMENTUM_TILT_ENABLED: bool = True
    MOMENTUM_TILT_FACTOR: float = 0.20               # 20% tilt toward momentum leaders

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
