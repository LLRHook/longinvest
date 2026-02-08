# Longinvest

Automated small-cap growth stock screening and trading bot. Screens US small-cap stocks ($300M-$2B market cap), scores candidates on growth fundamentals and momentum, optimizes portfolio allocations via Sharpe ratio, and executes trades through Alpaca paper trading. Reports daily performance to Discord with charts.

## Features

- **Growth Stock Screener**: Screens up to 2,000 small-cap US stocks from FMP, scores on a 100-point system
- **Portfolio Optimization**: Sharpe ratio maximization with position/sector constraints and momentum tilt
- **Parallelized API Calls**: Concurrent FMP requests with thread-safe token-bucket rate limiter (300 calls/min)
- **Tiered Risk Management**: 12% trailing stops for new positions, tightened to 8% for winners up >20%
- **Circuit Breaker**: Halts trading if portfolio down >5% or SPY down >3%
- **Intraday Momentum Check**: Skips buys on stocks down >3% on execution day
- **Position Rebalancing**: Trims overweight positions when drift exceeds 5% from target
- **Earnings Awareness**: Blackout period before earnings, scoring boost after strong beats
- **Technical Filters**: SMA-50 trend, RSI overbought, multi-timeframe relative strength
- **Discord Notifications**: Trade execution, daily reports with charts, screening results, circuit breaker alerts
- **Automated Scheduling**: GitHub Actions with NYSE holiday detection and auto-DST handling

## Requirements

- Python 3.11+
- [Alpaca](https://alpaca.markets) paper trading account
- [Financial Modeling Prep](https://financialmodelingprep.com) API key (Starter tier recommended: 300 calls/min)
- Discord webhook URL (optional, for notifications)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/LLRHook/longinvest.git
   cd longinvest
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your API keys:
   ```
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_SECRET_KEY=your_alpaca_secret
   FMP_API_KEY=your_fmp_key
   DISCORD_WEBHOOK_URL=your_webhook_url
   ENABLE_NOTIFICATIONS=true
   ```

## Usage

```bash
python main.py              # Execute full trading cycle
python main.py --dry-run    # Simulate without trading
python main.py --screen     # Run screener only
python main.py --report     # Send daily performance report
python main.py --status     # Show portfolio summary
python main.py --clear-cache  # Clear cached data
```

Add `--debug` to any command for verbose logging. Add `--force-refresh` to bypass cache.

## How It Works

### Execution Pipeline

1. **Circuit breaker check** - Halt if portfolio or market is crashing
2. **Sell review** - Check held positions for degraded fundamentals
3. **Rebalance** - Trim overweight positions
4. **Screen** - Fetch 2,000 candidates, apply guardrails, score on 100-point system
5. **Technical filters** - SMA-50 trend, RSI, momentum scores, relative strength
6. **Optimize** - Sharpe ratio maximization with momentum tilt
7. **Intraday check** - Skip stocks dumping on execution day
8. **Execute** - Place market buys and trailing stop orders
9. **Tighten stops** - Upgrade stops on profitable positions

### Scoring System (100 points)

| Metric | Weight | Signal |
|--------|--------|--------|
| Revenue growth | 30 | Top growth signal |
| EPS beats (4Q) | 20 | Momentum / execution |
| Earnings growth | 15 | Profit trajectory |
| Revenue acceleration | 10 | Growth rate increasing |
| Gross margin | 10 | Pricing power |
| FCF yield | 5 | Sustainability |
| ROE | 5 | Capital efficiency |
| Earnings acceleration | 5 | Accelerating profits |

Relative strength multiplier (1/3/6 month) adjusts final scores by up to +/-30%.

### Guardrails

- Revenue > $10M (filters penny stocks)
- Revenue growth > -10% (allows cyclical dips)
- D/E ratio < 3.0
- Price above SMA-50
- RSI < 75 (not overbought)
- Not within 5 days of earnings

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_MARKET_CAP` | $300M | Small-cap floor |
| `MAX_MARKET_CAP` | $2B | Small-cap ceiling |
| `MAX_POSITIONS` | 10 | Max portfolio positions |
| `MAX_SINGLE_POSITION_PCT` | 25% | Per-stock cap |
| `MAX_SECTOR_ALLOCATION` | 60% | Per-sector cap |
| `TRAILING_STOP_PCT` | 12% | Default trailing stop |
| `TRAILING_STOP_TIGHT_PCT` | 8% | Tight stop for winners |
| `CIRCUIT_BREAKER_PCT` | -5% | Portfolio halt threshold |
| `SMA_TREND_PERIOD` | 50 | Trend filter period |
| `MOMENTUM_TILT_FACTOR` | 0.20 | Optimizer momentum bias |
| `REBALANCE_THRESHOLD` | 5% | Drift trigger |
| `EARNINGS_BLACKOUT_DAYS` | 5 | Pre-earnings blackout |

## Project Structure

```
longinvest/
├── main.py                          # CLI entry point and orchestrator
├── config.py                        # All configurable parameters
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
├── ROADMAP.md                       # Development roadmap
├── .github/workflows/trading.yml    # Automated daily runs
└── src/
    ├── broker.py      # Alpaca integration (orders, positions, stops)
    ├── cache.py       # JSON-based caching with TTL
    ├── charter.py     # Performance charts (portfolio vs SPY)
    ├── data.py        # FMP API client (parallelized, rate-limited)
    ├── notifier.py    # Discord webhooks and embeds
    ├── optimizer.py   # Sharpe ratio optimization with momentum tilt
    ├── reporter.py    # Daily performance and risk metrics
    ├── strategy.py    # Screening, scoring, and sell logic
    ├── technicals.py  # SMA, RSI, momentum, relative strength
    └── tracker.py     # Trade history persistence
```

## Automated Scheduling

The bot runs via GitHub Actions on weekdays:

| Time (ET) | Action | Description |
|-----------|--------|-------------|
| 9:30 AM | `execute` | Screen, optimize, and trade |
| 1:00 PM | `dry-run` | Midday signal check |
| 4:30 PM | `report` | Daily performance report to Discord |

Automatically skips NYSE holidays via `exchange_calendars`. Handles EST/EDT transitions with dual cron entries.

Manual runs available via GitHub Actions `workflow_dispatch`.
