# Longinvest

An automated multi-factor stock screening and DCA trading bot built as a learning project. Screens US small-cap stocks ($300M-$2B market cap), scores candidates on growth fundamentals and momentum, optimizes portfolio allocations via inverse-volatility weighting, and executes trades through Alpaca paper trading. Reports daily performance to Discord with charts.

> **Project Status: Concluded.** See [Conclusion](#conclusion) for findings.

## Features

- **Multi-Factor Screener**: Screens up to 2,000 small-cap US stocks from FMP, scores on a 100-point system combining growth, quality, and momentum factors
- **Inverse-Volatility Position Sizing**: Lower-volatility stocks receive proportionally larger allocations
- **Momentum Tilt**: Allocation adjustment toward high-momentum leaders using percentile-ranked 12-1 month signals
- **Parallelized API Calls**: Concurrent FMP requests with thread-safe token-bucket rate limiter (300 calls/min)
- **ATR-Based Trailing Stops**: Dynamic stop distances computed from Average True Range, tightened on profitable positions
- **Circuit Breaker**: Halts trading if portfolio or market drops beyond configurable thresholds
- **Intraday Momentum Check**: Skips buys on stocks falling significantly on execution day
- **Earnings Awareness**: Blackout period before earnings, scoring boost after strong beats
- **Technical Filters**: SMA-50 trend, RSI overbought, multi-timeframe relative strength
- **DCA Deposit Tracking**: Records weekly deposits with SPY price for fair benchmark comparison
- **Discord Notifications**: Trade execution, daily reports with charts, screening results, circuit breaker alerts
- **Automated Scheduling**: GitHub Actions with NYSE holiday detection and auto-DST handling

## Requirements

- Python 3.11+
- [Alpaca](https://alpaca.markets) paper trading account
- [Financial Modeling Prep](https://financialmodelingprep.com) API key (Starter tier: 300 calls/min, ~$30/month)
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
python main.py                # Execute full trading cycle
python main.py --dry-run      # Simulate without trading
python main.py --screen       # Run screener only
python main.py --report       # Send daily performance report
python main.py --status       # Show portfolio summary
python main.py --reset        # Wipe account and clear all tracking data
python main.py --manual-buy   # One-time buy with fixed allocation ratios
python main.py --clear-cache  # Clear cached data
```

Add `--debug` to any command for verbose logging. Add `--force-refresh` to bypass cache.

## How It Works

### Execution Pipeline

1. **Circuit breaker check** — Halt if portfolio or market is crashing
2. **Screen** — Fetch 2,000 candidates, apply guardrails, score on 100-point system
3. **Technical filters** — SMA-50 trend, RSI, momentum percentiles, relative strength
4. **Inverse-vol allocation** — Size positions inversely proportional to volatility
5. **Momentum tilt** — Adjust allocations toward high-momentum leaders
6. **Intraday check** — Skip stocks dumping on execution day
7. **Execute** — Place limit/market orders with fallback logic
8. **Trailing stops** — Place ATR-based stops, tighten on profitable positions
9. **Record deposit** — Track investment for SPY DCA benchmark comparison

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
| `TRAILING_STOP_ATR_MULTIPLIER` | 2.5 | Stop distance = ATR x multiplier |
| `TRAILING_STOP_PROFIT_THRESHOLD` | 15% | Gain threshold for tightening |
| `CIRCUIT_BREAKER_PCT` | -8% | Portfolio halt threshold |
| `MOMENTUM_TILT_FACTOR` | 0.20 | Optimizer momentum bias |
| `INTRADAY_MIN_CHANGE` | -3% | Intraday skip threshold |
| `EARNINGS_BLACKOUT_DAYS` | 5 | Pre-earnings blackout |

## Project Structure

```
longinvest/
├── main.py                          # CLI entry point and orchestrator
├── config.py                        # All configurable parameters
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
├── .github/workflows/trading.yml    # Automated scheduling
├── src/
│   ├── broker.py      # Alpaca integration (orders, positions, stops)
│   ├── cache.py       # JSON-based caching with TTL
│   ├── charter.py     # Performance charts (portfolio vs SPY)
│   ├── data.py        # FMP API client (parallelized, rate-limited)
│   ├── deposits.py    # DCA deposit tracking and SPY benchmark
│   ├── notifier.py    # Discord webhooks and embeds
│   ├── optimizer.py   # Inverse-vol weighting with momentum tilt
│   ├── reporter.py    # Daily performance and risk metrics
│   ├── strategy.py    # Screening, scoring, and sell logic
│   ├── technicals.py  # SMA, RSI, momentum, ATR, relative strength
│   └── tracker.py     # Trade history persistence
└── tests/             # 155 tests (no API keys needed)
```

## Automated Scheduling

The bot runs via GitHub Actions:

| Time (ET) | Day | Action | Description |
|-----------|-----|--------|-------------|
| 9:30 AM | Monday | `execute` | Weekly DCA — screen, allocate, and trade |
| 4:30 PM | Weekdays | `report` | Daily performance report to Discord |

Automatically skips NYSE holidays via `exchange_calendars`. Handles EST/EDT transitions with dual cron entries. Manual runs (dry-run, execute, reset, manual-buy) available via `workflow_dispatch`.

## Conclusion

This project was built as a learning exercise in quantitative trading system design. After completing the full implementation — screening, scoring, optimization, execution, risk management, and automated deployment — the conclusion is clear:

**For most individual investors, this strategy is unlikely to beat a simple index fund.**

### What Was Learned

**On system design:**
- Building a multi-factor scoring pipeline from scratch teaches you exactly how quantitative funds think about stock selection — and how many moving pieces are involved
- Inverse-volatility weighting is elegant in theory but in practice just concentrates your portfolio into the least-volatile (often least-interesting) names
- ATR-based trailing stops, circuit breakers, and intraday checks add layers of protection, but also layers of complexity that each need to be right
- Automating trades via GitHub Actions + Alpaca is surprisingly straightforward — the infrastructure is the easy part

**On stock selection:**
- Small-cap screeners ($300M-$2B) tend to surface obscure, thinly-traded stocks — biotech, specialty finance, closed-end funds — not undiscovered quality companies
- Quantitative filters catch obvious junk but can't assess narrative, management quality, or competitive dynamics
- The screener selected stocks with wide bid-ask spreads and low liquidity, which erodes returns through transaction friction alone

**On the math:**
- ~85-90% of professionally managed active funds fail to beat the S&P 500 over 10+ years, despite having teams, real-time data, and institutional access. A rules-based screener running on delayed financial data has no structural edge
- Data costs matter at small scale. A $30/month API subscription represents a significant drag on modest portfolios — capital that could otherwise be compounding in an index fund
- Splitting small weekly investments across 5 stocks creates negligible positions where transaction friction (spreads, fractional share handling) consumes a meaningful percentage of each trade

**On what would be needed to make this viable:**
- Significantly larger capital base where data costs become negligible as a percentage of AUM
- A better universe — mid/large caps with real liquidity, not micro-cap biotech
- Longer holding periods — weekly rebalancing creates unnecessary churn
- Rigorous backtesting against 10+ years of historical data before deploying real capital
- A genuine informational or analytical edge that the broader market doesn't already price in

**The takeaway:**

The most valuable output of this project isn't the trading bot — it's the understanding of *why* passive index investing works for most people. Building the system from scratch makes it viscerally clear how many things need to go right for active management to beat the market, and how the simplest approach (buy a broad index, hold, keep adding) is also the hardest to beat.

If you're considering building something similar: do it for the learning, not for the returns.
