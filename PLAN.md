# Longinvest v2 — Improvement Plan

## Current Rating: 38/100

| Category | Score | Notes |
|----------|-------|-------|
| Factor model design | 7/10 | Quality + value + momentum is well-supported academically |
| Universe selection | 6/10 | Small-cap $300M-$2B is where factor premiums are widest |
| Position sizing | 2/10 | Single-stock daily concentration is the #1 problem |
| Rebalancing frequency | 2/10 | Daily turnover eats factor premium alive in transaction costs |
| Risk management | 4/10 | Circuit breakers good, but no vol-adjusted stops or position sizing |
| Data quality | 4/10 | FMP is budget-tier for small caps — stale/missing data common |
| Fraud/quality screening | 1/10 | No earnings manipulation or bankruptcy filters |
| Backtesting | 0/10 | No survivorship-bias-free testing — flying blind |
| Execution | 5/10 | Market orders at open, no VWAP/limit order strategy |
| Infrastructure | 7/10 | Clean code, good automation, Discord alerts |

### Three Critical Problems

1. **Single-stock concentration** — Fundamentals barely change day-to-day, so $5k/day piles into the same stock for days straight. Hits $25k+ in one small-cap before the 15% cap triggers. Enormous idiosyncratic risk for zero additional return.
2. **Daily turnover costs** — Research Affiliates found momentum factor returns are essentially zero after implementation costs (5.2% annualized shortfall). Daily rescoring + single-stock rotation generates maximum turnover.
3. **No fraud/distress filtering** — `REQUIRE_POSITIVE_FCF = False` and `MAX_DE_RATIO = 3.0` admits cash-burning, highly leveraged companies.

---

## Phase 1: Fix Critical Structural Issues

> Priority: HIGHEST — biggest impact on risk-adjusted returns

### 1a. Multi-stock buys instead of single-stock

Buy top 3-5 stocks each day ($1k-$1.7k each) instead of $5k into one. Or batch weekly: $25k spread across top 10 stocks every Monday. This alone probably doubles risk-adjusted returns.

**Changes:**
- Modify `strategy.get_dca_buy_target()` to return top N candidates instead of 1
- Update `cmd_execute()` to loop over candidates and split `DAILY_INVESTMENT` across them
- Add config: `DCA_TOP_N = 5`

### 1b. Volatility-adjusted position sizing

`optimizer.py` already computes `annual_volatility` per stock. Use it.

**Formula:** `allocation = target_risk / stock_volatility * budget`

A 60% vol biotech gets half the allocation of a 30% vol industrial. Equalizes risk contribution across positions.

**Changes:**
- Compute vol-weighted allocations in `cmd_execute()` after selecting top N
- Normalize so allocations sum to `DAILY_INVESTMENT`

### 1c. Tighten guardrails

| Parameter | Current | Proposed | Reason |
|-----------|---------|----------|--------|
| `MAX_DE_RATIO` | 3.0 | 1.5 | Exclude overleveraged junk |
| `REQUIRE_POSITIVE_FCF` | False | True | Or at minimum positive OCF |
| Min avg daily volume | None | $500k+ | Avoid illiquid names with wide spreads |
| Min analyst coverage | None | >= 1 | PEG ratio is meaningless without estimates |

---

## Phase 2: Integrate FinScan as Pre-Trade Risk Filter

> Priority: HIGH — we built the fraud/risk scoring system, we should use it

FinScan (at `/Users/victorivanov/Documents/personal projects/FinScan/`) provides Beneish M-Score, Altman Z-Score, Piotroski F-Score, accrual quality, red flags, and a composite risk score (0-100). Both projects use FMP as upstream data — FinScan adds the intelligence layer.

### 2a. Add FinScan API client to longinvest

- Call `GET /v1/scan/{ticker}` for each DCA candidate
- Cache results (FinScan already caches 24h server-side)
- Use own FinScan API key — we control the infrastructure and rate limits
- Consider calling FinScan's scoring engine directly as a library to skip HTTP overhead

### 2b. Gate buys on composite risk

| Composite Risk | Action |
|----------------|--------|
| HIGH (>70) | Hard reject |
| ELEVATED (50-70) | Reduce allocation by 50% |
| MODERATE/LOW (<50) | Proceed with full size |
| Beneish "LIKELY_MANIPULATOR" | Hard reject regardless |

### 2c. Add Piotroski F-Score to multi-factor scoring

Replace or supplement crude quality metrics with Piotroski's 9-criteria score:
- F-Score 7-9: quality premium in scoring model
- F-Score 0-3: reject

### 2d. Red flag monitoring on held positions

- Weekly cron: scan all held positions through FinScan
- Any new HIGH severity red flags: Discord alert for manual review
- Altman Z-Score entering distress zone: auto-sell or tighten stop

---

## Phase 3: Improve Execution & Reduce Costs

> Priority: MEDIUM — meaningful cost savings, moderate implementation effort

### 3a. Switch to weekly rebalancing

Run screener Monday morning, execute buys once per week. Reduces turnover by ~80%, saves 2-5% annualized in transaction costs.

**Changes:**
- Update cron to Monday-only execution
- Increase `DAILY_INVESTMENT` to `WEEKLY_INVESTMENT = 25_000`
- Spread across top 10 candidates

### 3b. Limit orders instead of market orders

Place limit orders at the bid price with 1-hour expiry. Small caps have wide spreads; market orders give up 0.3-1% per trade.

**Changes:**
- Add `broker.buy_limit()` method
- Monitor fill status, cancel unfilled after timeout
- Fall back to market order for must-fill situations

### 3c. ATR-based trailing stops (replace fixed %)

Use 3x ATR(20) as trailing stop distance. Automatically wider for volatile stocks, tighter for calm ones. Research shows ~15% improvement vs fixed stops, ~32% less max drawdown.

**Rules:**
- Only activate stops on fully-built positions, not during accumulation
- Requires fetching ATR data (already have price history infrastructure)

---

## Phase 4: Backtest & Validate

> Priority: HIGH (but sequenced after structural fixes so there's something worth testing)

### 4a. Get survivorship-bias-free data

Options:
- **Sharadar / Nasdaq Data Link** — point-in-time fundamentals including delisted companies (paid)
- **SimFin** — free, decent historical coverage
- **QuantConnect / Zipline** — backtesting platforms with point-in-time data

### 4b. Backtest the full v2 strategy

- Walk-forward validation: train on 5 years, test on 1, roll forward
- Include realistic transaction costs (50bps per trade for small caps)
- Compare against AVUV (Avantis US Small Cap Value ETF) benchmark
- Must beat AVUV net of costs or the system isn't worth running

---

## Phase 5: FinScan Production Hardening

> Priority: LOW (only if using FinScan as shared infra for longinvest)

### 5a. Fix HIGH severity issues

From FinScan audit (Feb 25, 2026):
- NullPointerException chains in ScanService (line 103)
- Missing `@Transactional` on auth/billing multi-step DB operations
- Division by zero in CompositeRiskCalculator (line 131)

### 5b. Internal API tier

- Create an internal/unlimited tier for longinvest (bypass rate limits)
- Or extract FinScan's scoring engine as a shared Python module callable directly

---

## Implementation Priority

| Order | Phase | Effort | Impact | Target Rating |
|-------|-------|--------|--------|---------------|
| 1 | 1a — Multi-stock buys | Small (config + minor code) | Very high | 45/100 |
| 2 | 2a-2b — FinScan risk filter | Medium (new API client) | High | 52/100 |
| 3 | 3a — Weekly rebalancing | Small (cron + config) | High | 58/100 |
| 4 | 1b — Vol-adjusted sizing | Medium (new sizing logic) | Medium | 62/100 |
| 5 | 1c — Tighter guardrails | Small (config changes) | Medium | 65/100 |
| 6 | 4 — Backtest & validate | Large (new data + framework) | Critical for confidence | 65/100 (validated) |
| 7 | 2c-2d — Piotroski + monitoring | Medium | Medium | 70/100 |
| 8 | 3b-3c — Limit orders + ATR stops | Medium | Medium | 75/100 |
| 9 | 5 — FinScan hardening | Medium | Low (for longinvest) | 75/100 |

---

## Key Research References

- Aberdeen: Multi-Factor — Why It Takes Value, Quality, and Momentum
- SSGA: Small Caps More Than Just a Factor Premium (March 2025)
- Research Affiliates: The Incredible Shrinking Factor Return (5.2% implementation shortfall)
- CFA Institute: Implementation Shortfalls Hamstring Factor Strategies
- Kaminski and Lo (2008): Stop-loss effectiveness — helps trend-following, harms mean-reversion
- MSCI: Small Caps — No Small Oversight (survivorship bias as main source of size premium)
- O'Shaughnessy: Microcaps — Factor Spreads, Structural Biases
- Quant Investing: Advanced Trailing Stop Loss Techniques (ATR-based 15% improvement)
