# Longinvest Aggressive Growth Overhaul — Roadmap

## Current State Summary

The bot screens US small-cap stocks ($300M–$2B), scores them on growth fundamentals, optimizes allocations via Sharpe ratio, executes through Alpaca paper trading, and reports to Discord. It runs daily via GitHub Actions (execute at market open, report at market close).

**Current stance: moderate-conservative.** The guardrails (positive FCF, D/E < 1.5, positive revenue growth), 20% trailing stops, SMA-200 trend filter, and RSI < 70 overbought filter all lean defensive. The scoring weights quality metrics (FCF yield, ROE, gross margin) nearly as much as pure growth (revenue growth, EPS beats).

**FMP subscription: 300 calls/minute.** Currently rate-limited to 200ms between calls (5 calls/sec = 300/min). The code is sequential — one call at a time — so it never approaches the 300/min ceiling. With parallelization we can process stocks 10-50x faster.

---

## Task List (Agent-Compatible)

Each task is self-contained with defined inputs/outputs, can be done in any order (unless a dependency is noted), and merges cleanly because tasks touch different files or well-isolated sections.

---

### TASK 1: Parallelize FMP API calls with concurrent.futures

**Files:** `src/data.py`
**Depends on:** Nothing
**Risk:** Low (internal refactor, same outputs)

**Current problem:** `get_stock_data()` makes 6 sequential API calls per stock with 200ms delay each = 1.2s/stock. For 200 candidates that's ~4 minutes just on fundamentals. `get_historical_prices_batch()` is also sequential.

**What to do:**
1. Add `concurrent.futures.ThreadPoolExecutor` to `FMPClient`
2. In `get_stock_data()`, fire all 6 API calls (profile, quote, ratios, metrics, growth, surprises) concurrently using `executor.submit()` — they're independent
3. In `get_historical_prices_batch()`, fetch all symbols concurrently with a pool of 10-20 workers
4. Replace the 200ms fixed delay with a `threading.Semaphore` or token-bucket rate limiter that enforces 300 calls/min globally across threads (5 calls/sec)
5. Keep the 429 retry logic per-thread
6. Update `FMP_RATE_LIMIT_MS` in `config.py` — either remove it or lower it to reflect the new concurrent approach. Add `FMP_MAX_CONCURRENT: int = 20` to config

**Acceptance criteria:**
- Full screening cycle completes in under 60 seconds for 200 stocks (vs ~4 min today)
- API call count stays the same (no extra calls)
- Rate limiter never exceeds 300 calls/min
- All existing CLI commands produce identical results

---

### TASK 2: Loosen guardrails for aggressive growth

**Files:** `src/strategy.py`, `config.py`
**Depends on:** Nothing
**Risk:** Medium (changes which stocks qualify)

**Current guardrails that filter too aggressively for a growth strategy:**
- Revenue growth must be > 0% — filters pre-revenue pivots and cyclical dips
- FCF must be positive — eliminates high-growth SaaS burning cash to scale
- D/E < 1.5 — excludes leveraged growth plays

**What to change:**

In `config.py`, add new aggressive-mode parameters:
```python
# Guardrails
MAX_DE_RATIO: float = 3.0          # Was 1.5 (hardcoded in strategy.py)
MIN_REVENUE_GROWTH: float = -0.10  # Allow 10% decline (was 0%)
REQUIRE_POSITIVE_FCF: bool = False  # Was True
```

In `strategy.py` `passes_guardrails()`:
- Change revenue growth check: allow up to `Config.MIN_REVENUE_GROWTH` decline (not strictly > 0)
- Change FCF check: skip if `Config.REQUIRE_POSITIVE_FCF` is False
- Change D/E check: use `Config.MAX_DE_RATIO` instead of hardcoded 1.5
- Add new guardrail: **minimum revenue** ($10M+) to filter penny-stock traps

**Acceptance criteria:**
- More stocks pass screening (expect 30-50% more candidates)
- Guardrail thresholds come from config, not hardcoded
- Existing tests (if any) still pass

---

### TASK 3: Reweight scoring for aggressive momentum

**Files:** `src/strategy.py`
**Depends on:** Nothing
**Risk:** Medium (changes portfolio composition)

**Current scoring (100 points):**
| Metric | Current Weight | Problem |
|--------|---------------|---------|
| Revenue growth | 30 | Good, keep high |
| FCF yield | 20 | Over-rewards value, penalizes growth |
| ROE | 15 | Mature-company metric |
| Gross margin | 10 | Good |
| FCF growth bonus | 5 | Too small to matter |
| EPS beats | 15 | Good momentum signal |
| Earnings growth | 5 | Under-weighted |

**New aggressive scoring (100 points):**
| Metric | New Weight | Rationale |
|--------|-----------|-----------|
| Revenue growth | 30 | Keep — top signal |
| Revenue acceleration | 10 | NEW — QoQ growth rate increasing |
| EPS beats | 20 | Increase — strongest momentum signal |
| Earnings growth | 15 | Increase — directly rewards profit growth |
| Gross margin | 10 | Keep — pricing power filter |
| FCF yield | 5 | Decrease — less relevant for growth |
| ROE | 5 | Decrease — less relevant for early growth |
| Earnings acceleration bonus | 5 | Keep existing flag, give it real points |

**What to change in `score_stock()`:**
1. Reweight per table above
2. Add revenue acceleration: compare most recent quarter's revenue growth to average of prior 3 (requires `get_financial_growth(symbol, limit=4)` which is already fetched)
3. Revenue acceleration needs a new `StockData` field: `revenue_growth_accelerating: bool | None = None`
4. Compute it in `data.py` `get_stock_data()` using the existing `growth` array (same pattern as `earnings_growth_accelerating`)
5. Make earnings acceleration worth real points instead of just a `reasons` annotation

**Acceptance criteria:**
- Scoring totals still max out at 100
- High-growth, high-momentum stocks rank higher than stable-but-slow growers
- `StockData` backward-compat maintained (new field defaults to `None`)
- Update `dict_to_stock_data()` in `cache.py` with `.setdefault("revenue_growth_accelerating", None)`

---

### TASK 4: Add price momentum and volatility scoring to technicals

**Files:** `src/technicals.py`
**Depends on:** Nothing
**Risk:** Low (additive filter, doesn't change existing logic)

**Currently:** Only filters OUT bad stocks (below SMA-200, RSI > 70). Doesn't reward good momentum.

**What to add:**
1. **Price momentum score** — 3-month return percentile rank across candidates. Stocks in top quartile get a bonus multiplier
2. **Volatility-adjusted momentum** — return / volatility ratio (mini Sharpe on 90 days)
3. **Volume surge detection** — flag stocks with recent volume > 2x 50-day average (institutional interest signal)

**Implementation:**
- Add `compute_momentum_scores(prices_df, lookback=63)` → returns dict of `{symbol: momentum_score}`
- Add `compute_volume_signals(prices_df, volumes_df)` → requires extending `get_historical_prices()` to also return volume (currently only returns `close`)
- Return momentum scores alongside the filtered DataFrame so `main.py` can pass them to the optimizer as a tilt factor

**Note on volume data:** `get_historical_prices()` in `data.py` already receives volume from FMP but drops it at line 232: `return df[["close"]]`. Change to `return df[["close", "volume"]]` and update downstream consumers.

**Acceptance criteria:**
- `apply_technical_filters()` returns an additional `momentum_scores: dict[str, float]` in its return tuple
- Existing SMA-200 and RSI filters unchanged
- Volume data preserved through the pipeline

---

### TASK 5: Tighten trailing stops and add tiered stop-loss

**Files:** `src/broker.py`, `config.py`, `main.py`
**Depends on:** Nothing
**Risk:** Medium (changes risk management behavior)

**Current:** Fixed 20% trailing stop on all positions. For an aggressive strategy this is too loose — a stock can drop 20% before selling.

**What to change:**

In `config.py`:
```python
TRAILING_STOP_PCT: float = 0.12          # Tighten from 20% to 12%
TRAILING_STOP_TIGHT_PCT: float = 0.08    # For positions up > 20%
PROFIT_TARGET_TIGHTEN_PCT: float = 0.20  # Threshold to switch to tight stop
```

In `main.py` `cmd_execute()` Step 4 (after buying):
1. After placing a buy, check if any existing positions have appreciated > `PROFIT_TARGET_TIGHTEN_PCT`
2. For those, cancel the old trailing stop and place a tighter one at `TRAILING_STOP_TIGHT_PCT`
3. This locks in gains on winners while letting newer positions breathe

In `broker.py`, add:
- `cancel_open_orders(symbol)` — cancels existing trailing stops before placing new ones
- `get_open_orders(symbol)` — list open orders for a symbol

**Acceptance criteria:**
- New buys get 12% trailing stop
- Positions up > 20% get 8% trailing stop (dynamically tightened)
- Old trailing stops are cancelled before new ones are placed
- Config values are not hardcoded

---

### TASK 6: Add intraday momentum check before execution

**Files:** `main.py`, `src/data.py`
**Depends on:** Nothing
**Risk:** Low (additive check, can be disabled via config)

**Purpose:** Don't buy into a stock that's dumping on execution day. Currently the bot buys based on daily-close data with no awareness of what's happening intraday.

**What to add:**

In `config.py`:
```python
INTRADAY_MIN_CHANGE: float = -0.03  # Skip if stock is down > 3% today
ENABLE_INTRADAY_CHECK: bool = True
```

In `main.py` `cmd_execute()`, between optimization (Step 3) and execution (Step 4):
1. For each symbol in the allocation, fetch a real-time quote via `fmp.get_quote(symbol)`
2. Compare `price` vs `previousClose` (both fields are in FMP quote response)
3. If the stock is down more than `INTRADAY_MIN_CHANGE`, skip the buy and log the reason
4. Redistribute that allocation proportionally among remaining stocks

**Acceptance criteria:**
- Stocks crashing on execution day are skipped
- Reallocation math is correct (remaining allocations sum to 1.0)
- Feature can be disabled via `ENABLE_INTRADAY_CHECK = False`
- Console output clearly shows skipped stocks and reasons

---

### TASK 7: Reduce SMA filter from 200 to 50 for faster signals

**Files:** `src/technicals.py`, `config.py`
**Depends on:** Nothing
**Risk:** Medium (more volatile stocks pass through)

**Current:** SMA-200 filter requires price > 200-day moving average. This is a very slow trend signal that lags badly for small caps.

**What to change:**

In `config.py`:
```python
SMA_TREND_PERIOD: int = 50              # Was 200 (hardcoded in technicals.py)
RSI_OVERBOUGHT: float = 75.0           # Was 70 — slightly more permissive
MIN_HISTORICAL_DAYS: int = 100          # Was 200 — since we only need SMA-50 now
```

In `technicals.py` `apply_technical_filters()`:
- Use `Config.SMA_TREND_PERIOD` instead of default `sma_period=200`
- Use `Config.RSI_OVERBOUGHT` instead of default `rsi_overbought=70.0`

In `main.py` and `data.py`:
- Update `Config.MIN_HISTORICAL_DAYS` references (currently 200, can reduce to 100)
- This allows more recently-IPO'd stocks to pass the data requirement

**Acceptance criteria:**
- Faster trend detection catches growth stocks earlier
- More small caps pass data requirements
- All parameters come from config, not hardcoded

---

### TASK 8: Increase position concentration limits

**Files:** `config.py`
**Depends on:** Nothing
**Risk:** Medium (higher concentration = higher risk)

**Current:** 15% max per position, 40% max per sector, 5% minimum allocation.

**What to change:**
```python
MAX_SINGLE_POSITION_PCT: float = 0.25   # Was 0.15 — let winners take more
MAX_SECTOR_ALLOCATION: float = 0.60     # Was 0.40 — allow tech/health concentration
MIN_ALLOCATION_THRESHOLD: float = 0.03  # Was 0.05 — allow smaller positions
OPTIMIZER_CANDIDATES: int = 20          # Was 15 — wider funnel
```

**Rationale:**
- Aggressive strategies benefit from conviction sizing
- Technology and healthcare dominate small-cap growth; 40% cap forces artificial diversification
- Lowering min allocation from 5% to 3% allows the optimizer to include marginal positions
- 20 candidates gives the optimizer more choices

**Acceptance criteria:**
- Config changes only, no code changes needed
- Optimizer still respects the new (wider) constraints

---

### TASK 9: Add multi-timeframe relative strength scoring

**Files:** `src/technicals.py`, `src/strategy.py`
**Depends on:** Task 4 (momentum scores)
**Risk:** Low (additive scoring layer)

**Purpose:** Rank stocks by relative strength across 1-month, 3-month, and 6-month windows. This is the single most predictive momentum factor in academic literature.

**What to add:**

New function in `technicals.py`:
```python
def compute_relative_strength(
    prices_df: pd.DataFrame,
    periods: list[int] = [21, 63, 126],  # 1mo, 3mo, 6mo
    weights: list[float] = [0.4, 0.35, 0.25],
) -> dict[str, float]:
```

For each stock:
1. Compute return over each period
2. Percentile-rank each return across all candidates
3. Weighted average of ranks = composite relative strength score (0–100)

**Integration in `strategy.py`:**
- After fundamental scoring, apply a relative-strength tilt: `final_score = fundamental_score * (0.7 + 0.3 * relative_strength / 100)`
- This boosts high-momentum stocks by up to 30% and penalizes laggards by up to 30%

**Acceptance criteria:**
- Relative strength scores computed from price data (no extra API calls)
- Integrated into strategy scoring as a multiplier
- Stocks with strong price momentum rank higher in final recommendations

---

### TASK 10: Add sector-aware execution with FMP batch endpoints

**Files:** `src/data.py`, `src/strategy.py`
**Depends on:** Task 1 (parallel API calls)
**Risk:** Low (optimization of existing flow)

**Purpose:** Use FMP batch endpoints to reduce API calls. Currently we make 6 calls per stock. FMP has batch endpoints that return data for multiple symbols in one call.

**What to investigate and implement:**
1. Check if FMP `/stable/profile` supports comma-separated symbols (e.g., `?symbol=AAPL,MSFT,GOOG`)
2. If batch endpoints exist, refactor `get_stock_data()` to use batch fetches for groups of 10-20 symbols
3. If not, the parallelization from Task 1 is the primary optimization

**FMP batch endpoints to check:**
- `quote` — likely supports batch via comma-separated symbols
- `profile` — may support batch
- `ratios-ttm`, `key-metrics-ttm` — likely single-symbol only

**Acceptance criteria:**
- Reduced total API call count where batch endpoints exist
- Fallback to individual calls if batch isn't supported
- Total call count logged for comparison

---

### TASK 11: Add earnings calendar awareness

**Files:** `src/data.py`, `src/strategy.py`, `config.py`
**Depends on:** Nothing
**Risk:** Low (additive feature)

**Purpose:** Avoid buying stocks right before earnings (binary event risk) and prefer buying right after strong earnings beats.

**What to add:**

In `config.py`:
```python
EARNINGS_BLACKOUT_DAYS: int = 5  # Don't buy within 5 days before earnings
EARNINGS_BOOST_DAYS: int = 10    # Boost score if strong beat within 10 days
```

In `data.py`:
- Add `get_earnings_calendar(symbol)` — use FMP `/stable/earning-calendar` endpoint
- Add `next_earnings_date: str | None = None` to `StockData`
- Add `days_since_last_earnings: int | None = None` to `StockData`

In `strategy.py`:
- In scoring: if last earnings was within `EARNINGS_BOOST_DAYS` and EPS beat → add 5 bonus points
- In `get_buy_recommendations()`: filter out stocks with next earnings within `EARNINGS_BLACKOUT_DAYS`

**Acceptance criteria:**
- No buys within 5 days of earnings
- Recent strong earnings get a scoring boost
- New fields backward-compatible with cache (`None` defaults)

---

### TASK 12: Add a circuit breaker / daily loss limit

**Files:** `main.py`, `config.py`
**Depends on:** Nothing
**Risk:** Low (safety feature)

**Purpose:** Halt execution if the portfolio is already down significantly today. Prevents selling into panic and buying into a crash.

**What to add:**

In `config.py`:
```python
CIRCUIT_BREAKER_PCT: float = -0.05  # Halt if portfolio down > 5% today
MARKET_CIRCUIT_BREAKER_PCT: float = -0.03  # Halt if SPY down > 3% today
```

In `main.py` `cmd_execute()`, before Step 0:
1. Get account status
2. Compute today's change: `(portfolio_value - last_equity) / last_equity`
3. If below `CIRCUIT_BREAKER_PCT`, log a warning, send Discord alert, and exit early (return 0, not error)
4. Optionally check SPY's daily return via `fmp.get_quote("SPY")` — if market is crashing, don't trade

**Acceptance criteria:**
- Bot stops trading on crash days
- Discord notification sent explaining why
- Does NOT affect `--report`, `--status`, or `--screen` commands
- Non-destructive exit (return 0)

---

### TASK 13: Improve GitHub Actions scheduling

**Files:** `.github/workflows/trading.yml`
**Depends on:** Nothing
**Risk:** Low (CI/CD only)

**Current problems:**
1. Manual DST adjustment needed in March/November
2. No midday rebalance option
3. No holiday awareness (runs on market holidays)

**What to change:**

1. **Auto-DST:** Add a step that checks if US is in EDT or EST and adjusts timing:
   ```yaml
   - name: Check if market is open
     run: |
       # Use Python to check NYSE calendar
       pip install exchange_calendars
       python -c "
       import exchange_calendars as ec
       import datetime
       nyse = ec.get_calendar('XNYS')
       today = datetime.date.today()
       if not nyse.is_session(today):
           print('Market closed today')
           exit(1)
       "
   ```

2. **Add midday check** (optional second execution for rebalancing):
   ```yaml
   - cron: '00 18 * * 1-5'  # ~1 PM ET (check for midday signals)
   ```

3. **Add `exchange_calendars`** to requirements.txt for holiday detection

**Acceptance criteria:**
- Bot does not run on market holidays
- Correct schedule for both EST and EDT
- Optional midday execution slot

---

### TASK 14: Optimize the optimizer — add momentum tilt

**Files:** `src/optimizer.py`
**Depends on:** Task 4 (momentum scores)
**Risk:** Medium (changes allocation behavior)

**Current:** Pure Sharpe ratio maximization. This is backward-looking and tends to over-weight low-volatility stocks.

**What to change:**
1. Add optional `momentum_scores: dict[str, float]` parameter to `optimize_allocations()`
2. After Sharpe optimization, apply a momentum tilt:
   - For each stock, multiply its allocation by `(1 + tilt_factor * normalized_momentum_score)`
   - Re-normalize to sum to 1.0
3. Add `MOMENTUM_TILT_FACTOR: float = 0.20` to config (20% tilt toward momentum)

**Alternative approach (simpler):**
- Combine Sharpe optimization with a momentum objective: minimize `-(sharpe * (1 - w) + momentum * w)` where w = momentum weight
- This is a single-pass optimization that naturally blends both factors

**Acceptance criteria:**
- High-momentum stocks get larger allocations than pure Sharpe would give
- Tilt factor configurable
- Equal-weight fallback still works

---

### TASK 15: Clean up CLAUDE.md for token efficiency

**Files:** `C:\Users\victo\.claude\CLAUDE.md`
**Depends on:** Nothing
**Risk:** None (documentation only)

**Current problem:** CLAUDE.md contains Java/Spring Boot and React/TypeScript sections that are irrelevant to this Python project. These consume ~60 lines of tokens on every Claude interaction.

**What to change:**
- Remove the Java / Spring Boot section entirely
- Remove the React / TypeScript section entirely
- Keep Communication Protocol, Code Quality, and Process sections
- Add Python-specific guidance:
  ```markdown
  ## Python
  - Use type hints everywhere (Python 3.11+)
  - Use dataclasses for structured data
  - Use pathlib over os.path
  - Use logging module, not print() for debug info
  - Follow existing patterns in the codebase
  ```

**Expected token savings:** ~40-50% reduction in CLAUDE.md prompt tokens.

**Acceptance criteria:**
- Only Python-relevant guidance remains
- Communication Protocol and Process sections preserved
- File is under 30 lines

---

### TASK 16: Clean up settings.local.json permissions bloat

**Files:** `D:\Projects\personal\longinvest\.claude\settings.local.json`
**Depends on:** Nothing
**Risk:** None (tooling config only)

**Current problem:** The permissions array has 39 entries, many of which are one-off commit messages and specific git commands that will never be reused. This adds noise.

**What to change:**
- Remove all one-off `git commit -m "..."` entries (lines 11, 16)
- Remove specific `git show --stat` entries (line 20)
- Remove specific `git -C ... status/diff/log` entries (lines 36-39) — already covered by wildcard patterns
- Keep wildcard patterns: `Bash(git commit:*)`, `Bash(git add:*)`, etc.
- Keep `WebFetch` domain allowlists (useful)
- Deduplicate and alphabetize

**Acceptance criteria:**
- Permissions array reduced to ~15-20 entries
- All needed operations still permitted
- No functional regressions

---

### TASK 17: Add position rebalancing logic

**Files:** `main.py`, `src/broker.py`
**Depends on:** Nothing
**Risk:** Medium (triggers sells and buys)

**Current problem:** `cmd_execute()` only buys new positions. It never rebalances existing positions. If a position has grown from 10% to 30% of portfolio, it stays there.

**What to add:**

New step in `cmd_execute()` between Step 0 (sell review) and Step 1 (screening):
1. Get current portfolio allocations (position value / portfolio value)
2. Compare against target allocations (from last optimization, or re-run optimizer on held positions)
3. If any position deviates from target by > `REBALANCE_THRESHOLD` (e.g., 5%), sell the excess
4. Use freed cash for new buys in Step 4

In `config.py`:
```python
REBALANCE_THRESHOLD: float = 0.05  # Rebalance if position drifts > 5% from target
ENABLE_REBALANCING: bool = True
```

**Acceptance criteria:**
- Overweight positions are trimmed
- Freed cash is available for new buys
- Rebalancing can be disabled via config
- All sells tracked by TradeTracker

---

### TASK 18: Add webhook alerts for trade signals (not just execution)

**Files:** `src/notifier.py`, `main.py`
**Depends on:** Nothing
**Risk:** Low (additive notifications)

**What to add:**
1. `format_screening_embed(recommendations)` — send top 5 picks to Discord after screening
2. `format_circuit_breaker_embed(reason, portfolio_change)` — alert when circuit breaker fires
3. `format_rebalance_embed(trimmed, added)` — notify rebalance actions
4. Add notification hooks in `cmd_execute()` at each major step

**Acceptance criteria:**
- Discord receives screening results, trade signals, and alerts
- Notifications respect `ENABLE_NOTIFICATIONS` flag
- Each message type has distinct embed color (blue for screening, orange for rebalance, red for circuit breaker)

---

## Dependency Graph

```
Independent (can start immediately):
  Task 1:  Parallelize FMP API calls
  Task 2:  Loosen guardrails
  Task 3:  Reweight scoring
  Task 5:  Tiered trailing stops
  Task 6:  Intraday momentum check
  Task 7:  Reduce SMA-200 to SMA-50
  Task 8:  Increase position limits (config only)
  Task 11: Earnings calendar
  Task 12: Circuit breaker
  Task 13: GitHub Actions scheduling
  Task 15: Clean up CLAUDE.md
  Task 16: Clean up settings.local.json
  Task 17: Rebalancing logic
  Task 18: Webhook alerts

Depends on Task 1:
  Task 10: Batch FMP endpoints

Depends on Task 4:
  Task 9:  Relative strength scoring
  Task 14: Momentum tilt in optimizer

Standalone:
  Task 4:  Momentum & volume in technicals
```

## Priority Order (Suggested)

**Phase 1 — Quick wins (config changes + cleanup):**
1. Task 8: Position limits (config only, 5 min)
2. Task 15: Clean CLAUDE.md (documentation, 10 min)
3. Task 16: Clean settings.local.json (config, 10 min)
4. Task 7: SMA-50 + config extraction (small code change)

**Phase 2 — Core strategy shift:**
5. Task 2: Loosen guardrails
6. Task 3: Reweight scoring
7. Task 1: Parallelize FMP calls
8. Task 5: Tiered trailing stops

**Phase 3 — Execution intelligence:**
9. Task 6: Intraday check
10. Task 12: Circuit breaker
11. Task 17: Rebalancing
12. Task 11: Earnings calendar

**Phase 4 — Advanced momentum:**
13. Task 4: Momentum/volume technicals
14. Task 9: Relative strength
15. Task 14: Optimizer momentum tilt

**Phase 5 — Polish:**
16. Task 10: Batch FMP endpoints
17. Task 13: GitHub Actions DST/holidays
18. Task 18: Expanded Discord alerts

---

## Automated Daily Run Compatibility

All tasks maintain backward compatibility with the existing CLI interface:
- `python main.py` (execute) — still works, gains new features
- `python main.py --dry-run` — still works
- `python main.py --report` — still works
- `python main.py --screen` — still works
- `python main.py --status` — still works

The GitHub Actions workflow (`trading.yml`) continues to call the same commands. Task 13 improves the workflow itself but doesn't break the existing schedule.

**Critical invariant:** No task changes the CLI argument interface. All new behavior is controlled via `config.py` parameters with sensible defaults that match current behavior until explicitly changed.

---

## FMP Subscription Optimization (300 calls/min)

| Current | After Task 1 | After Task 10 |
|---------|-------------|---------------|
| 5 calls/sec (sequential, 200ms delay) | 50+ calls/sec (concurrent, rate-limited) | Fewer total calls via batch endpoints |
| ~4 min for 200 stocks | ~30 sec for 200 stocks | ~15 sec for 200 stocks |
| 1,201 calls per full run | 1,201 calls (same count, faster) | ~400-600 calls (batch reduces count) |

The subscription tier is currently **massively underutilized**. Task 1 alone will cut screening time by 5-10x.
