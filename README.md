# Long-Term Growth Investment Bot

Automated stock screening and trading bot focused on growth stocks using Alpaca (broker) and Financial Modeling Prep (data).

## Features

- **Growth Stock Screener**: Filters US stocks by market cap, growth metrics, and quality guardrails
- **Automated Trading**: Places fractional share orders via Alpaca
- **Paper Trading**: Configured for paper trading only (no live trading)
- **Position Management**: Limits portfolio to configurable number of positions

## Requirements

- Python 3.11+
- Alpaca paper trading account (free at https://alpaca.markets)
- Financial Modeling Prep API key (free at https://financialmodelingprep.com)

**Note:** FMP free tier has rate limits. The bot screens ~35 stocks (5 API calls each = ~175 calls). If you hit rate limits, wait a few minutes before retrying.

## Setup

1. Clone the repository and navigate to the project directory

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
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` with your credentials:
   ```
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_SECRET_KEY=your_alpaca_secret
   FMP_API_KEY=your_fmp_key
   ```

## Usage

### Check Portfolio Status
```bash
python main.py --status
```
Shows current positions, portfolio value, and cash balance.

### Run Screener
```bash
python main.py --screen
```
Runs the growth stock screener and displays top candidates without trading.

### Dry Run
```bash
python main.py --dry-run
```
Simulates a full trading cycle showing what would be bought, without executing trades.

### Execute Trades
```bash
python main.py
```
Runs the full cycle and executes buy orders.

### Debug Mode
Add `--debug` to any command for verbose logging:
```bash
python main.py --screen --debug
```

## Configuration

Edit `config.py` to adjust:

- `MAX_POSITIONS`: Maximum number of stocks to hold (default: 10)
- `MIN_MARKET_CAP`: Minimum market cap filter (default: $5B)

## Screening Criteria

**Stock Universe:**
- Curated list of ~35 mega/large-cap US stocks
- Spans Tech, Healthcare, Finance, Consumer, Industrial, Energy, and Comm sectors

**Scoring (100 points max):**
- Revenue growth: up to 30 points
- EPS growth: up to 30 points
- Return on equity: up to 20 points
- Gross margin (>30%): up to 10 points
- Operating margin (>10%): up to 10 points

**Guardrails (disqualifying):**
- Negative or P/E > 100
- Debt-to-equity > 2.0
- Current ratio < 1.0

## Project Structure

```
longinvest/
├── main.py           # CLI entry point
├── config.py         # Configuration settings
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
├── .gitignore        # Git ignore patterns
└── src/
    ├── __init__.py
    ├── data.py       # FMP API integration
    ├── broker.py     # Alpaca integration
    └── strategy.py   # Screening logic
```

## Future Enhancements (Phase 2)

- Sell logic for position management
- Sector diversification constraints
- Scheduled execution
- Performance tracking
