import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    symbol: str
    name: str
    price: float
    market_cap: float
    sector: str
    country: str
    is_etf: bool
    is_actively_trading: bool
    # Ratios
    de_ratio: float | None
    roe: float | None
    gross_margin: float | None
    # Growth
    revenue_growth: float | None
    # Cash flow
    free_cash_flow: float | None
    free_cash_flow_growth: float | None
    # Revenue
    revenue: float | None = None
    # Earnings momentum
    earnings_growth: float | None = None
    eps_beat_count: int | None = None
    earnings_growth_accelerating: bool | None = None
    revenue_growth_accelerating: bool | None = None
    next_earnings_date: str | None = None
    days_since_last_earnings: int | None = None


class FMPClient:
    def __init__(self):
        self.api_key = Config.FMP_API_KEY
        self.base_url = Config.FMP_BASE_URL
        self.max_concurrent = Config.FMP_MAX_CONCURRENT
        self._api_calls: int = 0
        self._lock = threading.Lock()
        # Token-bucket rate limiter: 300 calls/min = 5 calls/sec
        self._min_interval = 1.0 / 5.0  # 200ms between calls
        self._last_request_time: float = 0
        self._rate_lock = threading.Lock()

    def _rate_limit(self) -> None:
        with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        with self._lock:
            self._api_calls += 1
            call_num = self._api_calls
        logger.debug(f"FMP API call #{call_num}: {endpoint}")

        for attempt in range(retries):
            response = requests.get(url, params=request_params, timeout=30)
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()

        response.raise_for_status()
        return response.json()

    def get_api_call_count(self) -> int:
        return self._api_calls

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        result = self._get("quote", {"symbol": symbol})
        return result[0] if result else None

    def get_profile(self, symbol: str) -> dict[str, Any] | None:
        result = self._get("profile", {"symbol": symbol})
        return result[0] if result else None

    def get_ratios_ttm(self, symbol: str) -> dict[str, Any] | None:
        result = self._get("ratios-ttm", {"symbol": symbol})
        return result[0] if result else None

    def get_key_metrics_ttm(self, symbol: str) -> dict[str, Any] | None:
        result = self._get("key-metrics-ttm", {"symbol": symbol})
        return result[0] if result else None

    def get_financial_growth(self, symbol: str, limit: int = 1) -> list[dict[str, Any]]:
        return self._get("financial-growth", {"symbol": symbol, "limit": limit})

    def get_earnings_surprises(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch recent earnings surprises for a symbol."""
        try:
            return self._get("earnings-surprises", {"symbol": symbol})
        except Exception:
            return []

    def get_earnings_calendar(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch earnings calendar for a symbol."""
        try:
            return self._get("earning-calendar", {"symbol": symbol})
        except Exception:
            return []

    def get_stock_data(self, symbol: str) -> StockData | None:
        """Fetch and combine all relevant data for a stock.

        Fires all 6 API calls concurrently since they are independent.
        """
        try:
            with ThreadPoolExecutor(max_workers=7) as executor:
                f_profile = executor.submit(self.get_profile, symbol)
                f_quote = executor.submit(self.get_quote, symbol)
                f_ratios = executor.submit(self.get_ratios_ttm, symbol)
                f_metrics = executor.submit(self.get_key_metrics_ttm, symbol)
                f_growth = executor.submit(self.get_financial_growth, symbol, 4)
                f_surprises = executor.submit(self.get_earnings_surprises, symbol)
                f_earnings_cal = executor.submit(self.get_earnings_calendar, symbol)

            profile = f_profile.result()
            if not profile:
                logger.warning(f"No profile found for {symbol}")
                return None

            quote = f_quote.result()
            ratios = f_ratios.result()
            metrics = f_metrics.result()
            growth = f_growth.result()
            surprises = f_surprises.result()
            earnings_cal = f_earnings_cal.result()

            growth_data = growth[0] if growth else {}

            # Earnings momentum: count EPS beats in last 4 quarters
            eps_beat_count = 0
            if surprises:
                for s in surprises[:4]:
                    actual = s.get("actualEarningResult")
                    estimated = s.get("estimatedEarning")
                    if actual is not None and estimated is not None and actual > estimated:
                        eps_beat_count += 1

            # Earnings growth from financial-growth (most recent quarter)
            earnings_growth = growth_data.get("epsgrowth") or growth_data.get("netIncomeGrowth")

            # Earnings acceleration: check if growth is trending up over 4 quarters
            earnings_growth_accelerating = False
            if len(growth) >= 3:
                eg_values = []
                for g in growth[:4]:
                    val = g.get("epsgrowth") or g.get("netIncomeGrowth")
                    if val is not None:
                        eg_values.append(val)
                if len(eg_values) >= 3:
                    # Most recent > average of older quarters
                    earnings_growth_accelerating = eg_values[0] > sum(eg_values[1:]) / len(eg_values[1:])

            # Revenue acceleration: check if revenue growth is trending up over 4 quarters
            revenue_growth_accelerating = False
            if len(growth) >= 3:
                rg_values = []
                for g in growth[:4]:
                    val = g.get("revenueGrowth")
                    if val is not None:
                        rg_values.append(val)
                if len(rg_values) >= 3:
                    revenue_growth_accelerating = rg_values[0] > sum(rg_values[1:]) / len(rg_values[1:])

            # Revenue (from metrics TTM)
            revenue = metrics.get("revenuePerShareTTM") if metrics else None
            if revenue is not None and quote:
                # revenuePerShareTTM * shares outstanding approximation
                shares = quote.get("sharesOutstanding")
                if shares:
                    revenue = revenue * shares
                else:
                    revenue = None

            # Earnings calendar: find next and last earnings dates
            next_earnings_date = None
            days_since_last_earnings = None
            today = datetime.now().date()
            if earnings_cal:
                future_dates = []
                past_dates = []
                for ec in earnings_cal:
                    date_str = ec.get("date")
                    if not date_str:
                        continue
                    try:
                        ec_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        if ec_date >= today:
                            future_dates.append(ec_date)
                        else:
                            past_dates.append(ec_date)
                    except ValueError:
                        continue
                if future_dates:
                    next_earnings_date = min(future_dates).isoformat()
                if past_dates:
                    last_earnings = max(past_dates)
                    days_since_last_earnings = (today - last_earnings).days

            return StockData(
                symbol=symbol,
                name=profile.get("companyName", ""),
                price=quote.get("price", 0) if quote else profile.get("price", 0),
                market_cap=profile.get("mktCap", 0),
                sector=profile.get("sector", ""),
                country=profile.get("country", ""),
                is_etf=profile.get("isEtf", False),
                is_actively_trading=profile.get("isActivelyTrading", True),
                de_ratio=ratios.get("debtEquityRatioTTM") if ratios else None,
                roe=metrics.get("roeTTM") if metrics else None,
                gross_margin=ratios.get("grossProfitMarginTTM") if ratios else None,
                revenue_growth=growth_data.get("revenueGrowth"),
                free_cash_flow=metrics.get("freeCashFlowTTM") if metrics else None,
                free_cash_flow_growth=growth_data.get("freeCashFlowGrowth"),
                revenue=revenue,
                earnings_growth=earnings_growth,
                eps_beat_count=eps_beat_count,
                earnings_growth_accelerating=earnings_growth_accelerating,
                revenue_growth_accelerating=revenue_growth_accelerating,
                next_earnings_date=next_earnings_date,
                days_since_last_earnings=days_since_last_earnings,
            )
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_screener_stocks(
        self,
        min_market_cap: float = 300e6,
        max_market_cap: float | None = None,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        """Get US stock universe from FMP company screener.

        Returns list of dicts with symbol and marketCap for sorting.
        """
        params = {
            "country": "US",
            "isEtf": "false",
            "isActivelyTrading": "true",
            "marketCapMoreThan": int(min_market_cap),
            "limit": limit,
        }
        if max_market_cap is not None:
            params["marketCapLowerThan"] = int(max_market_cap)
        result = self._get("company-screener", params)
        return [
            {"symbol": s["symbol"], "marketCap": s.get("marketCap", 0)}
            for s in result
            if s.get("symbol")
        ]

    def get_historical_prices(
        self, symbol: str, from_date: str, to_date: str
    ) -> pd.DataFrame | None:
        """Get historical daily prices for a symbol.

        Args:
            symbol: Stock symbol
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with Date index and 'close' column, or None on error.
        """
        try:
            params = {"from": from_date, "to": to_date, "symbol": symbol}
            result = self._get("historical-price-eod/full", params)

            if not result:
                logger.warning(f"No historical data for {symbol}")
                return None

            # Handle both response formats: flat array or {"historical": [...]}
            if isinstance(result, list):
                historical = result
            elif isinstance(result, dict) and "historical" in result:
                historical = result["historical"]
            else:
                logger.warning(f"No historical data for {symbol}")
                return None

            if not historical:
                return None

            df = pd.DataFrame(historical)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            cols = ["close"]
            if "volume" in df.columns:
                cols.append("volume")
            return df[cols]

        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return None

    def get_historical_prices_batch(
        self,
        symbols: list[str],
        from_date: str,
        to_date: str,
        min_days: int = 200,
    ) -> pd.DataFrame:
        """Fetch historical prices for multiple symbols concurrently.

        Returns:
            DataFrame with Date index, columns = symbols, values = adjusted close.
            Drops symbols with fewer than min_days of data.
        """
        price_data: dict[str, pd.Series] = {}

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {
                executor.submit(self.get_historical_prices, symbol, from_date, to_date): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    df = future.result()
                except Exception as e:
                    logger.error(f"Error fetching prices for {symbol}: {e}")
                    continue
                if df is not None and len(df) >= min_days:
                    price_data[symbol] = df["close"]
                else:
                    days = len(df) if df is not None else 0
                    logger.warning(
                        f"Dropping {symbol}: only {days} days of data (need {min_days})"
                    )

        if not price_data:
            return pd.DataFrame()

        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()
        return prices_df
