import logging
import time
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


class FMPClient:
    def __init__(self):
        self.api_key = Config.FMP_API_KEY
        self.base_url = Config.FMP_BASE_URL
        self.rate_limit_ms = Config.FMP_RATE_LIMIT_MS
        self._last_request_time: float = 0
        self._api_calls: int = 0

    def _rate_limit(self) -> None:
        elapsed = (time.time() - self._last_request_time) * 1000
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        self._api_calls += 1
        logger.debug(f"FMP API call #{self._api_calls}: {endpoint}")

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

    def get_stock_data(self, symbol: str) -> StockData | None:
        """Fetch and combine all relevant data for a stock."""
        try:
            profile = self.get_profile(symbol)
            if not profile:
                logger.warning(f"No profile found for {symbol}")
                return None

            quote = self.get_quote(symbol)
            ratios = self.get_ratios_ttm(symbol)
            metrics = self.get_key_metrics_ttm(symbol)
            growth = self.get_financial_growth(symbol, limit=1)

            growth_data = growth[0] if growth else {}

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

            return df[["close"]]

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
        """Fetch historical prices for multiple symbols.

        Returns:
            DataFrame with Date index, columns = symbols, values = adjusted close.
            Drops symbols with fewer than min_days of data.
        """
        price_data: dict[str, pd.Series] = {}

        for symbol in symbols:
            df = self.get_historical_prices(symbol, from_date, to_date)
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
