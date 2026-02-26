"""Thin HTTP client for the FinScan REST API."""

import logging
from dataclasses import dataclass, field

import requests

from src.cache import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class FinScanResult:
    ticker: str
    composite_score: int  # 0-100
    risk_rating: str  # LOW, MODERATE, ELEVATED, HIGH
    piotroski_score: int | None = None  # 0-9
    piotroski_signal: str | None = None  # STRONG, NEUTRAL, WEAK
    beneish_signal: str | None = None  # UNLIKELY_MANIPULATOR, GREY_ZONE, LIKELY_MANIPULATOR
    altman_zone: str | None = None  # SAFE, GREY, DISTRESS
    red_flags: list[dict] = field(default_factory=list)


class FinScanClient:
    """Client for the FinScan financial health scanning API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://finscan.io",
        cache: CacheManager | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache = cache

    def scan(self, ticker: str) -> FinScanResult | None:
        """Scan a ticker for financial health signals.

        Returns FinScanResult on success, None on error.
        """
        ticker = ticker.upper()

        # Check cache
        if self.cache:
            date_key = self.cache.get_date_key()
            cache_key = f"{ticker}_{date_key}"
            cached = self.cache.load("finscan", cache_key)
            if cached is not None:
                return self._dict_to_result(cached)

        # Make API request
        try:
            resp = requests.get(
                f"{self.base_url}/v1/scan/{ticker}",
                headers={"X-API-Key": self.api_key},
                timeout=15,
            )
        except requests.RequestException as e:
            logger.error(f"FinScan request failed for {ticker}: {e}")
            return None

        if resp.status_code == 401:
            logger.warning("Invalid FinScan API key")
            return None

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "unknown")
            logger.warning(f"FinScan rate limited, retry after {retry_after}s")
            return None

        if resp.status_code != 200:
            logger.error(f"FinScan error for {ticker}: HTTP {resp.status_code}")
            return None

        # Parse response
        try:
            body = resp.json()
            result = self._parse_response(body)
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse FinScan response for {ticker}: {e}")
            return None

        # Cache the result
        if self.cache:
            self.cache.save("finscan", cache_key, self._result_to_dict(result))

        return result

    def _parse_response(self, body: dict) -> FinScanResult:
        """Map API JSON response to FinScanResult."""
        scan = body["scanResult"]
        composite = scan["compositeRisk"]

        piotroski = scan.get("piotroski")
        beneish = scan.get("beneish")
        altman = scan.get("altman")
        red_flags_raw = scan.get("redFlags") or []

        return FinScanResult(
            ticker=body["ticker"],
            composite_score=composite["score"],
            risk_rating=composite["rating"],
            piotroski_score=piotroski["score"] if piotroski else None,
            piotroski_signal=piotroski["signal"] if piotroski else None,
            beneish_signal=beneish["signal"] if beneish else None,
            altman_zone=altman["zone"] if altman else None,
            red_flags=[
                {
                    "severity": flag.get("severity", "UNKNOWN"),
                    "category": flag.get("category", ""),
                    "message": flag.get("message", ""),
                }
                for flag in red_flags_raw
            ],
        )

    @staticmethod
    def _result_to_dict(result: FinScanResult) -> dict:
        """Serialize FinScanResult to a dict for caching."""
        return {
            "ticker": result.ticker,
            "composite_score": result.composite_score,
            "risk_rating": result.risk_rating,
            "piotroski_score": result.piotroski_score,
            "piotroski_signal": result.piotroski_signal,
            "beneish_signal": result.beneish_signal,
            "altman_zone": result.altman_zone,
            "red_flags": result.red_flags,
        }

    @staticmethod
    def _dict_to_result(data: dict) -> FinScanResult:
        """Deserialize a cached dict back to FinScanResult."""
        return FinScanResult(
            ticker=data["ticker"],
            composite_score=data["composite_score"],
            risk_rating=data["risk_rating"],
            piotroski_score=data.get("piotroski_score"),
            piotroski_signal=data.get("piotroski_signal"),
            beneish_signal=data.get("beneish_signal"),
            altman_zone=data.get("altman_zone"),
            red_flags=data.get("red_flags", []),
        )
