"""Tests for src/finscan.py — API client, caching, serialization."""

from unittest.mock import MagicMock, patch

import pytest

from src.cache import CacheManager
from src.finscan import FinScanClient, FinScanResult
from tests.conftest import make_finscan_result


@pytest.fixture
def client():
    return FinScanClient(api_key="test-key", base_url="https://finscan.io")


def _mock_response(status_code=200, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    return resp


SAMPLE_BODY = {
    "ticker": "AAPL",
    "scanResult": {
        "compositeRisk": {"score": 25, "rating": "LOW"},
        "piotroski": {"score": 7, "signal": "STRONG"},
        "beneish": {"signal": "UNLIKELY_MANIPULATOR"},
        "altman": {"zone": "SAFE"},
        "redFlags": [
            {"severity": "LOW", "category": "debt", "message": "Minor concern"}
        ],
    },
}


class TestScan:
    @patch("src.finscan.requests.get")
    def test_scan_success(self, mock_get, client):
        mock_get.return_value = _mock_response(200, SAMPLE_BODY)
        result = client.scan("AAPL")
        assert result is not None
        assert result.ticker == "AAPL"
        assert result.composite_score == 25
        assert result.risk_rating == "LOW"
        assert result.piotroski_score == 7
        assert result.beneish_signal == "UNLIKELY_MANIPULATOR"
        assert len(result.red_flags) == 1

    @patch("src.finscan.requests.get")
    def test_scan_401(self, mock_get, client):
        mock_get.return_value = _mock_response(401)
        result = client.scan("AAPL")
        assert result is None

    @patch("src.finscan.requests.get")
    def test_scan_429(self, mock_get, client):
        mock_get.return_value = _mock_response(429, headers={"Retry-After": "60"})
        result = client.scan("AAPL")
        assert result is None

    @patch("src.finscan.requests.get")
    def test_scan_500(self, mock_get, client):
        mock_get.return_value = _mock_response(500)
        result = client.scan("AAPL")
        assert result is None

    @patch("src.finscan.requests.get")
    def test_scan_network_error(self, mock_get, client):
        import requests
        mock_get.side_effect = requests.RequestException("Connection failed")
        result = client.scan("AAPL")
        assert result is None


class TestParseResponse:
    def test_parse_all_fields(self, client):
        result = client._parse_response(SAMPLE_BODY)
        assert result.ticker == "AAPL"
        assert result.composite_score == 25
        assert result.piotroski_score == 7
        assert result.altman_zone == "SAFE"

    def test_parse_missing_optional_fields(self, client):
        body = {
            "ticker": "XYZ",
            "scanResult": {
                "compositeRisk": {"score": 40, "rating": "MODERATE"},
            },
        }
        result = client._parse_response(body)
        assert result.ticker == "XYZ"
        assert result.piotroski_score is None
        assert result.beneish_signal is None
        assert result.altman_zone is None
        assert result.red_flags == []


class TestCaching:
    @patch("src.finscan.requests.get")
    def test_caching_uses_real_cache(self, mock_get, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
        # Ensure the finscan category directory exists (CacheManager only creates default categories)
        (tmp_path / "cache" / "finscan").mkdir(parents=True, exist_ok=True)
        client = FinScanClient(api_key="test-key", cache=cache)
        mock_get.return_value = _mock_response(200, SAMPLE_BODY)

        # First call hits API
        result1 = client.scan("AAPL")
        assert mock_get.call_count == 1
        assert result1 is not None

        # Second call uses cache
        result2 = client.scan("AAPL")
        assert mock_get.call_count == 1  # Not called again
        assert result2.ticker == result1.ticker


class TestTickerUppercased:
    @patch("src.finscan.requests.get")
    def test_lowercase_uppercased(self, mock_get, client):
        mock_get.return_value = _mock_response(200, SAMPLE_BODY)
        client.scan("aapl")
        call_url = mock_get.call_args[0][0]
        assert "/AAPL" in call_url


class TestSerialization:
    def test_roundtrip(self):
        original = make_finscan_result(
            ticker="TEST",
            composite_score=55,
            risk_rating="ELEVATED",
            piotroski_score=5,
            red_flags=[{"severity": "HIGH", "category": "debt", "message": "Overleveraged"}],
        )
        d = FinScanClient._result_to_dict(original)
        restored = FinScanClient._dict_to_result(d)
        assert restored.ticker == original.ticker
        assert restored.composite_score == original.composite_score
        assert restored.risk_rating == original.risk_rating
        assert restored.piotroski_score == original.piotroski_score
        assert restored.red_flags == original.red_flags
