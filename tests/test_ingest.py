import pandas as pd
import pytest
import requests
from unittest.mock import MagicMock, patch

from src.ingest import fetch_daily_grid_data


# ── Helpers ──────────────────────────────────────────────────────────────────

_EIA_RECORDS = [
    {"period": "2023-01-01", "value": 280_000.0, "respondent": "PSCO"},
    {"period": "2023-01-02", "value": 295_000.0, "respondent": "PSCO"},
]


def _ok_mock(records=None) -> MagicMock:
    """Mock a successful requests.Response with valid EIA payload."""
    if records is None:
        records = _EIA_RECORDS
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"response": {"data": records}}
    return mock


# ── Happy path ────────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_dataframe(self):
        with patch("src.ingest.requests.get", return_value=_ok_mock()):
            result = fetch_daily_grid_data()
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_api_records(self):
        with patch("src.ingest.requests.get", return_value=_ok_mock()):
            result = fetch_daily_grid_data()
        assert len(result) == len(_EIA_RECORDS)

    def test_period_and_value_columns_present(self):
        with patch("src.ingest.requests.get", return_value=_ok_mock()):
            result = fetch_daily_grid_data()
        assert "period" in result.columns
        assert "value" in result.columns

    def test_empty_record_list_returns_empty_dataframe(self):
        with patch("src.ingest.requests.get", return_value=_ok_mock(records=[])):
            result = fetch_daily_grid_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ── HTTP errors ───────────────────────────────────────────────────────────────


class TestHTTPErrors:
    def test_404_raises_request_exception(self):
        mock = MagicMock()
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        with patch("src.ingest.requests.get", return_value=mock):
            with pytest.raises(requests.exceptions.RequestException):
                fetch_daily_grid_data()

    def test_500_raises_request_exception(self):
        mock = MagicMock()
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        with patch("src.ingest.requests.get", return_value=mock):
            with pytest.raises(requests.exceptions.RequestException):
                fetch_daily_grid_data()

    def test_connection_error_propagated(self):
        with patch(
            "src.ingest.requests.get",
            side_effect=requests.exceptions.ConnectionError("unreachable"),
        ):
            with pytest.raises(requests.exceptions.RequestException):
                fetch_daily_grid_data()

    def test_timeout_propagated(self):
        with patch(
            "src.ingest.requests.get",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            with pytest.raises(requests.exceptions.RequestException):
                fetch_daily_grid_data()


# ── Malformed API responses ───────────────────────────────────────────────────


class TestMalformedResponse:
    def test_missing_response_key_raises_key_error(self):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"unexpected": {}}
        with patch("src.ingest.requests.get", return_value=mock):
            with pytest.raises(KeyError):
                fetch_daily_grid_data()

    def test_missing_data_key_raises_key_error(self):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"response": {"no_data_here": []}}
        with patch("src.ingest.requests.get", return_value=mock):
            with pytest.raises(KeyError):
                fetch_daily_grid_data()

    def test_completely_empty_json_raises_key_error(self):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {}
        with patch("src.ingest.requests.get", return_value=mock):
            with pytest.raises(KeyError):
                fetch_daily_grid_data()


# ── Request parameters ────────────────────────────────────────────────────────


class TestRequestParameters:
    def _capture(self, records=None):
        """Return (captured_params dict, mock_get) via call_args inspection."""
        mock_get = MagicMock(return_value=_ok_mock(records))
        return mock_get

    def test_default_rto_code_is_psco(self):
        mock_get = self._capture()
        with patch("src.ingest.requests.get", mock_get):
            fetch_daily_grid_data()
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["facets[respondent][]"] == "PSCO"

    def test_custom_rto_code_passed_to_api(self):
        mock_get = self._capture()
        with patch("src.ingest.requests.get", mock_get):
            fetch_daily_grid_data(rto_code="ERCO")
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["facets[respondent][]"] == "ERCO"

    def test_start_date_passed_to_api(self):
        mock_get = self._capture()
        with patch("src.ingest.requests.get", mock_get):
            fetch_daily_grid_data(start_date="2020-06-01")
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["start"] == "2020-06-01"

    def test_end_date_passed_when_provided(self):
        mock_get = self._capture()
        with patch("src.ingest.requests.get", mock_get):
            fetch_daily_grid_data(end_date="2023-12-31")
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["end"] == "2023-12-31"

    def test_end_date_is_none_by_default(self):
        mock_get = self._capture()
        with patch("src.ingest.requests.get", mock_get):
            fetch_daily_grid_data()
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["end"] is None

    def test_timeout_is_set(self):
        """Verify a network timeout is always configured — no infinite hangs."""
        mock_get = self._capture()
        with patch("src.ingest.requests.get", mock_get):
            fetch_daily_grid_data()
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] is not None
        assert kwargs["timeout"] > 0
