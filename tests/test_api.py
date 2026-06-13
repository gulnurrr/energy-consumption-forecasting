import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as api_module
from app.main import app
from src.config import config

# Feature columns that exactly mirror what _build_feature_row produces
_FEATURE_COLS = [
    "day_of_week", "month", "is_weekend", "day_of_year", "is_holiday",
    "lag_1", "lag_7", "lag_14", "lag_21", "lag_30",
    "rolling_mean_7d", "rolling_std_7d", "rolling_mean_14d", "temperature",
]

# A valid request payload used across multiple tests
_VALID_PAYLOAD = {
    "date": "2024-03-15",
    "temperature": 12.5,
    "lag_1": 280_000.0,
    "lag_7": 275_000.0,
    "lag_14": 278_000.0,
    "lag_21": 272_000.0,
    "lag_30": 268_000.0,
    "rolling_mean_7d": 276_000.0,
    "rolling_std_7d": 8_000.0,
    "rolling_mean_14d": 274_000.0,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.predict.return_value = [285_432.5]
    return model


@pytest.fixture
def client(tmp_path, mock_model):
    """
    TestClient with lifespan running, but model loading mocked.
    tmp_path holds real artifact files so Path.exists() passes;
    joblib.load is patched so we never need a real pickle.
    """
    model_path = tmp_path / "model.pkl"
    features_path = tmp_path / "feature_columns.json"

    model_path.write_bytes(b"placeholder")
    features_path.write_text(json.dumps(_FEATURE_COLS))

    with patch.object(config, "ARTIFACT_DIR", tmp_path), \
         patch("app.main.joblib.load", return_value=mock_model):
        with TestClient(app) as c:
            yield c


# ── Lifespan / startup ────────────────────────────────────────────────────────


class TestStartup:
    def test_raises_when_model_pkl_missing(self, tmp_path):
        """Lifespan must raise RuntimeError — not silently proceed — if pkl absent."""
        with patch.object(config, "ARTIFACT_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="Model artifact not found"):
                with TestClient(app):
                    pass

    def test_raises_when_feature_columns_json_missing(self, tmp_path, mock_model):
        """pkl exists but feature_columns.json is absent → RuntimeError."""
        (tmp_path / "model.pkl").write_bytes(b"placeholder")
        with patch.object(config, "ARTIFACT_DIR", tmp_path), \
             patch("app.main.joblib.load", return_value=mock_model):
            with pytest.raises(RuntimeError, match="Feature schema not found"):
                with TestClient(app):
                    pass

    def test_model_loaded_after_successful_startup(self, client):
        """After a clean startup, _model must not be None."""
        assert api_module._model is not None

    def test_feature_columns_loaded_after_successful_startup(self, client):
        assert api_module._feature_columns == _FEATURE_COLS


# ── GET /health ───────────────────────────────────────────────────────────────


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_ok_when_model_loaded(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_feature_count_correct(self, client):
        assert client.get("/health").json()["feature_count"] == len(_FEATURE_COLS)

    def test_api_version_present(self, client):
        assert client.get("/health").json()["api_version"] == app.version

    def test_status_degraded_when_model_none(self, client):
        with patch.object(api_module, "_model", None):
            resp = client.get("/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["model_loaded"] is False


# ── POST /predict — happy path ────────────────────────────────────────────────


class TestPredictHappyPath:
    def test_returns_200(self, client):
        assert client.post("/predict", json=_VALID_PAYLOAD).status_code == 200

    def test_response_contains_expected_fields(self, client):
        body = client.post("/predict", json=_VALID_PAYLOAD).json()
        assert {"date", "predicted_demand_mwh", "unit"} <= body.keys()

    def test_date_echoed_in_response(self, client):
        body = client.post("/predict", json=_VALID_PAYLOAD).json()
        assert body["date"] == _VALID_PAYLOAD["date"]

    def test_unit_is_mwh(self, client):
        body = client.post("/predict", json=_VALID_PAYLOAD).json()
        assert body["unit"] == "MWh"

    def test_prediction_rounded_to_two_decimals(self, client, mock_model):
        mock_model.predict.return_value = [285_432.123456]
        body = client.post("/predict", json=_VALID_PAYLOAD).json()
        # Value must have at most 2 decimal places
        assert body["predicted_demand_mwh"] == round(285_432.123456, 2)

    def test_model_predict_called_once_per_request(self, client, mock_model):
        client.post("/predict", json=_VALID_PAYLOAD)
        assert mock_model.predict.call_count == 1

    def test_weekend_flag_derived_from_date(self, client, mock_model):
        """2024-03-16 is Saturday — is_weekend must be 1 in the feature row."""
        payload = {**_VALID_PAYLOAD, "date": "2024-03-16"}
        client.post("/predict", json=payload)
        feature_df = mock_model.predict.call_args[0][0]
        assert feature_df["is_weekend"].iloc[0] == 1

    def test_weekday_flag_derived_from_date(self, client, mock_model):
        """2024-03-15 is Friday — is_weekend must be 0."""
        client.post("/predict", json=_VALID_PAYLOAD)
        feature_df = mock_model.predict.call_args[0][0]
        assert feature_df["is_weekend"].iloc[0] == 0

    def test_future_date_accepted(self, client):
        payload = {**_VALID_PAYLOAD, "date": "2030-06-01"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_past_date_accepted(self, client):
        payload = {**_VALID_PAYLOAD, "date": "2018-01-01"}
        assert client.post("/predict", json=payload).status_code == 200


# ── POST /predict — error cases ───────────────────────────────────────────────


class TestPredictErrors:
    def test_503_when_model_not_loaded(self, client):
        with patch.object(api_module, "_model", None):
            resp = client.post("/predict", json=_VALID_PAYLOAD)
        assert resp.status_code == 503

    def test_503_detail_message_present(self, client):
        with patch.object(api_module, "_model", None):
            resp = client.post("/predict", json=_VALID_PAYLOAD)
        assert "not loaded" in resp.json()["detail"].lower()

    def test_422_for_invalid_date_string(self, client):
        """'not-a-date' must return 422, not 500 — Pydantic validates date type."""
        payload = {**_VALID_PAYLOAD, "date": "not-a-date"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_for_nonsense_date(self, client):
        payload = {**_VALID_PAYLOAD, "date": "9999-99-99"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_for_missing_required_field(self, client):
        payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "lag_1"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_for_wrong_type_on_numeric_field(self, client):
        payload = {**_VALID_PAYLOAD, "temperature": "warm"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_422_for_empty_body(self, client):
        assert client.post("/predict", json={}).status_code == 422
