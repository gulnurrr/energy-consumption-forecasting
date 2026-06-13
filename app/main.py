import json
from contextlib import asynccontextmanager

import holidays
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

US_HOLIDAYS = holidays.UnitedStates()

# ---------------------------------------------------------------------------
# Model state — populated at startup, shared across requests
# ---------------------------------------------------------------------------
_model = None
_feature_columns: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup; release on shutdown."""
    global _model, _feature_columns

    model_path = config.ARTIFACT_DIR / "model.pkl"
    features_path = config.ARTIFACT_DIR / "feature_columns.json"

    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at '{model_path}'. "
            "Run scripts/run_pipeline.py to train the model first."
        )
    if not features_path.exists():
        raise RuntimeError(
            f"Feature schema not found at '{features_path}'. "
            "Run scripts/run_pipeline.py to regenerate artifacts."
        )

    _model = joblib.load(model_path)
    logger.info(f"Model loaded | path={model_path}")

    with open(features_path) as f:
        _feature_columns = json.load(f)
    logger.info(f"Feature schema loaded | {len(_feature_columns)} features")

    yield

    _model = None
    _feature_columns = []
    logger.info("Model unloaded")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Energy Demand Forecasting API",
    description=(
        "Predicts next-day electricity demand (MWh) for the PSCO grid region "
        "using an XGBoost model trained on EIA + Open-Meteo data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)", examples=["2024-03-15"])
    temperature: float = Field(..., description="Mean daily temperature in °C", examples=[12.5])
    lag_1: float = Field(..., description="Electricity demand 1 day ago (MWh)", examples=[280000.0])
    lag_7: float = Field(..., description="Electricity demand 7 days ago (MWh)", examples=[275000.0])
    lag_14: float = Field(..., description="Electricity demand 14 days ago (MWh)", examples=[278000.0])
    lag_21: float = Field(..., description="Electricity demand 21 days ago (MWh)", examples=[272000.0])
    lag_30: float = Field(..., description="Electricity demand 30 days ago (MWh)", examples=[268000.0])
    rolling_mean_7d: float = Field(..., description="7-day rolling mean demand (MWh)", examples=[276000.0])
    rolling_std_7d: float = Field(..., description="7-day rolling std of demand (MWh)", examples=[8000.0])
    rolling_mean_14d: float = Field(..., description="14-day rolling mean demand (MWh)", examples=[274000.0])


class PredictionResponse(BaseModel):
    date: str
    predicted_demand_mwh: float
    unit: str = "MWh"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    feature_count: int
    api_version: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_feature_row(req: PredictionRequest) -> pd.DataFrame:
    """
    Derive calendar features from `date`, combine with caller-supplied
    lag/rolling/weather values, and return a single-row DataFrame whose
    column order matches the trained feature schema.
    """
    dt = pd.Timestamp(req.date)

    row: dict[str, float | int] = {
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "is_weekend": int(dt.dayofweek in (5, 6)),
        "day_of_year": dt.dayofyear,
        "is_holiday": int(dt.date() in US_HOLIDAYS),
        "lag_1": req.lag_1,
        "lag_7": req.lag_7,
        "lag_14": req.lag_14,
        "lag_21": req.lag_21,
        "lag_30": req.lag_30,
        "rolling_mean_7d": req.rolling_mean_7d,
        "rolling_std_7d": req.rolling_std_7d,
        "rolling_mean_14d": req.rolling_mean_14d,
        "temperature": req.temperature,
    }

    df = pd.DataFrame([row])

    missing = set(_feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Feature schema mismatch — columns missing from request: {missing}")

    return df[_feature_columns]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict daily electricity demand",
)
def predict(request: PredictionRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health for details.")

    try:
        features = _build_feature_row(request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    prediction = float(_model.predict(features)[0])
    logger.info(f"Prediction | date={request.date} | demand={prediction:.2f} MWh")

    return PredictionResponse(
        date=request.date,
        predicted_demand_mwh=round(prediction, 2),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
def health() -> HealthResponse:
    model_path = config.ARTIFACT_DIR / "model.pkl"
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_path=str(model_path),
        feature_count=len(_feature_columns),
        api_version=app.version,
    )
