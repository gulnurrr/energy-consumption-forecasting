# CLAUDE.md — Energy Demand Forecasting Project

## What this project is

End-to-end ML pipeline that forecasts **daily electricity demand (MWh)** for the PSCO grid region (Colorado, USA).

Data sources → EIA Open Data API (demand), Open-Meteo API (temperature).  
Model → XGBoost with time-series cross-validation.  
Serving → FastAPI REST endpoint (`/predict`, `/health`).  
Tracking → MLflow experiment logging + Model Registry.

---

## Quick start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install all dependencies
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env
# Edit .env and set EIA_API_KEY=<your key from https://www.eia.gov/opendata/>
```

---

## Project layout

```
src/               Business logic — import this, test this, never couple to framework
  config.py        Single source of truth: paths, env vars, grid coordinates
  logger.py        Structured logger factory (call get_logger(__name__))
  ingest.py        EIA API → raw DataFrame
  preprocess.py    Raw DataFrame → clean DatetimeIndex DataFrame
  data_quality.py  Validation gate before training (returns bool, [errors])
  features.py      Calendar / lag / rolling / weather feature engineering
  tune.py          Optuna hyperparameter search — each trial logged to MLflow
  train.py         TimeSeriesSplit CV → final model → MLflow Model Registry

app/
  main.py          FastAPI service: POST /predict, GET /health

scripts/
  run_pipeline.py  Orchestrates ingest → preprocess → features → train

tests/
  conftest.py      Shared fixtures + EIA_API_KEY env shim for CI
  test_*.py        One file per src/ module

.env               Real secrets (gitignored — never commit)
.env.example       Template listing required variables
pyproject.toml     Package metadata + all dependency groups
requirements.txt   Minimal runtime deps for Docker / deployment
```

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EIA_API_KEY` | **yes** | — | EIA Open Data API key |
| `WEATHER_API_URL` | no | Open-Meteo archive URL | Override weather endpoint |
| `GRID_LAT` | no | `39.7` | Latitude for weather fetch |
| `GRID_LON` | no | `-104.9` | Longitude for weather fetch |
| `MLFLOW_EXPERIMENT_NAME` | no | `Energy_Consumption_Forecasting` | MLflow experiment |

`config.py` raises a `RuntimeError` with the list of missing variables at import time — you will never see a silent failure.

---

## Running things

### Full training pipeline
```bash
python scripts/run_pipeline.py
```
Runs: ingest → data quality → preprocess → weather features → time-series features → hyperparameter tuning (optional) → train → MLflow logging.

### API server
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

Requires `artifacts/model.pkl` and `artifacts/feature_columns.json` — run the pipeline first.

### MLflow UI
```bash
mlflow ui
```
Opens at `http://localhost:5000`. Shows all experiment runs, nested Optuna trials, and registered model versions.

### Tests
```bash
pytest tests/ -v                    # all tests
pytest tests/test_api.py -v         # API tests only
pytest tests/ --cov=src --cov-report=term-missing   # with coverage
```

---

## Key design decisions

### `config.py` — no module-level side effects
`Settings()` is instantiated at module level (so `from src.config import config` works everywhere), but `create_dirs()` is **not** called at import time. Entry points (`run_pipeline.py`, `app/main.py`) call it explicitly. This keeps tests fast and import-safe.

### `data_quality()` returns `(bool, list[str])`
Never raises — callers decide what to do with failures. The pipeline raises `ValueError`; a future monitoring job might just log a warning. Returning the error list makes failure reasons introspectable.

### `TimeSeriesSplit` for cross-validation
Standard k-fold would leak future data into training folds. `TimeSeriesSplit` respects temporal ordering: each validation fold is always strictly after its training fold.

### `train.py` drops NaN before splitting
Lag and rolling features leave ~30 NaN rows at the start of the dataset. These are dropped in `train_model()` before `X`/`y` split to avoid passing NaN to XGBoost silently.

### MLflow nested runs in `tune.py`
Each Optuna trial → one child `mlflow.start_run(nested=True)` under the parent `"optuna_tuning"` run. This makes the full search history queryable in the MLflow UI without polluting the top-level experiment view.

### `app/main.py` — `datetime.date` not `str` for the date field
Pydantic validates the type at deserialization. Invalid dates return `422` (client error) before any Python date parsing happens. If `str` were used, `pd.Timestamp("not-a-date")` would produce a `500` (server error).

### Mocking strategy in tests
- **MLflow** is never mocked in `test_tune.py` — it writes to a per-test SQLite DB (`tmp_path/mlflow.db`) so logged params/metrics/tags are verified by querying the actual store.
- **XGBRegressor** is mocked in `test_tune.py` — real training inside Optuna trials would make tests unacceptably slow.
- **HTTP calls** (`requests.get`) are always mocked — tests must not make network requests.
- **FastAPI lifespan** is exercised for real in `test_api.py` — only `joblib.load` and `ARTIFACT_DIR` are patched, so startup/shutdown behavior is tested.

---

## Adding a new feature

1. Write the function in the relevant `src/` module.
2. Write a test in the corresponding `tests/test_*.py` file.
3. If it needs a new env variable, add it to `Settings` in `config.py` and document it in `.env.example`.
4. Run `pytest tests/ -v` — all 140+ tests must stay green before committing.
