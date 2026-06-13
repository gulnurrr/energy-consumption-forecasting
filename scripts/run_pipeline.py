import argparse
import sys
import pandas as pd

from src.config import config
from src.logger import get_logger
from src.ingest import fetch_daily_grid_data
from src.preprocess import preprocess_data
from src.features import create_features, add_weather_features
from src.train import train_model
from src.tune import tune_model
from src.data_quality import data_quality

logger = get_logger(__name__)

_RAW_CACHE = config.RAW_DATA_DIR / "demand_raw.parquet"
_PROCESSED_CACHE = config.PROCESSED_DATA_DIR / "features.parquet"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Energy demand forecasting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search before training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        metavar="N",
        help="Number of Optuna trials (only used with --tune)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-fetch from EIA API and recompute features even if caches exist",
    )
    return parser.parse_args()


def _load_or_fetch_raw(no_cache: bool) -> pd.DataFrame:
    if not no_cache and _RAW_CACHE.exists():
        logger.info(f"Loading raw data from cache | path={_RAW_CACHE}")
        return pd.read_parquet(_RAW_CACHE)

    logger.info("Fetching raw data from EIA API...")
    df = fetch_daily_grid_data()
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned from ingestion")

    logger.info(f"Saving raw data cache | path={_RAW_CACHE}")
    df.to_parquet(_RAW_CACHE, index=False)
    return df


def _build_features(no_cache: bool) -> pd.DataFrame:
    """Return fully-featured DataFrame, using the processed cache when available."""
    if not no_cache and _PROCESSED_CACHE.exists():
        logger.info(f"Loading processed features from cache | path={_PROCESSED_CACHE}")
        return pd.read_parquet(_PROCESSED_CACHE)

    df = _load_or_fetch_raw(no_cache)

    logger.info("Preprocessing data...")
    df = preprocess_data(df)

    logger.info("Running data quality checks...")
    is_valid, errors = data_quality(df)
    if not is_valid:
        raise ValueError(f"Data quality checks failed: {errors}")

    logger.info("Adding weather features...")
    df = add_weather_features(df, lat=config.GRID_LAT, lon=config.GRID_LON)

    logger.info("Building time-series features...")
    df = create_features(df)

    logger.info(f"Saving processed features | path={_PROCESSED_CACHE}")
    df.to_parquet(_PROCESSED_CACHE, index=True)

    return df


def main():
    args = _parse_args()
    config.create_dirs()

    try:
        logger.info("PIPELINE STARTED")

        # ── 1–4. INGEST → PREPROCESS → QUALITY → FEATURES ────────────────────
        df = _build_features(no_cache=args.no_cache)
        logger.info(f"Feature matrix ready | shape={df.shape} | columns={df.columns.tolist()}")

        # ── 5. HYPERPARAMETER TUNING (optional) ───────────────────────────────
        best_params = None
        if args.tune:
            logger.info(f"Running hyperparameter tuning | n_trials={args.n_trials}")
            df_clean = df.dropna()
            X_tune = df_clean.drop(columns=["demand_mwh"])
            y_tune = df_clean["demand_mwh"]
            best_params = tune_model(X_tune, y_tune, n_trials=args.n_trials)
            logger.info(f"Tuning complete | best_params={best_params}")

        # ── 6. TRAINING ───────────────────────────────────────────────────────
        logger.info("Training model...")
        model = train_model(df, best_params=best_params)

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        return model

    except Exception:
        logger.exception("PIPELINE FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
