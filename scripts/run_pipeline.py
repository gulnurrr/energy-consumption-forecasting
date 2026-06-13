import sys

from src.config import config
from src.logger import get_logger
from src.ingest import fetch_daily_grid_data
from src.preprocess import preprocess_data
from src.features import create_features, add_weather_features
from src.train import train_model
from src.data_quality import data_quality

logger = get_logger(__name__)


def main():
    config.create_dirs()

    try:
        logger.info("PIPELINE STARTED")

        # 1. INGESTION
        logger.info("Loading data...")
        df = fetch_daily_grid_data()

        if df is None or df.empty:
            raise ValueError("Empty dataframe returned from ingestion")

        logger.info(f"Raw columns: {df.columns.tolist()}")

        # 2. DATA QUALITY CHECK
        logger.info("Running data quality checks...")
        is_valid, errors = data_quality(df)
        if not is_valid:
            raise ValueError(f"Data quality checks failed: {errors}")

        # 3. PREPROCESSING
        logger.info("Preprocessing data...")
        df = preprocess_data(df)

        # 4. FEATURE ENGINEERING
        logger.info("Adding weather features...")
        df = add_weather_features(df, lat=config.GRID_LAT, lon=config.GRID_LON)

        logger.info("Building time-series features...")
        df = create_features(df)

        logger.info(f"Final feature columns: {df.columns.tolist()}")

        # 5. TRAINING
        logger.info("Training model...")
        model = train_model(df)

        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        return model

    except Exception:
        logger.exception("PIPELINE FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
