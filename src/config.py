from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths — derived from package location, not CWD
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    MODEL_DIR: Path = BASE_DIR / "models"
    LOG_DIR: Path = BASE_DIR / "logs"
    ARTIFACT_DIR: Path = BASE_DIR / "artifacts"
    APP_DIR: Path = BASE_DIR / "app"

    # Required — must be present in .env
    EIA_API_KEY: str
    WEATHER_API_URL: str = "https://archive-api.open-meteo.com/v1/archive"

    # MLflow
    MLFLOW_EXPERIMENT_NAME: str = "Energy_Consumption_Forecasting"

    def create_dirs(self) -> None:
        """Create all required project directories.

        Call this explicitly at pipeline/app entry points, never at import time.
        """
        for path in [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_DIR,
            self.LOG_DIR,
            self.ARTIFACT_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


try:
    config = Settings()
except ValidationError as exc:
    missing = [str(err["loc"][0]) for err in exc.errors() if err["type"] == "missing"]
    raise RuntimeError(
        f"Missing required environment variables: {missing}. "
        "Copy .env.example to .env and set the missing values."
    ) from exc
