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

    # Grid coordinates — PSCO service territory (Denver, CO)
    GRID_LAT: float = 39.7
    GRID_LON: float = -104.9

    # MLflow
    MLFLOW_EXPERIMENT_NAME: str = "Energy_Consumption_Forecasting"
    # Explicit URI keeps all scripts and notebooks pointing at the same store.
    # Defaults to a SQLite file at the project root so the location is
    # always absolute and CWD-independent (notebooks/ won't create a stray DB).
    MLFLOW_TRACKING_URI: str = ""

    @property
    def mlflow_uri(self) -> str:
        return self.MLFLOW_TRACKING_URI or f"sqlite:///{self.BASE_DIR / 'mlruns.db'}"

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
