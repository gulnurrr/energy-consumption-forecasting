import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Project root directory (one level up from src/)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Data and Model Directory Paths
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "models"
    
    # EIA API Settings
    EIA_API_KEY = os.getenv("EIA_API_KEY")
    EIA_API_URL = "http://api.eia.gov/v2/electricity/retail-sales/data/"
    
    # MLflow Settings
    MLFLOW_EXPERIMENT_NAME = "Energy_Consumption_Forecasting"
    
    @classmethod
    def create_dirs(cls):
        """
        Creates the necessary project directories automatically if they do not exist.
        """
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate the configuration object and initialize directories
config = Config()
config.create_dirs()