import logging
import sys
from pathlib import Path

# Setup central logging format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Define log file path inside the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),          # Prints clean logs to console/terminal
        logging.FileHandler(LOG_FILE, encoding="utf-8") # Permanently saves logs to file
    ]
)

# Create a reusable logger instance
logger = logging.getLogger("energy_pipeline")