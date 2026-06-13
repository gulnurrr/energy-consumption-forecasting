import requests
import pandas as pd
from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


def fetch_daily_grid_data(
    rto_code: str = "PSCO",
    start_date: str = "2016-01-01",
    end_date: str | None = None
) -> pd.DataFrame:
    """
    Fetch daily electricity demand data from EIA API.
    If end_date is None, API returns latest available data.
    """

    api_key = config.EIA_API_KEY

    base_url = (
        "https://api.eia.gov/v2/"
        "electricity/rto/"
        "daily-region-data/data/"
    )

    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[respondent][]": rto_code,
        "facets[type][]": "D",
        "start": start_date,
        "end": end_date
    }

    logger.info(
        f"Fetching EIA data | RTO={rto_code} | "
        f"Start={start_date} | End={end_date}"
    )

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data["response"]["data"])

        logger.info(f"Successfully fetched {len(df)} records for {rto_code}")

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise

    except KeyError as e:
        logger.error(f"Unexpected API response structure: {str(e)}")
        raise