import os
import requests
import pandas as pd
from src.config import config

# Updated ingest function with the officially correct EIA API v2 structure
def fetch_daily_grid_data(rto_code: str = "PSCO") -> pd.DataFrame:
    api_key = config.EIA_API_KEY or os.getenv("EIA_API_KEY")
    
    # The correct endpoint for daily grid data is 'daily-region-data'
    # Start and End dates are passed as direct URL parameters
    base_url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
    
    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[respondent][]": rto_code,
        "facets[type][]": "D",
        "start": "2016-01-01",
        "end": "2026-06-01" 
    }
    
    print(f"Ingest Pipeline: Fetching 10 years of DAILY data for RTO: {rto_code}...")
    response = requests.get(base_url, params=params, timeout=15)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["response"]["data"])
        print(f"Ingest Pipeline: Successfully fetched {len(df)} records.")
        return df
    else:
        raise Exception(f" API Error {response.status_code}: {response.text}")