import pandas as pd
import requests
import holidays
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

US_HOLIDAYS = holidays.UnitedStates()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Calendar
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_year"] = df.index.dayofyear

    # Holiday
    df["is_holiday"] = df.index.map(lambda x: 1 if x in US_HOLIDAYS else 0)

    # Lags
    for lag in [1, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df["demand_mwh"].shift(lag)

    # Rolling
    df["rolling_mean_7d"] = df["demand_mwh"].shift(1).rolling(7).mean()
    df["rolling_std_7d"] = df["demand_mwh"].shift(1).rolling(7).std()
    df["rolling_mean_14d"] = df["demand_mwh"].shift(1).rolling(14).mean()

    return df


def add_weather_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    df = df.copy()

    try:
        url = config.WEATHER_API_URL

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date": df.index.max().strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean",
            "timezone": "auto"
        }

        logger.info(f"Fetching weather data | lat={lat} | lon={lon}")

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        weather_df = pd.DataFrame({
            "temperature": data["daily"]["temperature_2m_mean"]
        }, index=pd.to_datetime(data["daily"]["time"]))

        df = df.merge(weather_df, left_index=True, right_index=True, how="left")

        df["temperature"] = df["temperature"].interpolate()

        logger.info("Weather features merged successfully")

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
        raise