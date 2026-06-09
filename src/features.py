# src/features.py
import pandas as pd
import requests

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds date-based features (day of week, month, weekend flag) to the dataframe."""
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def add_weather_features(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    # Open-Meteo API URL
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={df.index.min().strftime('%Y-%m-%d')}&end_date={df.index.max().strftime('%Y-%m-%d')}&daily=temperature_2m_mean&timezone=auto"
    
    response = requests.get(url).json()
    
    if 'daily' in response:
        temp_data = response['daily']['temperature_2m_mean']
        dates = response['daily']['time']
        df_weather = pd.DataFrame({'temp_c': temp_data}, index=pd.to_datetime(dates))
        
        df = df.merge(df_weather, left_index=True, right_index=True, how='left')
        df['temp_c'] = df['temp_c'].ffill().bfill()
    else:
        print("Open-Meteo verisi alınamadı!")
        df['temp_c'] = 0 
        
    return df