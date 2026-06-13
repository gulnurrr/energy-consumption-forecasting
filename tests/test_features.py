# tests/test_features.py
import pandas as pd
import pytest
from src.features import create_features

def test_create_features_output_columns():
    # Test verisi oluştur
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({'demand_mwh': range(10)}, index=idx)
    
    # Fonksiyonu çalıştır
    result = create_features(df)
    
    # Kontrol et: Beklediğimiz sütunlar oluşmuş mu?
    expected_cols = ['day_of_week', 'month', 'is_weekend', 'day_of_year', 'lag_7', 'rolling_mean_7d']
    for col in expected_cols:
        assert col in result.columns