import os

# Must run before any src.* import — config.py calls Settings() at module
# level and raises RuntimeError when EIA_API_KEY is absent.
os.environ.setdefault("EIA_API_KEY", "test-api-key-placeholder")

import pandas as pd
import pytest


@pytest.fixture
def demand_df():
    """
    31-row preprocessed DataFrame with DatetimeIndex + demand_mwh.
    31 rows so lag_30 can produce at least one non-NaN value.
    Shared across test_features and test_data_quality.
    """
    idx = pd.date_range("2023-01-01", periods=31, freq="D")
    values = [float(250_000 + i * 500) for i in range(31)]
    return pd.DataFrame({"demand_mwh": values}, index=idx)


@pytest.fixture
def raw_eia_df():
    """
    Mimics the raw DataFrame you get after calling pd.DataFrame(response["data"]).
    Period column is intentionally UNSORTED to test sort behaviour in preprocess.
    """
    return pd.DataFrame(
        {
            "period": ["2023-01-03", "2023-01-01", "2023-01-02"],
            "value": [310_000.0, 280_000.0, 295_000.0],
            "respondent": ["PSCO"] * 3,
            "respondent-name": ["Public Service Co of Colorado"] * 3,
            "type": ["D"] * 3,
            "type-name": ["Demand"] * 3,
            "timezone": ["MT"] * 3,
            "timezone-description": ["Mountain Time"] * 3,
            "value-units": ["megawatthours"] * 3,
        }
    )
