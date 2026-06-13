import pandas as pd
def data_quality(df):
    errors = []

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        errors.append("Time index not sorted")

    if df["demand_mwh"].isnull().sum() > 0:
        errors.append("Target has NaN")

    if df.index.duplicated().sum() > 0:
        errors.append("Duplicate timestamps")

    if (df["demand_mwh"] < 0).any():
        errors.append("Negative values found")

    return len(errors) == 0, errors