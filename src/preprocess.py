import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Rename target column FIRST
    df = df.rename(columns={
        "value": "demand_mwh"
    })

    # 2. Drop useless columns
    drop_cols = [
        "respondent",
        "respondent-name",
        "type",
        "type-name",
        "timezone",
        "timezone-description",
        "value-units"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Handle missing values
    df = df.dropna()

    # 4. Type conversion
    df["period"] = pd.to_datetime(df["period"])

    df = df.sort_values("period")
    df = df.set_index("period")

    # 5. Ensure numeric target — coerce drops any string that slipped past dropna()
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
    df = df.dropna(subset=["demand_mwh"])

    return df