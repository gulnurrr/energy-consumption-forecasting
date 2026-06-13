import pandas as pd
import pytest
from src.preprocess import preprocess_data


class TestColumnRenaming:
    def test_value_renamed_to_demand_mwh(self, raw_eia_df):
        result = preprocess_data(raw_eia_df)
        assert "demand_mwh" in result.columns
        assert "value" not in result.columns

    def test_extra_non_eia_columns_survive(self):
        df = pd.DataFrame(
            {"period": ["2023-01-01"], "value": [100.0], "my_custom_col": ["x"]}
        )
        result = preprocess_data(df)
        assert "my_custom_col" in result.columns


class TestDropColumns:
    EIA_METADATA_COLS = [
        "respondent", "respondent-name", "type", "type-name",
        "timezone", "timezone-description", "value-units",
    ]

    def test_all_eia_metadata_columns_dropped(self, raw_eia_df):
        result = preprocess_data(raw_eia_df)
        for col in self.EIA_METADATA_COLS:
            assert col not in result.columns, f"'{col}' should have been dropped"

    def test_partial_eia_columns_does_not_raise(self, raw_eia_df):
        """Drop list uses a safe filter — missing columns are silently skipped."""
        df = raw_eia_df.drop(columns=["timezone", "timezone-description"])
        result = preprocess_data(df)
        assert "demand_mwh" in result.columns


class TestIndex:
    def test_period_becomes_datetime_index(self, raw_eia_df):
        result = preprocess_data(raw_eia_df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_name_is_period(self, raw_eia_df):
        result = preprocess_data(raw_eia_df)
        assert result.index.name == "period"

    def test_output_sorted_ascending(self, raw_eia_df):
        """raw_eia_df fixture is intentionally unsorted to verify sort step."""
        result = preprocess_data(raw_eia_df)
        assert result.index.is_monotonic_increasing


class TestMissingValues:
    def test_rows_with_nan_dropped(self):
        df = pd.DataFrame(
            {
                "period": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "value": [100.0, None, 300.0],
            }
        )
        result = preprocess_data(df)
        assert len(result) == 2
        assert result["demand_mwh"].isnull().sum() == 0

    def test_all_nan_rows_returns_empty_df(self):
        df = pd.DataFrame(
            {"period": ["2023-01-01", "2023-01-02"], "value": [None, None]}
        )
        result = preprocess_data(df)
        assert len(result) == 0
        assert isinstance(result.index, pd.DatetimeIndex)


class TestTypeCoercion:
    def test_demand_mwh_is_numeric(self, raw_eia_df):
        result = preprocess_data(raw_eia_df)
        assert pd.api.types.is_numeric_dtype(result["demand_mwh"])

    def test_string_numbers_coerced_to_float(self):
        df = pd.DataFrame(
            {"period": ["2023-01-01", "2023-01-02"], "value": ["280000", "295000"]}
        )
        result = preprocess_data(df)
        assert result["demand_mwh"].iloc[0] == pytest.approx(280_000.0)

    def test_non_parseable_strings_become_nan_then_dropped(self):
        df = pd.DataFrame(
            {"period": ["2023-01-01", "2023-01-02"], "value": [100.0, "N/A"]}
        )
        result = preprocess_data(df)
        assert len(result) == 1
        assert result["demand_mwh"].iloc[0] == pytest.approx(100.0)


class TestSideEffects:
    def test_does_not_mutate_input(self, raw_eia_df):
        original_columns = raw_eia_df.columns.tolist()
        original_len = len(raw_eia_df)
        preprocess_data(raw_eia_df)
        assert raw_eia_df.columns.tolist() == original_columns
        assert len(raw_eia_df) == original_len
