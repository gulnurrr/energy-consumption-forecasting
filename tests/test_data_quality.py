import pandas as pd
import pytest
from src.data_quality import data_quality


def _valid_df(n: int = 10) -> pd.DataFrame:
    """Minimal DataFrame that passes every quality check."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"demand_mwh": [float(i * 1000) for i in range(1, n + 1)]}, index=idx)


class TestReturnType:
    def test_returns_two_element_tuple(self):
        result = data_quality(_valid_df())
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_bool(self):
        is_valid, _ = data_quality(_valid_df())
        assert isinstance(is_valid, bool)

    def test_second_element_is_list(self):
        _, errors = data_quality(_valid_df())
        assert isinstance(errors, list)

    def test_error_list_contains_strings(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"demand_mwh": [100.0, None, -10.0]}, index=idx)
        _, errors = data_quality(df)
        for err in errors:
            assert isinstance(err, str)


class TestHappyPath:
    def test_valid_dataframe_passes(self):
        is_valid, errors = data_quality(_valid_df())
        assert is_valid is True
        assert errors == []

    def test_single_row_df_passes(self):
        is_valid, errors = data_quality(_valid_df(n=1))
        assert is_valid is True
        assert errors == []

    def test_zero_demand_is_valid(self):
        """Zero is not negative — grid data can briefly reach zero."""
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"demand_mwh": [0.0, 0.0, 0.0]}, index=idx)
        is_valid, _ = data_quality(df)
        assert is_valid is True


class TestNonDatetimeIndex:
    def test_range_index_fails(self):
        df = pd.DataFrame({"demand_mwh": [100.0, 200.0]}, index=[0, 1])
        is_valid, errors = data_quality(df)
        assert is_valid is False
        assert any("DatetimeIndex" in e for e in errors)

    def test_string_index_fails(self):
        df = pd.DataFrame(
            {"demand_mwh": [100.0, 200.0]}, index=["2023-01-01", "2023-01-02"]
        )
        is_valid, errors = data_quality(df)
        assert is_valid is False


class TestSortOrder:
    def test_descending_index_fails(self):
        idx = pd.to_datetime(["2023-01-03", "2023-01-02", "2023-01-01"])
        df = pd.DataFrame({"demand_mwh": [300.0, 200.0, 100.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False
        assert any("sorted" in e.lower() for e in errors)

    def test_shuffled_index_fails(self):
        idx = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-02"])
        df = pd.DataFrame({"demand_mwh": [100.0, 300.0, 200.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False


class TestNaNCheck:
    def test_single_nan_fails(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"demand_mwh": [100.0, None, 300.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False
        assert any("NaN" in e for e in errors)

    def test_all_nan_fails(self):
        idx = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({"demand_mwh": [None, None]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False


class TestDuplicateTimestamps:
    def test_duplicate_timestamps_fail(self):
        idx = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"])
        df = pd.DataFrame({"demand_mwh": [100.0, 200.0, 300.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False
        assert any("duplicate" in e.lower() for e in errors)

    def test_all_same_timestamp_fails(self):
        idx = pd.to_datetime(["2023-01-01", "2023-01-01"])
        df = pd.DataFrame({"demand_mwh": [100.0, 200.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False


class TestNegativeValues:
    def test_single_negative_value_fails(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"demand_mwh": [100.0, -1.0, 300.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False
        assert any("negative" in e.lower() for e in errors)

    def test_all_negative_fails(self):
        idx = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({"demand_mwh": [-100.0, -200.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False

    def test_very_small_negative_fails(self):
        """Floating-point edge: -0.0001 should still be caught."""
        idx = pd.date_range("2023-01-01", periods=2, freq="D")
        df = pd.DataFrame({"demand_mwh": [100.0, -0.0001]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False


class TestMultipleErrors:
    def test_nan_and_negative_both_reported(self):
        """Both conditions active simultaneously — both must appear in the error list."""
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"demand_mwh": [None, -100.0, 300.0]}, index=idx)
        is_valid, errors = data_quality(df)
        assert is_valid is False
        assert any("NaN" in e for e in errors)
        assert any("negative" in e.lower() for e in errors)

    def test_error_count_matches_number_of_violations(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"demand_mwh": [None, -100.0, 300.0]}, index=idx)
        _, errors = data_quality(df)
        assert len(errors) == 2
