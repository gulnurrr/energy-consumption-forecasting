import pandas as pd
import pytest
import requests
from unittest.mock import MagicMock, patch

from src.features import add_weather_features, create_features


def _make_df(n: int = 31, start: str = "2023-01-01") -> pd.DataFrame:
    """Build a clean test DataFrame with DatetimeIndex and monotonic demand."""
    idx = pd.date_range(start, periods=n, freq="D")
    values = [float(250_000 + i * 500) for i in range(n)]
    return pd.DataFrame({"demand_mwh": values}, index=idx)


# ── create_features ──────────────────────────────────────────────────────────


class TestCreateFeaturesColumns:
    EXPECTED_COLS = [
        "day_of_week", "month", "is_weekend", "day_of_year", "is_holiday",
        "lag_1", "lag_7", "lag_14", "lag_21", "lag_30",
        "rolling_mean_7d", "rolling_std_7d", "rolling_mean_14d",
    ]

    def test_all_expected_columns_created(self):
        result = create_features(_make_df())
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: '{col}'"

    def test_demand_mwh_preserved_in_output(self):
        result = create_features(_make_df())
        assert "demand_mwh" in result.columns


class TestCalendarFeatures:
    def test_day_of_week_monday_is_zero(self):
        # 2023-01-02 is a Monday
        result = create_features(_make_df(n=1, start="2023-01-02"))
        assert result["day_of_week"].iloc[0] == 0

    def test_day_of_week_sunday_is_six(self):
        # 2023-01-08 is a Sunday
        result = create_features(_make_df(n=1, start="2023-01-08"))
        assert result["day_of_week"].iloc[0] == 6

    def test_month_january_is_one(self):
        result = create_features(_make_df(n=1, start="2023-01-15"))
        assert result["month"].iloc[0] == 1

    def test_month_december_is_twelve(self):
        result = create_features(_make_df(n=1, start="2023-12-01"))
        assert result["month"].iloc[0] == 12

    def test_day_of_year_jan_first_is_one(self):
        result = create_features(_make_df(n=1, start="2023-01-01"))
        assert result["day_of_year"].iloc[0] == 1

    def test_day_of_year_dec_31_is_365(self):
        result = create_features(_make_df(n=1, start="2023-12-31"))
        assert result["day_of_year"].iloc[0] == 365

    def test_day_of_year_within_valid_range(self):
        result = create_features(_make_df(n=365))
        assert result["day_of_year"].between(1, 366).all()


class TestWeekendFeature:
    def test_saturday_is_weekend(self):
        # 2023-01-07 is Saturday
        result = create_features(_make_df(n=1, start="2023-01-07"))
        assert result["is_weekend"].iloc[0] == 1

    def test_sunday_is_weekend(self):
        # 2023-01-08 is Sunday
        result = create_features(_make_df(n=1, start="2023-01-08"))
        assert result["is_weekend"].iloc[0] == 1

    def test_monday_is_not_weekend(self):
        # 2023-01-02 is Monday
        result = create_features(_make_df(n=1, start="2023-01-02"))
        assert result["is_weekend"].iloc[0] == 0

    def test_friday_is_not_weekend(self):
        # 2023-01-06 is Friday
        result = create_features(_make_df(n=1, start="2023-01-06"))
        assert result["is_weekend"].iloc[0] == 0

    def test_is_weekend_is_binary(self):
        result = create_features(_make_df(n=31))
        assert result["is_weekend"].isin([0, 1]).all()


class TestHolidayFeature:
    def test_observed_new_years_day_is_holiday(self):
        # 2023-01-01 is Sunday → observed on 2023-01-02 (Monday)
        result = create_features(_make_df(n=1, start="2023-01-02"))
        assert result["is_holiday"].iloc[0] == 1

    def test_july_4th_is_holiday(self):
        result = create_features(_make_df(n=1, start="2023-07-04"))
        assert result["is_holiday"].iloc[0] == 1

    def test_regular_wednesday_is_not_holiday(self):
        # 2023-03-15 — no US federal holiday
        result = create_features(_make_df(n=1, start="2023-03-15"))
        assert result["is_holiday"].iloc[0] == 0

    def test_is_holiday_is_binary(self):
        result = create_features(_make_df(n=31))
        assert result["is_holiday"].isin([0, 1]).all()


class TestLagFeatures:
    def test_lag_1_equals_previous_row_demand(self):
        df = _make_df(n=5)
        result = create_features(df)
        assert result["lag_1"].iloc[1] == pytest.approx(df["demand_mwh"].iloc[0])
        assert result["lag_1"].iloc[2] == pytest.approx(df["demand_mwh"].iloc[1])

    def test_lag_1_nan_for_first_row(self):
        result = create_features(_make_df(n=5))
        assert pd.isna(result["lag_1"].iloc[0])

    def test_lag_7_equals_demand_seven_days_prior(self):
        df = _make_df(n=10)
        result = create_features(df)
        assert result["lag_7"].iloc[7] == pytest.approx(df["demand_mwh"].iloc[0])

    def test_lag_30_all_nan_when_fewer_than_31_rows(self):
        """shift(30) on 30 rows maps all positions to non-existent rows."""
        result = create_features(_make_df(n=30))
        assert result["lag_30"].isna().all()

    def test_lag_30_has_value_with_sufficient_rows(self):
        df = _make_df(n=35)
        result = create_features(df)
        # Last row's lag_30 must equal demand from 30 rows before
        assert result["lag_30"].iloc[-1] == pytest.approx(df["demand_mwh"].iloc[-31])


class TestRollingFeatures:
    def test_rolling_mean_7d_nan_for_first_seven_rows(self):
        """shift(1) + rolling(7) needs 8 rows of history before producing a value."""
        result = create_features(_make_df(n=7))
        assert result["rolling_mean_7d"].isna().all()

    def test_rolling_mean_7d_correct_at_row_7(self):
        """At row index 7: mean of demand at indices 0–6 (after shift+rolling)."""
        df = _make_df(n=10)
        result = create_features(df)
        expected = df["demand_mwh"].iloc[:7].mean()
        assert result["rolling_mean_7d"].iloc[7] == pytest.approx(expected, rel=1e-6)

    def test_rolling_std_7d_nan_for_insufficient_history(self):
        result = create_features(_make_df(n=7))
        assert result["rolling_std_7d"].isna().all()

    def test_rolling_mean_14d_nan_for_insufficient_history(self):
        result = create_features(_make_df(n=14))
        assert result["rolling_mean_14d"].isna().all()


class TestCreateFeaturesSideEffects:
    def test_does_not_mutate_input_columns(self):
        df = _make_df(n=10)
        original_cols = df.columns.tolist()
        create_features(df)
        assert df.columns.tolist() == original_cols

    def test_does_not_mutate_input_values(self):
        df = _make_df(n=10)
        original_values = df["demand_mwh"].tolist()
        create_features(df)
        assert df["demand_mwh"].tolist() == original_values


# ── add_weather_features ─────────────────────────────────────────────────────


def _make_weather_mock(df: pd.DataFrame) -> MagicMock:
    """Return a mock requests.Response whose JSON matches df's date range."""
    dates = df.index.strftime("%Y-%m-%d").tolist()
    temps = [15.0 + i * 0.5 for i in range(len(df))]
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"daily": {"time": dates, "temperature_2m_mean": temps}}
    return mock


class TestAddWeatherFeatures:
    def test_temperature_column_added(self):
        df = _make_df(n=5)
        with patch("src.features.requests.get", return_value=_make_weather_mock(df)):
            result = add_weather_features(df, lat=39.7, lon=-104.9)
        assert "temperature" in result.columns

    def test_temperature_values_correct(self):
        df = _make_df(n=5)
        with patch("src.features.requests.get", return_value=_make_weather_mock(df)):
            result = add_weather_features(df, lat=39.7, lon=-104.9)
        assert result["temperature"].iloc[0] == pytest.approx(15.0)
        assert result["temperature"].iloc[1] == pytest.approx(15.5)

    def test_row_count_preserved(self):
        """Left-merge must not drop any original rows."""
        df = _make_df(n=5)
        with patch("src.features.requests.get", return_value=_make_weather_mock(df)):
            result = add_weather_features(df, lat=39.7, lon=-104.9)
        assert len(result) == len(df)

    def test_does_not_mutate_input(self):
        df = _make_df(n=5)
        original_cols = df.columns.tolist()
        with patch("src.features.requests.get", return_value=_make_weather_mock(df)):
            add_weather_features(df, lat=39.7, lon=-104.9)
        assert df.columns.tolist() == original_cols

    def test_http_error_raises_request_exception(self):
        df = _make_df(n=5)
        mock = MagicMock()
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
        with patch("src.features.requests.get", return_value=mock):
            with pytest.raises(requests.exceptions.RequestException):
                add_weather_features(df, lat=39.7, lon=-104.9)

    def test_connection_error_propagated(self):
        df = _make_df(n=5)
        with patch(
            "src.features.requests.get",
            side_effect=requests.exceptions.ConnectionError,
        ):
            with pytest.raises(requests.exceptions.RequestException):
                add_weather_features(df, lat=39.7, lon=-104.9)

    def test_timeout_propagated(self):
        df = _make_df(n=5)
        with patch(
            "src.features.requests.get",
            side_effect=requests.exceptions.Timeout,
        ):
            with pytest.raises(requests.exceptions.RequestException):
                add_weather_features(df, lat=39.7, lon=-104.9)
