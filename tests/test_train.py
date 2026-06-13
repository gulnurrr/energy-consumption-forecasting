"""
Unit tests for src/train.py — train_model()

Strategy:
- XGBRegressor is mocked (speed): real 5-fold CV would be too slow.
- mlflow.sklearn.log_model and mlflow.log_artifact are mocked:
  avoids binary serialization of a MagicMock into the MLflow artifact store.
- joblib.dump is NOT mocked: we verify model.pkl is actually written to disk.
- MLflow tracking is NOT mocked: writes to an isolated tmp_path SQLite DB
  so logged params/metrics/tags are verified against the actual store.
"""
import json
import numpy as np
import pandas as pd
import mlflow
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import config
from src.train import train_model


# ── Shared test data ───────────────────────────────────────────────────────────


def _make_featured_df(n: int = 60) -> pd.DataFrame:
    """
    Minimal DataFrame matching train_model() input shape:
    DatetimeIndex + demand_mwh target + two feature columns.
    n=60 gives TimeSeriesSplit(n_splits=5) non-empty folds (~10 rows each).
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "demand_mwh": rng.random(n) * 300_000 + 200_000,
            "temperature": rng.random(n) * 30,
            "day_of_week": rng.integers(0, 7, n).astype(float),
        },
        index=idx,
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mlflow_temp_dir(tmp_path):
    """
    Redirect MLflow to an isolated SQLite DB per test.
    autouse=True means every test gets a clean MLflow state.
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.set_tracking_uri(None)


@pytest.fixture(autouse=True)
def mock_external_io():
    """
    Mock all external I/O that would fail or be too slow in tests:
    - joblib.dump: MagicMock is not picklable by joblib; side_effect
      touches the file so existence checks still pass.
    - mlflow.sklearn.log_model: MagicMock doesn't implement the sklearn
      interface that MLflow's serializer expects.
    - mlflow.log_artifact: avoids copying files into the MLflow artifact store.
    """
    def _touch_file(_model, path):
        Path(str(path)).touch()

    with (
        patch("src.train.joblib.dump", side_effect=_touch_file),
        patch("src.train.mlflow.sklearn.log_model"),
        patch("src.train.mlflow.log_artifact"),
    ):
        yield


@pytest.fixture
def fast_xgb():
    """
    Patch XGBRegressor so no real training happens.
    predict() returns a constant array so RMSE/MAE can still be computed.
    All XGBRegressor() calls return the same shared MagicMock instance.
    """
    with patch("src.train.XGBRegressor") as mock_cls:
        instance = MagicMock()
        instance.predict.side_effect = lambda X: np.ones(len(X)) * 250_000.0
        mock_cls.return_value = instance
        yield mock_cls


@pytest.fixture
def artifact_dir(tmp_path):
    """Redirect artifact writes to tmp_path so tests don't touch production dirs."""
    with patch("src.train.config.ARTIFACT_DIR", tmp_path):
        yield tmp_path


# ── Helper: query MLflow after a train run ─────────────────────────────────────


def _get_train_run():
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    assert experiment is not None, "MLflow experiment was not created"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.`mlflow.runName` = 'xgboost_ts_final'",
    )
    assert len(runs) == 1, f"Expected 1 training run, found {len(runs)}"
    return runs[0]


# ── Tests: return value ────────────────────────────────────────────────────────


class TestReturnValue:
    def test_returns_xgb_instance(self, fast_xgb, artifact_dir):
        result = train_model(_make_featured_df())
        assert result is fast_xgb.return_value

    def test_fit_called_on_final_model(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        assert fast_xgb.return_value.fit.called

    def test_fit_called_for_cv_folds_plus_final(self, fast_xgb, artifact_dir):
        """5 CV folds + 1 final fit = 6 total XGBRegressor().fit() calls."""
        train_model(_make_featured_df())
        assert fast_xgb.return_value.fit.call_count == 6


# ── Tests: NaN handling ────────────────────────────────────────────────────────


class TestNaNHandling:
    def test_nan_rows_dropped_without_error(self, fast_xgb, artifact_dir):
        df = _make_featured_df()
        df.iloc[:5, 0] = np.nan
        train_model(df)  # must not raise

    def test_final_fit_receives_no_nan(self, fast_xgb, artifact_dir):
        df = _make_featured_df()
        df.iloc[:3, 1] = np.nan
        train_model(df)
        final_fit_call = fast_xgb.return_value.fit.call_args_list[-1]
        X_passed = final_fit_call[0][0]
        assert not X_passed.isnull().any().any()


# ── Tests: hyperparameter params ──────────────────────────────────────────────


class TestParams:
    def test_default_params_used_when_none(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df(), best_params=None)
        first_call_kwargs = fast_xgb.call_args_list[0][1]
        assert first_call_kwargs["n_estimators"] == 300
        assert first_call_kwargs["max_depth"] == 5
        assert first_call_kwargs["learning_rate"] == 0.05

    def test_custom_params_forwarded_to_xgb(self, fast_xgb, artifact_dir):
        custom = {"n_estimators": 42, "max_depth": 3, "learning_rate": 0.01}
        train_model(_make_featured_df(), best_params=custom)
        first_call_kwargs = fast_xgb.call_args_list[0][1]
        assert first_call_kwargs["n_estimators"] == 42
        assert first_call_kwargs["max_depth"] == 3
        assert first_call_kwargs["learning_rate"] == 0.01

    def test_objective_always_set_to_squarederror(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        first_call_kwargs = fast_xgb.call_args_list[0][1]
        assert first_call_kwargs["objective"] == "reg:squarederror"

    def test_random_state_always_42(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        first_call_kwargs = fast_xgb.call_args_list[0][1]
        assert first_call_kwargs["random_state"] == 42


# ── Tests: MLflow run ─────────────────────────────────────────────────────────


class TestMLflowRun:
    def test_experiment_name_matches_config(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        assert exp is not None

    def test_run_name_is_xgboost_ts_final(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        assert run.data.tags["mlflow.runName"] == "xgboost_ts_final"

    def test_hyperparams_logged_as_params(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        assert "n_estimators" in run.data.params
        assert "max_depth" in run.data.params
        assert "learning_rate" in run.data.params

    def test_custom_params_appear_in_mlflow(self, fast_xgb, artifact_dir):
        custom = {"n_estimators": 99, "max_depth": 4, "learning_rate": 0.02}
        train_model(_make_featured_df(), best_params=custom)
        run = _get_train_run()
        assert run.data.params["n_estimators"] == "99"

    def test_per_fold_rmse_logged_for_all_5_folds(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        for fold in range(1, 6):
            assert f"rmse_fold_{fold}" in run.data.metrics, f"Missing rmse_fold_{fold}"

    def test_per_fold_mae_logged_for_all_5_folds(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        for fold in range(1, 6):
            assert f"mae_fold_{fold}" in run.data.metrics, f"Missing mae_fold_{fold}"

    def test_mean_cv_rmse_logged(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        assert "mean_cv_rmse" in run.data.metrics
        assert run.data.metrics["mean_cv_rmse"] > 0

    def test_mean_cv_mae_logged(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        assert "mean_cv_mae" in run.data.metrics
        assert run.data.metrics["mean_cv_mae"] > 0

    def test_task_tag_is_energy_forecasting(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        assert run.data.tags["task"] == "energy_forecasting"

    def test_model_type_tag_is_xgboost(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        run = _get_train_run()
        assert run.data.tags["model_type"] == "xgboost"

    def test_n_features_tag_matches_actual_feature_count(self, fast_xgb, artifact_dir):
        df = _make_featured_df()
        train_model(df)
        run = _get_train_run()
        expected_n_features = len(df.columns) - 1  # exclude demand_mwh
        assert run.data.tags["n_features"] == str(expected_n_features)

    def test_n_samples_tag_matches_row_count(self, fast_xgb, artifact_dir):
        df = _make_featured_df(n=60)
        train_model(df)
        run = _get_train_run()
        assert run.data.tags["n_samples"] == "60"

    def test_n_samples_reflects_nan_drop(self, fast_xgb, artifact_dir):
        df = _make_featured_df(n=60)
        df.iloc[:4, 0] = np.nan  # 4 rows will be dropped
        train_model(df)
        run = _get_train_run()
        assert run.data.tags["n_samples"] == "56"


# ── Tests: artifact persistence ───────────────────────────────────────────────


class TestArtifacts:
    def test_model_pkl_written_to_artifact_dir(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        assert (artifact_dir / "model.pkl").exists()

    def test_feature_columns_json_written(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        assert (artifact_dir / "feature_columns.json").exists()

    def test_feature_columns_is_valid_list(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        cols = json.loads((artifact_dir / "feature_columns.json").read_text())
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_feature_columns_excludes_target(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        cols = json.loads((artifact_dir / "feature_columns.json").read_text())
        assert "demand_mwh" not in cols

    def test_feature_columns_includes_all_features(self, fast_xgb, artifact_dir):
        train_model(_make_featured_df())
        cols = json.loads((artifact_dir / "feature_columns.json").read_text())
        assert "temperature" in cols
        assert "day_of_week" in cols
