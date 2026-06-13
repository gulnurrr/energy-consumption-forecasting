"""
Unit tests for src/tune.py — tune_model()

Strategy:
- XGBRegressor is mocked (speed): predict() returns a fixed array.
- Optuna study is mocked: optimize() calls the internal objective once
  so we can verify what the objective logs to MLflow.
- MLflow is NOT mocked: it writes to an isolated tmp_path directory,
  letting us verify actual logged params/metrics/tags after the call.
"""
import numpy as np
import pandas as pd
import mlflow
import optuna
import pytest
from unittest.mock import MagicMock, patch

from src.config import config
from src.tune import tune_model


# ── Shared test data ──────────────────────────────────────────────────────────


def _make_data(n: int = 60):
    """
    Minimal X/y pair.
    n=60 ensures TimeSeriesSplit(n_splits=5) produces non-empty folds
    (~10 samples per fold).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.random(n), "f2": rng.random(n)})
    y = pd.Series(rng.random(n) * 100_000, name="demand_mwh")
    return X, y


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mlflow_temp_dir(tmp_path):
    """
    Redirect MLflow to an isolated SQLite DB per test.
    SQLite is the recommended local backend in MLflow 2.x+
    (file store is deprecated).
    autouse=True means every test in this file gets a clean MLflow state.
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.set_tracking_uri(None)


@pytest.fixture
def fast_xgb():
    """
    Patch XGBRegressor so no real training happens.
    predict() returns a constant array shaped to the input length.
    """
    with patch("src.tune.XGBRegressor") as mock_cls:
        instance = MagicMock()
        instance.predict.side_effect = lambda X_val: np.ones(len(X_val)) * 50_000.0
        mock_cls.return_value = instance
        yield mock_cls


_BEST_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 2,
}


@pytest.fixture
def mock_study():
    """
    Mock Optuna study whose optimize() calls the objective exactly once
    with a controlled trial, then exposes deterministic best_params/best_value.
    """
    study = MagicMock()
    study.best_params = _BEST_PARAMS
    study.best_value = 1_234.56

    def run_one_trial(objective, n_trials, **kwargs):
        trial = MagicMock(spec=optuna.trial.Trial)
        trial.number = 0
        trial.suggest_int.return_value = 350
        trial.suggest_float.return_value = 0.04
        objective(trial)

    study.optimize.side_effect = run_one_trial
    return study


# ── Helper: query MLflow after a tune run ─────────────────────────────────────


def _get_parent_run():
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    assert experiment is not None, "MLflow experiment was not created"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.`mlflow.runName` = 'optuna_tuning'",
    )
    assert len(runs) == 1, f"Expected 1 parent run, found {len(runs)}"
    return runs[0]


def _get_trial_runs():
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    return [r for r in all_runs if r.data.tags.get("mlflow.runName", "").startswith("trial_")]


# ── Tests: return value ───────────────────────────────────────────────────────


class TestReturnValue:
    def test_returns_dict(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            result = tune_model(X, y, n_trials=1)
        assert isinstance(result, dict)

    def test_returns_study_best_params(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            result = tune_model(X, y, n_trials=1)
        assert result == _BEST_PARAMS

    def test_result_contains_expected_xgboost_keys(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            result = tune_model(X, y, n_trials=1)
        for key in ["n_estimators", "learning_rate", "max_depth", "subsample"]:
            assert key in result, f"Expected key '{key}' in returned params"


# ── Tests: Optuna integration ─────────────────────────────────────────────────


class TestOptunaBehavior:
    def test_create_study_direction_is_minimize(self, fast_xgb, mock_study):
        """Optuna must minimise RMSE, not maximise."""
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study) as mock_create:
            tune_model(X, y, n_trials=1)
        _, kwargs = mock_create.call_args
        assert kwargs.get("direction") == "minimize"

    def test_optimize_called_once(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        mock_study.optimize.assert_called_once()

    def test_n_trials_forwarded_to_optimize(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=7)
        _, kwargs = mock_study.optimize.call_args
        assert kwargs.get("n_trials") == 7

    def test_custom_n_trials_default_is_30(self, fast_xgb, mock_study):
        """Default n_trials value must stay 30 (search budget)."""
        import inspect
        sig = inspect.signature(tune_model)
        assert sig.parameters["n_trials"].default == 30


# ── Tests: MLflow parent run ──────────────────────────────────────────────────


class TestMLflowParentRun:
    def test_experiment_name_matches_config(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        assert experiment is not None

    def test_parent_run_created_with_correct_name(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        run = _get_parent_run()
        assert run.data.tags["mlflow.runName"] == "optuna_tuning"

    def test_best_params_logged_with_best_prefix(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        run = _get_parent_run()
        for key in _BEST_PARAMS:
            assert f"best_{key}" in run.data.params, f"Missing 'best_{key}' in parent run params"

    def test_best_cv_rmse_metric_logged(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        run = _get_parent_run()
        assert "best_cv_rmse" in run.data.metrics
        assert run.data.metrics["best_cv_rmse"] == pytest.approx(1_234.56)

    def test_task_tag_is_hyperparameter_tuning(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        run = _get_parent_run()
        assert run.data.tags["task"] == "hyperparameter_tuning"

    def test_n_trials_tag_matches_argument(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=7)
        run = _get_parent_run()
        assert run.data.tags["n_trials"] == "7"


# ── Tests: MLflow trial (nested) run — objective execution ────────────────────


class TestMLflowTrialRun:
    def test_one_nested_trial_run_created(self, fast_xgb, mock_study):
        """mock_study.optimize calls objective once → 1 trial run expected."""
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        assert len(_get_trial_runs()) == 1

    def test_trial_run_named_trial_0(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        trial_run = _get_trial_runs()[0]
        assert trial_run.data.tags["mlflow.runName"] == "trial_0"

    def test_trial_run_is_nested_under_parent(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        parent_run = _get_parent_run()
        trial_run = _get_trial_runs()[0]
        assert trial_run.data.tags["mlflow.parentRunId"] == parent_run.info.run_id

    def test_trial_run_logs_mean_cv_rmse(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        trial_run = _get_trial_runs()[0]
        assert "mean_cv_rmse" in trial_run.data.metrics
        assert trial_run.data.metrics["mean_cv_rmse"] > 0

    def test_trial_run_logs_mean_cv_mae(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        trial_run = _get_trial_runs()[0]
        assert "mean_cv_mae" in trial_run.data.metrics
        assert trial_run.data.metrics["mean_cv_mae"] > 0

    def test_trial_run_logs_params(self, fast_xgb, mock_study):
        X, y = _make_data()
        with patch("src.tune.optuna.create_study", return_value=mock_study):
            tune_model(X, y, n_trials=1)
        trial_run = _get_trial_runs()[0]
        for key in ["n_estimators", "learning_rate", "max_depth", "subsample"]:
            assert key in trial_run.data.params, f"'{key}' not logged in trial run"
