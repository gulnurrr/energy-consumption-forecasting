import numpy as np
import mlflow
import optuna

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

# Suppress Optuna's per-trial stdout noise; progress comes through logger
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(X, y, n_trials: int = 30) -> dict:
    """
    Bayesian hyperparameter search via Optuna.

    Each trial is logged as a nested MLflow run under a parent
    "optuna_tuning" run so the full search history is queryable in the UI.
    Returns best_params dict ready to pass directly to train_model().
    """
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    tscv = TimeSeriesSplit(n_splits=5)

    with mlflow.start_run(run_name="optuna_tuning") as parent_run:
        logger.info(
            f"Optuna study started | n_trials={n_trials} "
            f"| mlflow_run={parent_run.info.run_id}"
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                # log=True biases sampling toward smaller LR values, which
                # tend to generalise better for time-series boosting
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": 42,
                "objective": "reg:squarederror",
            }

            fold_rmses, fold_maes = [], []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = XGBRegressor(**params)
                model.fit(X_train, y_train)

                preds = model.predict(X_val)
                fold_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
                fold_maes.append(float(mean_absolute_error(y_val, preds)))

            mean_rmse = float(np.mean(fold_rmses))
            mean_mae = float(np.mean(fold_maes))

            # Each trial → one nested child run
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_rmse", mean_rmse)
                mlflow.log_metric("mean_cv_mae", mean_mae)

            return mean_rmse  # Optuna minimises this

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials)

        # Summarise the winning trial back onto the parent run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_rmse", study.best_value)
        mlflow.set_tag("n_trials", str(n_trials))
        mlflow.set_tag("task", "hyperparameter_tuning")

    logger.info(
        f"Tuning complete | best_cv_rmse={study.best_value:.2f} "
        f"| best_params={study.best_params}"
    )
    return study.best_params
