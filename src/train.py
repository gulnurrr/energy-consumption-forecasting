import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

# Name under which the model appears in MLflow Model Registry
_REGISTERED_MODEL_NAME = "energy-demand-xgboost"


def train_model(df, best_params: dict | None = None):
    """
    Train a final XGBoost model with TimeSeriesSplit cross-validation.

    Logs per-fold metrics, mean CV scores, and the final model to MLflow.
    Registers the model in the MLflow Model Registry under
    'energy-demand-xgboost' so it can be loaded by alias elsewhere.
    """
    if config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # Drop NaN rows from lag/rolling features before splitting
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} NaN rows from lag/rolling warm-up | {len(df)} rows remaining")

    X = df.drop(columns=["demand_mwh"])
    y = df["demand_mwh"]

    tscv = TimeSeriesSplit(n_splits=5)

    with mlflow.start_run(run_name="xgboost_ts_final") as run:
        logger.info(f"MLflow run started | run_id={run.info.run_id}")

        if best_params is None:
            best_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
            }

        mlflow.log_params(best_params)
        mlflow.set_tag("task", "energy_forecasting")
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("n_features", str(X.shape[1]))
        mlflow.set_tag("n_samples", str(len(X)))

        # ── Cross-validation ─────────────────────────────────────────────
        fold_rmses: list[float] = []
        fold_maes: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBRegressor(
                **best_params,
                objective="reg:squarederror",
                random_state=42,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            mae = float(mean_absolute_error(y_val, preds))

            fold_rmses.append(rmse)
            fold_maes.append(mae)

            mlflow.log_metric(f"rmse_fold_{fold}", rmse)
            mlflow.log_metric(f"mae_fold_{fold}", mae)
            logger.info(f"Fold {fold}/5 | RMSE={rmse:.2f} | MAE={mae:.2f}")

        mean_rmse = float(np.mean(fold_rmses))
        mean_mae = float(np.mean(fold_maes))
        mlflow.log_metric("mean_cv_rmse", mean_rmse)
        mlflow.log_metric("mean_cv_mae", mean_mae)
        logger.info(f"CV complete | mean_rmse={mean_rmse:.2f} | mean_mae={mean_mae:.2f}")

        # ── Final model on full data ──────────────────────────────────────
        final_model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
        )
        final_model.fit(X, y)

        # ── Local artifacts ───────────────────────────────────────────────
        model_path = config.ARTIFACT_DIR / "model.pkl"
        features_path = config.ARTIFACT_DIR / "feature_columns.json"

        joblib.dump(final_model, model_path)

        feature_cols = X.columns.tolist()
        with open(features_path, "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(features_path))

        # ── MLflow Model Registry ─────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=_REGISTERED_MODEL_NAME,
        )

        logger.info(
            f"Model registered | name={_REGISTERED_MODEL_NAME} "
            f"| mean_cv_rmse={mean_rmse:.2f} | mean_cv_mae={mean_mae:.2f}"
        )

    return final_model
