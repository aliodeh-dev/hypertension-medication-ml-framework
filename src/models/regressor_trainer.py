"""
Regressor Training Module for Continuous Blood Pressure Drop Estimation (ΔSBP & ΔDBP)
Trains and benchmarks XGBoost, LightGBM, CatBoost, Random Forest, and Ridge Regressors.
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR
from src.evaluation.metrics import compute_regression_metrics

def get_regressor_candidates(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Returns candidate regression models."""
    return {
        "XGBoost_Regressor": xgb.XGBRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=2
        ),
        "LightGBM_Regressor": lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbosity=-1,
            n_jobs=2
        ),
        "CatBoost_Regressor": cb.CatBoostRegressor(
            iterations=200,
            learning_rate=0.07,
            depth=5,
            random_seed=random_state,
            verbose=False,
            thread_count=2
        ),
        "Random_Forest_Regressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=2
        ),
        "Ridge_Regressor": Ridge(
            alpha=1.0,
            random_state=random_state
        )
    }

def train_and_evaluate_regressors(
    X_train: pd.DataFrame,
    y_sbp_train: pd.Series,
    y_dbp_train: pd.Series,
    X_test: pd.DataFrame,
    y_sbp_test: pd.Series,
    y_dbp_test: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, str]:
    """
    Trains regressors for both ΔSBP and ΔDBP, evaluates RMSE, MAE, and R2 on the test set.
    """
    candidates_sbp = get_regressor_candidates()
    candidates_dbp = get_regressor_candidates()

    trained_sbp_models = {}
    trained_dbp_models = {}
    benchmark_records = []

    print(f"\n=======================================================")
    print(f"  COMMENCING CONTINUOUS BP REDUCTION REGRESSION BENCHMARKING")
    print(f"=======================================================\n")

    for name, model_sbp in candidates_sbp.items():
        print(f"--> Training Regressors: {name} (Delta_SBP & Delta_DBP)...")
        # Train for SBP
        model_sbp.fit(X_train, y_sbp_train)
        trained_sbp_models[name] = model_sbp
        y_sbp_pred = model_sbp.predict(X_test)
        metrics_sbp = compute_regression_metrics(y_sbp_test.values, y_sbp_pred)

        # Train for DBP
        model_dbp = candidates_dbp[name]
        model_dbp.fit(X_train, y_dbp_train)
        trained_dbp_models[name] = model_dbp
        y_dbp_pred = model_dbp.predict(X_test)
        metrics_dbp = compute_regression_metrics(y_dbp_test.values, y_dbp_pred)

        # Save model artifacts
        joblib.dump(model_sbp, MODELS_DIR / f"{name.lower()}_sbp.joblib")
        joblib.dump(model_dbp, MODELS_DIR / f"{name.lower()}_dbp.joblib")

        record = {
            "Model": name,
            "SBP_RMSE_mmHg": round(metrics_sbp["RMSE"], 3),
            "SBP_MAE_mmHg": round(metrics_sbp["MAE"], 3),
            "SBP_R2": round(metrics_sbp["R2"], 4),
            "SBP_MAPE_pct": round(metrics_sbp["MAPE_pct"], 2),
            "DBP_RMSE_mmHg": round(metrics_dbp["RMSE"], 3),
            "DBP_MAE_mmHg": round(metrics_dbp["MAE"], 3),
            "DBP_R2": round(metrics_dbp["R2"], 4),
            "DBP_MAPE_pct": round(metrics_dbp["MAPE_pct"], 2),
        }
        benchmark_records.append(record)
        print(f"    [Result] {name} -> SBP RMSE: {record['SBP_RMSE_mmHg']} mmHg (R2: {record['SBP_R2']:.3f}) | DBP RMSE: {record['DBP_RMSE_mmHg']} mmHg (R2: {record['DBP_R2']:.3f})")

    df_benchmark = pd.DataFrame(benchmark_records).sort_values(by="SBP_RMSE_mmHg", ascending=True).reset_index(drop=True)
    best_regressor_name = df_benchmark.iloc[0]["Model"]

    # Save benchmark table
    df_benchmark.to_csv(REPORTS_DIR / "regressor_benchmark.csv", index=False)
    with open(REPORTS_DIR / "regressor_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(df_benchmark.to_dict(orient="records"), f, indent=4)

    # Save best models
    joblib.dump(trained_sbp_models[best_regressor_name], MODELS_DIR / "best_regressor_sbp.joblib")
    joblib.dump(trained_dbp_models[best_regressor_name], MODELS_DIR / "best_regressor_dbp.joblib")

    print(f"\n=======================================================")
    print(f"  TOP PERFORMING REGRESSOR: {best_regressor_name} (SBP RMSE = {df_benchmark.iloc[0]['SBP_RMSE_mmHg']} mmHg)")
    print(f"=======================================================\n")

    return trained_sbp_models, trained_dbp_models, df_benchmark, best_regressor_name
