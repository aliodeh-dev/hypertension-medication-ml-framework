"""
Classifier Training and Cross-Validation Module for Hypertension ML Framework
Trains, cross-validates, and benchmarks XGBoost, LightGBM, CatBoost, Random Forest,
Logistic Regression, and Multi-Layer Perceptron (Neural Network).
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR
from src.evaluation.metrics import compute_classification_metrics

def get_classifier_candidates(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Returns initialized classifier models with clinically appropriate hyperparameters."""
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=2
        ),
        "LightGBM": lgb.LGBMClassifier(
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
        "CatBoost": cb.CatBoostClassifier(
            iterations=200,
            learning_rate=0.07,
            depth=5,
            random_seed=random_state,
            verbose=False,
            thread_count=2
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=9,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=2
        ),
        "Logistic_Regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ),
        "Neural_Network_MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=350,
            early_stopping=True,
            random_state=random_state
        )
    }

def train_and_evaluate_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """
    Performs 5-fold Stratified Cross-Validation, trains candidate models on full training set,
    evaluates on unseen test set, saves artifacts, and returns trained models with performance ranking.
    """
    candidates = get_classifier_candidates()
    trained_models = {}
    benchmark_records = []
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"\n=======================================================")
    print(f"  COMMENCING CLASSIFIER BENCHMARKING (N={len(candidates)} MODELS)")
    print(f"=======================================================\n")

    for name, model in candidates.items():
        print(f"--> Training and Cross-Validating: {name}...")

        # 1. 5-Fold Stratified Cross Validation
        scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]
        cv_res = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
        
        cv_acc_mean = float(cv_res["test_accuracy"].mean())
        cv_acc_std = float(cv_res["test_accuracy"].std())
        cv_auc_mean = float(cv_res["test_roc_auc"].mean())
        cv_auc_std = float(cv_res["test_roc_auc"].std())
        cv_f1_mean = float(cv_res["test_f1"].mean())
        cv_f1_std = float(cv_res["test_f1"].std())

        # 2. Fit on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Save model artifact
        model_path = MODELS_DIR / f"{name.lower()}_classifier.joblib"
        joblib.dump(model, model_path)

        # 3. Evaluate on Holdout Test Set
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        test_metrics = compute_classification_metrics(y_test.values, y_test_pred, y_test_prob)

        record = {
            "Model": name,
            "CV_Accuracy_Mean": round(cv_acc_mean, 4),
            "CV_Accuracy_Std": round(cv_acc_std, 4),
            "CV_ROC_AUC_Mean": round(cv_auc_mean, 4),
            "CV_ROC_AUC_Std": round(cv_auc_std, 4),
            "CV_F1_Mean": round(cv_f1_mean, 4),
            "Test_Accuracy": round(test_metrics["Accuracy"], 4),
            "Test_Balanced_Accuracy": round(test_metrics["Balanced_Accuracy"], 4),
            "Test_Sensitivity_Recall": round(test_metrics["Recall_Sensitivity"], 4),
            "Test_Specificity": round(test_metrics["Specificity"], 4),
            "Test_Precision_PPV": round(test_metrics["Precision_PPV"], 4),
            "Test_NPV": round(test_metrics["NPV"], 4),
            "Test_F1_Score": round(test_metrics["F1_Score"], 4),
            "Test_ROC_AUC": round(test_metrics["ROC_AUC"], 4) if test_metrics["ROC_AUC"] is not None else None,
            "Test_PR_AUC": round(test_metrics["PR_AUC"], 4) if test_metrics["PR_AUC"] is not None else None,
            "Test_Brier_Score": round(test_metrics["Brier_Score"], 4) if test_metrics["Brier_Score"] is not None else None,
            "Test_MCC": round(test_metrics["MCC"], 4)
        }
        benchmark_records.append(record)
        print(f"    [Result] {name} -> Test Acc: {record['Test_Accuracy']:.4f} | Test AUC: {record['Test_ROC_AUC']} | Test F1: {record['Test_F1_Score']:.4f}")

    # Build Comparison DataFrame
    df_benchmark = pd.DataFrame(benchmark_records).sort_values(by="Test_ROC_AUC", ascending=False).reset_index(drop=True)
    best_model_name = df_benchmark.iloc[0]["Model"]

    # Save benchmark reports
    df_benchmark.to_csv(REPORTS_DIR / "classifier_benchmark.csv", index=False)
    with open(REPORTS_DIR / "classifier_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(df_benchmark.to_dict(orient="records"), f, indent=4)

    # Save best model reference
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODELS_DIR / "best_classifier.joblib")
    with open(MODELS_DIR / "best_classifier_meta.json", "w", encoding="utf-8") as f:
        json.dump({"best_model_name": best_model_name, "test_auc": float(df_benchmark.iloc[0]["Test_ROC_AUC"])}, f, indent=4)

    print(f"\n=======================================================")
    print(f"  TOP PERFORMING CLASSIFIER: {best_model_name} (ROC-AUC = {df_benchmark.iloc[0]['Test_ROC_AUC']:.4f})")
    print(f"=======================================================\n")

    return trained_models, df_benchmark, best_model_name
