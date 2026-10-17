"""
Comprehensive Medical and Machine Learning Evaluation Metrics
Includes diagnostic clinical metrics (Sensitivity, Specificity, PPV, NPV, LR)
and standard AI performance indices (ROC-AUC, PR-AUC, F1, Brier Score, RMSE, R2).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Computes rigorous clinical diagnostic and machine learning metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Clinical Diagnostic Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Diagnostic Likelihood Ratios
    pos_lr = (sensitivity / (1.0 - specificity)) if (1.0 - specificity) > 0 else np.nan
    neg_lr = ((1.0 - sensitivity) / specificity) if specificity > 0 else np.nan

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision_PPV": float(ppv),
        "Recall_Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "NPV": float(npv),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Positive_Likelihood_Ratio": float(pos_lr) if not np.isnan(pos_lr) else None,
        "Negative_Likelihood_Ratio": float(neg_lr) if not np.isnan(neg_lr) else None,
        "True_Positives": int(tp),
        "False_Positives": int(fp),
        "True_Negatives": int(tn),
        "False_Negatives": int(fn)
    }

    if y_prob is not None:
        try:
            metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
            metrics["PR_AUC"] = float(average_precision_score(y_true, y_prob))
            metrics["Brier_Score"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            metrics["ROC_AUC"] = None
            metrics["PR_AUC"] = None
            metrics["Brier_Score"] = None

    return metrics

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes performance metrics for continuous blood pressure reduction models.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero in MAPE
    denom = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE_pct": float(mape),
        "Max_Error": float(np.max(np.abs(y_true - y_pred)))
    }
