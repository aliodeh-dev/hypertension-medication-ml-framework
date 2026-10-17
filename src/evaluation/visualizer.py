"""
Publication-Quality Clinical Visualization Module
Generates high-resolution figures for Master's thesis presentation and evaluation reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

from src.config import FIGURES_DIR

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "roc_curves_comparison.png"
):
    """Plots multi-model ROC curves with AUC annotations."""
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10")

    for idx, (name, model) in enumerate(models_dict.items()):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            from sklearn.metrics import roc_auc_score
            auc_val = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=palette[idx % 10], lw=2.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curves - Model Comparison")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves figure: {save_path}")

def plot_precision_recall_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "precision_recall_comparison.png"
):
    """Plots Precision-Recall curves with Average Precision."""
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10")

    for idx, (name, model) in enumerate(models_dict.items()):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            from sklearn.metrics import average_precision_score
            ap_val = average_precision_score(y_test, y_prob)
            plt.plot(rec, prec, label=f"{name} (AP = {ap_val:.3f})", color=palette[idx % 10], lw=2.2)

    no_skill = y_test.mean()
    plt.plot([0, 1], [no_skill, no_skill], "k--", lw=1.5, label=f"Baseline Prevalence ({no_skill:.2f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision-Recall (PR) Curves - Model Comparison")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Precision-Recall curves figure: {save_path}")

def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
    save_path: Path = FIGURES_DIR / "confusion_matrix.png"
):
    """Plots dual-panel confusion matrix: raw counts and normalized proportions."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Poor Response (0)", "Responded (1)"],
                yticklabels=["Poor Response (0)", "Responded (1)"])
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Diagnosis")
    axes[0].set_ylabel("Actual Clinical Outcome")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=True, ax=axes[1],
                xticklabels=["Poor Response (0)", "Responded (1)"],
                yticklabels=["Poor Response (0)", "Responded (1)"])
    axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted Diagnosis")
    axes[1].set_ylabel("Actual Clinical Outcome")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix figure: {save_path}")

def plot_calibration_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "calibration_reliability.png"
):
    """Plots reliability calibration curves comparing predicted risks to observed response rates."""
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10")

    for idx, (name, model) in enumerate(models_dict.items()):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy="uniform")
            plt.plot(prob_pred, prob_true, marker="o", lw=2, label=name, color=palette[idx % 10])

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Response Probability")
    plt.ylabel("Observed Fraction of Responders")
    plt.title("Clinical Calibration Curves (Reliability Diagram)")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Calibration figure: {save_path}")

def plot_feature_importance_figure(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "XGBoost",
    top_n: int = 15,
    save_path: Path = FIGURES_DIR / "feature_importance.png"
):
    """Plots top-N feature importances bar chart."""
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=df_imp, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title(f"{model_name} - Top {top_n} Most Influential Clinical Features")
    plt.xlabel("Relative Importance Weight")
    plt.ylabel("Clinical / Physiological Feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance figure: {save_path}")

def plot_bp_reduction_distributions(
    df: pd.DataFrame,
    save_path: Path = FIGURES_DIR / "bp_reduction_by_drug_class.png"
):
    """Plots violin and boxplots of Systolic and Diastolic reductions by Drug Class."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=df, x="drug_class", y="delta_sbp", hue="combination_therapy", palette="Set2", ax=axes[0])
    axes[0].set_title("Systolic Blood Pressure Reduction (ΔSBP) by Drug Class")
    axes[0].set_xlabel("Antihypertensive Drug Class")
    axes[0].set_ylabel("ΔSBP Drop (mmHg)")
    axes[0].tick_params(axis="x", rotation=25)

    sns.boxplot(data=df, x="drug_class", y="delta_dbp", hue="combination_therapy", palette="Set2", ax=axes[1])
    axes[1].set_title("Diastolic Blood Pressure Reduction (ΔDBP) by Drug Class")
    axes[1].set_xlabel("Antihypertensive Drug Class")
    axes[1].set_ylabel("ΔDBP Drop (mmHg)")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved BP reduction distributions figure: {save_path}")

def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Path = FIGURES_DIR / "clinical_correlation_matrix.png"
):
    """Plots correlation heatmap among major hemodynamic, laboratory, and outcome features."""
    num_cols = [
        "age", "bmi", "dietary_sodium_mg", "baseline_sbp", "baseline_dbp", "heart_rate",
        "serum_creatinine_mg_dl", "fasting_glucose_mg_dl", "hba1c_pct", "total_cholesterol_mg_dl",
        "pulse_pressure", "mean_arterial_pressure", "egfr", "sodium_potassium_ratio",
        "adherence_rate", "delta_sbp", "delta_dbp", "treatment_response"
    ]
    corr_cols = [c for c in num_cols if c in df.columns]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.6, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Clinical, Physiological, and Response Variables", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Correlation Matrix figure: {save_path}")

def plot_regression_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_label: str = "ΔSBP (mmHg)",
    model_name: str = "Best Regressor",
    save_path: Path = FIGURES_DIR / "regression_performance.png"
):
    """Plots Actual vs. Predicted scatter and residual analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    residuals = y_true - y_pred

    # Scatter: Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.45, color="#1f77b4", edgecolor="none", s=25)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Identity Line (y=x)")
    axes[0].set_xlabel(f"Actual Observed {target_label}")
    axes[0].set_ylabel(f"Predicted {target_label}")
    axes[0].set_title(f"{model_name}: Actual vs. Predicted {target_label}")
    axes[0].legend()

    # Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2ca02c", bins=30)
    axes[1].axvline(0, color="k", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Prediction Residual (Actual - Predicted)")
    axes[1].set_ylabel("Frequency Density")
    axes[1].set_title(f"Residual Error Distribution (Mean={np.mean(residuals):.2f}, Std={np.std(residuals):.2f})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Regression analysis figure: {save_path}")
