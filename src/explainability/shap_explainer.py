"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations)
Provides global feature importance, beeswarm plots, dependence plots, and patient-level waterfall explanations.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import FIGURES_DIR, MODELS_DIR

def run_shap_analysis(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: List[str],
    max_display: int = 15,
    sample_size: int = 500
) -> Dict[str, Any]:
    """
    Computes SHAP values using TreeExplainer and outputs publication-grade explainability plots.
    """
    print("\n--> Initializing SHAP TreeExplainer...")
    
    # Subsample for efficient computation if test set is large
    if len(X_test) > sample_size:
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # In binary classification, if shap_values has 3 dimensions (samples, features, classes), select positive class (index 1)
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # 1. SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 7.5))
    shap.summary_plot(shap_values_class1, X_sample, feature_names=feature_names, max_display=max_display, show=False)
    plt.title("SHAP Summary (Beeswarm) Plot: Directional Feature Impact on Medication Response", pad=15)
    plt.tight_layout()
    beeswarm_path = FIGURES_DIR / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"Saved SHAP Beeswarm plot: {beeswarm_path}")

    # 2. SHAP Global Feature Importance Bar Plot
    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_values_class1, max_display=max_display, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", pad=15)
    plt.tight_layout()
    bar_path = FIGURES_DIR / "shap_feature_importance_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved SHAP Bar plot: {bar_path}")

    # 3. SHAP Waterfall Plot for an Individual Representative Patient
    patient_idx = 0
    plt.figure(figsize=(10, 6.5))
    shap.plots.waterfall(shap_values_class1[patient_idx], max_display=12, show=False)
    plt.title(f"Patient #{patient_idx} Personalized SHAP Decision Decomposition", pad=15)
    plt.tight_layout()
    waterfall_path = FIGURES_DIR / "shap_waterfall_patient_example.png"
    plt.savefig(waterfall_path, dpi=300)
    plt.close()
    print(f"Saved SHAP Patient Waterfall plot: {waterfall_path}")

    # Save explainer object
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    # Extract top features by mean absolute SHAP
    vals = np.abs(shap_values_class1.values).mean(axis=0)
    top_indices = np.argsort(vals)[::-1]
    top_features = [
        {"feature": feature_names[i], "mean_abs_shap": round(float(vals[i]), 4)}
        for i in top_indices[:max_display]
    ]

    return {
        "explainer": explainer,
        "shap_values": shap_values_class1,
        "top_features": top_features,
        "figures": {
            "beeswarm": str(beeswarm_path),
            "bar": str(bar_path),
            "waterfall": str(waterfall_path)
        }
    }

def explain_patient_prediction(
    model: Any,
    explainer: Any,
    preprocessor: Any,
    patient_features_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generates local feature attribution contributions for an individual patient in the CDSS application.
    """
    X_proc = preprocessor.transform(patient_features_df)
    shap_vals = explainer(X_proc)

    if len(shap_vals.shape) == 3:
        sv = shap_vals[0, :, 1]
    else:
        sv = shap_vals[0]

    contributions = []
    for feat_name, val, impact in zip(preprocessor.feature_names, X_proc.iloc[0].values, sv.values):
        contributions.append({
            "feature": feat_name,
            "transformed_value": round(float(val), 3),
            "shap_impact": round(float(impact), 4),
            "direction": "Favorable (Increases Success)" if impact > 0 else "Unfavorable (Reduces Success)"
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
    return contributions
