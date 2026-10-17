"""
Clinical Feature Engineering Module for Hypertension ML Framework
Derives established physiological indices, hemodynamic parameters, and composite cardiovascular risk markers.
Fully vectorized for ultra-fast performance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all advanced physiological and clinical features for modeling using vectorized operations.
    """
    df_eng = df.copy()

    # 1. Hemodynamics
    df_eng["pulse_pressure"] = (df_eng["baseline_sbp"] - df_eng["baseline_dbp"]).round(1)
    df_eng["mean_arterial_pressure"] = (df_eng["baseline_dbp"] + (df_eng["pulse_pressure"] / 3.0)).round(1)

    # 2. Kidney Function (eGFR) vectorized via CKD-EPI formula
    age = df_eng["age"].values.astype(float)
    is_female = (df_eng["gender"].astype(str).str.lower() == "female").values
    scr = df_eng["serum_creatinine_mg_dl"].values.astype(float)

    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)
    gender_mult = np.where(is_female, 1.012, 1.0)

    scr_k = scr / kappa
    egfr = 142.0 * (np.minimum(scr_k, 1.0) ** alpha) * (np.maximum(scr_k, 1.0) ** -1.200) * (0.9938 ** age) * gender_mult
    df_eng["egfr"] = np.round(egfr, 2)

    # 3. Metabolic & Electrolyte Indices
    potassium_proxy_mg = df_eng["serum_potassium_meq_l"] * 39.1
    df_eng["sodium_potassium_ratio"] = (df_eng["dietary_sodium_mg"] / (potassium_proxy_mg + 1e-5)).round(2)

    # Atherogenic Index of Plasma: log10(TG / HDL)
    df_eng["atherogenic_index"] = np.log10(np.maximum(df_eng["triglycerides_mg_dl"], 1.0) / np.maximum(df_eng["hdl_mg_dl"], 1.0)).round(3)

    # 4. Composite Cardiovascular Risk Proxy (Framingham Proxy)
    points = np.maximum(0.0, (age - 35.0) / 5.0)
    sbp = df_eng["baseline_sbp"].values.astype(float)
    points += np.where(sbp >= 160.0, 4.0, np.where(sbp >= 140.0, 2.5, np.where(sbp >= 130.0, 1.5, 0.0)))
    
    smoking = df_eng["smoking_status"].astype(str).str.lower().values
    points += np.where(smoking == "current", 3.0, np.where(smoking == "former", 1.0, 0.0))

    diabetes = df_eng["diabetes_t2"].astype(str).str.lower().values
    points += np.where(diabetes == "yes", 3.5, 0.0)

    chol = df_eng["total_cholesterol_mg_dl"].values.astype(float)
    points += np.where(chol >= 240.0, 3.0, np.where(chol >= 200.0, 1.5, 0.0))

    hdl = df_eng["hdl_mg_dl"].values.astype(float)
    points += np.where(hdl < 40.0, 2.0, np.where(hdl >= 60.0, -1.0, 0.0))

    df_eng["framingham_risk_proxy"] = np.round(points, 2)

    # 5. Non-linear Interactions
    df_eng["sbp_sodium_interaction"] = (df_eng["baseline_sbp"] * (df_eng["dietary_sodium_mg"] / 1000.0)).round(2)
    df_eng["age_pulse_pressure_interaction"] = (df_eng["age"] * df_eng["pulse_pressure"]).round(1)
    df_eng["adherence_dosage_interaction"] = np.where(
        df_eng["dosage_level"] == "High",
        df_eng["adherence_rate"] * 1.25,
        df_eng["adherence_rate"] * 1.0
    ).round(2)

    return df_eng

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    raw = pd.read_csv(RAW_DATA_PATH)
    eng = engineer_features(raw)
    print("Vectorized feature engineering completed successfully!")
    print(f"Original shape: {raw.shape}, Engineered shape: {eng.shape}")
    print(eng[["pulse_pressure", "mean_arterial_pressure", "egfr", "sodium_potassium_ratio", "framingham_risk_proxy"]].head())
