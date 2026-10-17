"""
Configuration and Medical Knowledge Constants for Hypertension ML Framework
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "hypertension_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "hypertension_processed.csv"
DATA_DICTIONARY_PATH = DATA_DIR / "data_dictionary.json"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for p in [DATA_DIR / "raw", DATA_DIR / "processed", MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Clinical Guideline Thresholds (ACC/AHA 2017 & ESH 2023)
TARGET_SBP_GOAL = 130.0  # mmHg
TARGET_DBP_GOAL = 80.0   # mmHg
SIGNIFICANT_SBP_REDUCTION = 10.0  # mmHg drop is clinically meaningful
SIGNIFICANT_DBP_REDUCTION = 5.0   # mmHg drop

# Antihypertensive Drug Classes
DRUG_CLASSES = [
    "ACE_Inhibitor",        # Lisinopril, Ramipril, Enalapril
    "ARB",                  # Losartan, Valsartan, Telmisartan
    "CCB",                  # Calcium Channel Blocker: Amlodipine, Nifedipine
    "Thiazide_Diuretic",    # Hydrochlorothiazide, Chlorthalidone, Indapamide
    "Beta_Blocker"          # Metoprolol, Bisoprolol, Carvedilol
]

# Baseline Clinical Feature Names
DEMOGRAPHIC_FEATURES = ["age", "gender", "bmi", "smoking_status", "alcohol_intake", "physical_activity", "dietary_sodium_mg", "family_history_htn"]
VITAL_FEATURES = ["baseline_sbp", "baseline_dbp", "heart_rate", "hypertension_duration_years"]
LAB_FEATURES = ["serum_creatinine_mg_dl", "fasting_glucose_mg_dl", "hba1c_pct", "total_cholesterol_mg_dl", "ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl", "serum_potassium_meq_l", "serum_sodium_meq_l", "proteinuria"]
COMORBIDITY_FEATURES = ["diabetes_t2", "ckd", "cad", "dyslipidemia", "lvh"]
TREATMENT_FEATURES = ["drug_class", "dosage_level", "combination_therapy", "adherence_rate"]

NUMERICAL_FEATURES = [
    "age", "bmi", "dietary_sodium_mg", "baseline_sbp", "baseline_dbp", "heart_rate",
    "hypertension_duration_years", "serum_creatinine_mg_dl", "fasting_glucose_mg_dl",
    "hba1c_pct", "total_cholesterol_mg_dl", "ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl",
    "serum_potassium_meq_l", "serum_sodium_meq_l", "adherence_rate"
]

CATEGORICAL_FEATURES = [
    "gender", "smoking_status", "alcohol_intake", "physical_activity", "family_history_htn",
    "proteinuria", "diabetes_t2", "ckd", "cad", "dyslipidemia", "lvh", "drug_class",
    "dosage_level", "combination_therapy"
]

ENGINEERED_NUMERICAL = [
    "pulse_pressure", "mean_arterial_pressure", "egfr", "sodium_potassium_ratio",
    "atherogenic_index", "framingham_risk_proxy"
]

# Primary Target Variables
CLASSIFICATION_TARGET = "treatment_response"  # 1 = Responded/Controlled, 0 = Poor/Non-responsive
REGRESSION_TARGET_SBP = "delta_sbp"          # Baseline SBP - Post SBP
REGRESSION_TARGET_DBP = "delta_dbp"          # Baseline DBP - Post DBP
POST_SBP_TARGET = "post_treatment_sbp"
POST_DBP_TARGET = "post_treatment_dbp"
