"""
Clinically Grounded Synthetic Dataset Generator for Hypertension ML Framework
Simulates patient cohorts adhering to NHANES statistical distributions and
ACC/AHA/ESH pharmacological response dynamics.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.config import (
    RAW_DATA_PATH, DATA_DICTIONARY_PATH, RANDOM_STATE, DRUG_CLASSES,
    TARGET_SBP_GOAL, TARGET_DBP_GOAL, SIGNIFICANT_SBP_REDUCTION, SIGNIFICANT_DBP_REDUCTION
)

def calculate_ckd_epi_egfr(creatinine: float, age: float, is_female: bool) -> float:
    """Computes Estimated Glomerular Filtration Rate using the CKD-EPI equation."""
    if is_female:
        kappa = 0.7
        alpha = -0.241
        gender_mult = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        gender_mult = 1.0
    
    scr_k = creatinine / kappa
    egfr = 142.0 * (min(scr_k, 1.0) ** alpha) * (max(scr_k, 1.0) ** -1.200) * (0.9938 ** age) * gender_mult
    return round(float(egfr), 1)

def generate_hypertension_cohort(
    n_samples: int = 5000,
    seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a realistic clinical cohort of hypertensive patients under medical therapy.
    """
    np.random.seed(seed)

    # 1. Demographics
    patient_ids = [f"HTN-{100000 + i}" for i in range(n_samples)]
    age = np.random.normal(loc=58.5, scale=11.5, size=n_samples).clip(28, 88).round(1)
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.51, 0.49])
    is_female = (gender == "Female")

    # BMI with slight positive skew
    bmi = np.random.gamma(shape=25.0, scale=1.18, size=n_samples).clip(18.5, 48.0).round(1)

    smoking = np.random.choice(["Never", "Former", "Current"], size=n_samples, p=[0.55, 0.25, 0.20])
    alcohol = np.random.choice(["None", "Moderate", "High"], size=n_samples, p=[0.50, 0.38, 0.12])
    physical_activity = np.random.choice(["Sedentary", "Moderate", "Active"], size=n_samples, p=[0.45, 0.40, 0.15])
    
    # Dietary Sodium (mg/day) - High intake correlates with western diet / hypertension
    dietary_sodium = (np.random.normal(loc=3400, scale=800, size=n_samples) + (alcohol == "High") * 400).clip(1500, 6500).round(0)
    family_history = np.random.choice(["Yes", "No"], size=n_samples, p=[0.62, 0.38])
    htn_duration = np.random.exponential(scale=6.0, size=n_samples).clip(0.5, 30.0).round(1)

    # 2. Baseline Hemodynamics
    # Higher age and BMI push baseline SBP higher
    base_sbp_noise = np.random.normal(0, 8, n_samples)
    baseline_sbp = (130 + 0.35 * (age - 50) + 0.5 * (bmi - 25) + 0.002 * (dietary_sodium - 2000) + base_sbp_noise).clip(135, 215).round(1)
    
    # DBP correlates with SBP but peaks around age 55-60 then declines (arterial stiffness / widening pulse pressure)
    age_dbp_factor = np.where(age < 55, 0.25 * (age - 40), 0.25 * 15 - 0.15 * (age - 55))
    base_dbp_noise = np.random.normal(0, 5, n_samples)
    baseline_dbp = (82 + age_dbp_factor + 0.3 * (bmi - 25) + 0.25 * (baseline_sbp - 140) + base_dbp_noise).clip(78, 128).round(1)
    
    # Resting heart rate
    heart_rate = (np.random.normal(loc=74, scale=9.5, size=n_samples) + (smoking == "Current") * 4 + (physical_activity == "Sedentary") * 4).clip(52, 115).round(0)

    # 3. Laboratory / Biochemical Markers
    # Serum Creatinine influenced by gender, age, and chronic conditions
    base_scr = np.where(is_female, np.random.normal(0.85, 0.20, n_samples), np.random.normal(1.05, 0.25, n_samples))
    serum_creatinine = (base_scr + 0.005 * htn_duration).clip(0.5, 3.2).round(2)
    
    # Compute eGFR
    egfr_vals = [calculate_ckd_epi_egfr(serum_creatinine[i], age[i], is_female[i]) for i in range(n_samples)]
    egfr = np.array(egfr_vals)
    ckd = np.where(egfr < 60.0, "Yes", "No")

    # Fasting glucose & HbA1c
    glucose_latent = np.random.normal(loc=98, scale=22, size=n_samples) + (bmi > 30) * 18
    fasting_glucose = glucose_latent.clip(70, 260).round(1)
    hba1c = (4.4 + 0.025 * fasting_glucose + np.random.normal(0, 0.35, n_samples)).clip(4.8, 12.5).round(1)
    diabetes = np.where((hba1c >= 6.5) | (fasting_glucose >= 126), "Yes", "No")

    # Lipid profile
    total_chol = np.random.normal(loc=205, scale=38, size=n_samples).clip(120, 340).round(1)
    hdl = np.where(is_female, np.random.normal(54, 12, n_samples), np.random.normal(46, 10, n_samples)).clip(25, 95).round(1)
    ldl = (total_chol * 0.62 + np.random.normal(0, 15, n_samples)).clip(50, 240).round(1)
    triglycerides = np.random.exponential(scale=110, size=n_samples) + 60
    triglycerides = triglycerides.clip(60, 500).round(1)
    dyslipidemia = np.where((ldl >= 130) | (total_chol >= 220) | (triglycerides >= 200), "Yes", "No")

    # Electrolytes
    serum_k = np.random.normal(loc=4.3, scale=0.38, size=n_samples).clip(3.4, 5.8).round(2)
    serum_na = np.random.normal(loc=140.5, scale=2.5, size=n_samples).clip(132, 148).round(1)

    # Proteinuria / Albuminuria
    proteinuria_prob = np.where(diabetes == "Yes", 0.40, 0.12) + np.where(ckd == "Yes", 0.35, 0.0)
    proteinuria_list = []
    for p in proteinuria_prob:
        p_pos = float(np.clip(p * 0.4, 0.05, 0.50))
        p_trace = float(np.clip(p * 0.6, 0.10, 0.35))
        p_norm = max(0.05, 1.0 - (p_pos + p_trace))
        probs = np.array([p_norm, p_trace, p_pos])
        probs = probs / probs.sum()
        proteinuria_list.append(np.random.choice(["Normal", "Trace", "Positive"], p=probs))
    proteinuria = proteinuria_list

    # Additional comorbidities
    cad_prob = (age / 100.0) * 0.25 + (diabetes == "Yes") * 0.15 + (smoking == "Current") * 0.12
    cad = [np.random.choice(["Yes", "No"], p=[min(0.7, p), 1.0 - min(0.7, p)]) for p in cad_prob]
    
    lvh_prob = (baseline_sbp > 160) * 0.28 + (htn_duration > 8) * 0.20 + 0.05
    lvh = [np.random.choice(["Yes", "No"], p=[min(0.65, p), 1.0 - min(0.65, p)]) for p in lvh_prob]

    # 4. Medication Assignment (Real-World Clinical Distribution)
    # Clinical guideline preference: Diabetic -> ACEi/ARB; Elderly/Isolated SBP -> CCB/Diuretic; CAD/Tachycardia -> BB
    drug_class = np.random.choice(DRUG_CLASSES, size=n_samples, p=[0.26, 0.24, 0.22, 0.16, 0.12])
    dosage_level = np.random.choice(["Standard", "High"], size=n_samples, p=[0.65, 0.35])
    combination_therapy = np.random.choice(["Monotherapy", "Dual_Combination"], size=n_samples, p=[0.60, 0.40])
    
    # Adherence rate (Morisky scale simulation, 0.5 to 1.0)
    adherence_rate = np.random.beta(a=9, b=2, size=n_samples).clip(0.45, 1.0).round(2)

    # 5. Pharmacodynamic Response Simulation (Evidence-based physiological engine)
    # Base drop proportional to baseline severity (Law of Initial Value)
    delta_sbp_base = 0.22 * (baseline_sbp - 120)
    delta_dbp_base = 0.18 * (baseline_dbp - 75)

    # Drug class specific response modifiers
    delta_sbp_drug = np.zeros(n_samples)
    delta_dbp_drug = np.zeros(n_samples)

    for i in range(n_samples):
        dc = drug_class[i]
        d_age = age[i]
        d_na = dietary_sodium[i]
        d_hr = heart_rate[i]
        d_dm = (diabetes[i] == "Yes")
        d_prot = (proteinuria[i] in ["Trace", "Positive"])
        d_ckd = (ckd[i] == "Yes")

        if dc == "ACE_Inhibitor":
            # Highly effective in younger, diabetic, proteinuria; attenuated if high dietary salt
            eff = 12.0 + (d_age < 55) * 3.5 + d_dm * 2.5 + d_prot * 2.0 - (d_na > 4000) * 2.5
            d_sbp = eff
            d_dbp = eff * 0.58
        elif dc == "ARB":
            # Similar to ACEi with excellent tolerability
            eff = 12.5 + (d_age < 60) * 3.0 + d_dm * 2.8 + d_prot * 2.2 - (d_na > 4200) * 2.0
            d_sbp = eff
            d_dbp = eff * 0.60
        elif dc == "CCB":
            # Very consistent, excels in elderly, high salt, isolated systolic HTN
            eff = 14.0 + (d_age >= 60) * 4.0 + (d_na > 3500) * 2.5 + (bmi[i] > 30) * 1.5
            d_sbp = eff
            d_dbp = eff * 0.52
        elif dc == "Thiazide_Diuretic":
            # Excels with high sodium intake, elderly, but diminished in advanced CKD
            eff = 11.5 + (d_age >= 60) * 3.0 + (d_na > 3500) * 3.5 - d_ckd * 4.0
            d_sbp = eff
            d_dbp = eff * 0.54
        elif dc == "Beta_Blocker":
            # Excels in high resting heart rate, CAD, younger patients
            eff = 9.5 + (d_hr > 80) * 5.0 + (cad[i] == "Yes") * 3.0 + (d_age < 55) * 2.5 - (d_age >= 68) * 3.0
            d_sbp = eff
            d_dbp = eff * 0.65
        else:
            d_sbp = 10.0
            d_dbp = 6.0

        # Dosage effect: High dose gives ~20% boost
        if dosage_level[i] == "High":
            d_sbp *= 1.22
            d_dbp *= 1.20

        # Combination therapy gives significant synergy (+35%)
        if combination_therapy[i] == "Dual_Combination":
            d_sbp *= 1.38
            d_dbp *= 1.35

        # Adherence modulates response directly
        adh = adherence_rate[i]
        d_sbp *= (0.35 + 0.65 * adh)
        d_dbp *= (0.35 + 0.65 * adh)

        delta_sbp_drug[i] = d_sbp
        delta_dbp_drug[i] = d_dbp

    # Biological Gaussian noise (residual individual variability)
    sbp_noise = np.random.normal(0, 3.2, n_samples)
    dbp_noise = np.random.normal(0, 2.4, n_samples)

    final_delta_sbp = (delta_sbp_base * 0.4 + delta_sbp_drug * 0.6 + sbp_noise).clip(0, 55).round(1)
    final_delta_dbp = (delta_dbp_base * 0.4 + delta_dbp_drug * 0.6 + dbp_noise).clip(0, 35).round(1)

    post_sbp = (baseline_sbp - final_delta_sbp).round(1)
    post_dbp = (baseline_dbp - final_delta_dbp).round(1)

    # Response Definition according to ACC/AHA & ESH clinical criteria:
    # 1 (Responded/Controlled): Achieved goal (SBP < 130 & DBP < 80) OR achieved significant reduction (delta SBP >= 10 & delta DBP >= 5)
    achieved_goal = (post_sbp < TARGET_SBP_GOAL) & (post_dbp < TARGET_DBP_GOAL)
    achieved_significant_drop = (final_delta_sbp >= SIGNIFICANT_SBP_REDUCTION) & (final_delta_dbp >= SIGNIFICANT_DBP_REDUCTION)
    treatment_response = (achieved_goal | achieved_significant_drop).astype(int)

    # Assemble complete DataFrame
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "smoking_status": smoking,
        "alcohol_intake": alcohol,
        "physical_activity": physical_activity,
        "dietary_sodium_mg": dietary_sodium,
        "family_history_htn": family_history,
        "hypertension_duration_years": htn_duration,
        "baseline_sbp": baseline_sbp,
        "baseline_dbp": baseline_dbp,
        "heart_rate": heart_rate,
        "serum_creatinine_mg_dl": serum_creatinine,
        "fasting_glucose_mg_dl": fasting_glucose,
        "hba1c_pct": hba1c,
        "total_cholesterol_mg_dl": total_chol,
        "ldl_mg_dl": ldl,
        "hdl_mg_dl": hdl,
        "triglycerides_mg_dl": triglycerides,
        "serum_potassium_meq_l": serum_k,
        "serum_sodium_meq_l": serum_na,
        "proteinuria": proteinuria,
        "diabetes_t2": diabetes,
        "ckd": ckd,
        "cad": cad,
        "dyslipidemia": dyslipidemia,
        "lvh": lvh,
        "drug_class": drug_class,
        "dosage_level": dosage_level,
        "combination_therapy": combination_therapy,
        "adherence_rate": adherence_rate,
        "post_treatment_sbp": post_sbp,
        "post_treatment_dbp": post_dbp,
        "delta_sbp": final_delta_sbp,
        "delta_dbp": final_delta_dbp,
        "treatment_response": treatment_response
    })

    # Data dictionary documentation
    data_dict = {
        "dataset_name": "Hypertension Medication Response Clinical Benchmark",
        "total_samples": n_samples,
        "number_of_features": df.shape[1],
        "response_rate": float(treatment_response.mean()),
        "target_definitions": {
            "treatment_response": "1 if (Post_SBP < 130 and Post_DBP < 80 mmHg) or (Delta_SBP >= 10 and Delta_DBP >= 5 mmHg); 0 otherwise.",
            "delta_sbp": "Systolic blood pressure reduction in mmHg (Baseline SBP - Post SBP)",
            "delta_dbp": "Diastolic blood pressure reduction in mmHg (Baseline DBP - Post DBP)"
        },
        "feature_descriptions": {
            "age": "Patient age in years [28 - 88]",
            "gender": "Biological sex: Male or Female",
            "bmi": "Body Mass Index in kg/m2 [18.5 - 48.0]",
            "smoking_status": "Current, Former, or Never smoker",
            "alcohol_intake": "None, Moderate, or High alcohol consumption",
            "physical_activity": "Sedentary, Moderate, or Active lifestyle",
            "dietary_sodium_mg": "Estimated daily sodium intake in milligrams",
            "family_history_htn": "Family history of essential hypertension / early cardiovascular disease",
            "hypertension_duration_years": "Duration since primary hypertension diagnosis in years",
            "baseline_sbp": "Pre-treatment systolic blood pressure in mmHg",
            "baseline_dbp": "Pre-treatment diastolic blood pressure in mmHg",
            "heart_rate": "Resting heart rate in beats per minute",
            "serum_creatinine_mg_dl": "Serum creatinine level in mg/dL",
            "fasting_glucose_mg_dl": "Fasting blood plasma glucose in mg/dL",
            "hba1c_pct": "Glycated hemoglobin percentage",
            "total_cholesterol_mg_dl": "Serum total cholesterol in mg/dL",
            "ldl_mg_dl": "Low-density lipoprotein cholesterol in mg/dL",
            "hdl_mg_dl": "High-density lipoprotein cholesterol in mg/dL",
            "triglycerides_mg_dl": "Serum triglycerides in mg/dL",
            "serum_potassium_meq_l": "Serum potassium concentration in mEq/L",
            "serum_sodium_meq_l": "Serum sodium concentration in mEq/L",
            "proteinuria": "Urinary protein assessment: Normal, Trace, Positive",
            "diabetes_t2": "Type 2 Diabetes Mellitus diagnosis (Yes/No)",
            "ckd": "Chronic Kidney Disease defined by eGFR < 60 mL/min/1.73m2 (Yes/No)",
            "cad": "Coronary Artery Disease diagnosis (Yes/No)",
            "dyslipidemia": "Dyslipidemia diagnosis (Yes/No)",
            "lvh": "Left Ventricular Hypertrophy confirmed by ECG/Echo (Yes/No)",
            "drug_class": "Prescribed antihypertensive drug class (ACE_Inhibitor, ARB, CCB, Thiazide_Diuretic, Beta_Blocker)",
            "dosage_level": "Dosage level: Standard (1x) or High (2x)",
            "combination_therapy": "Monotherapy vs Dual_Combination regimen",
            "adherence_rate": "Patient medication adherence fraction [0.45 - 1.00]"
        }
    }

    return df, data_dict

def save_generated_dataset(output_path: str = RAW_DATA_PATH, dict_path: str = DATA_DICTIONARY_PATH):
    df, meta = generate_hypertension_cohort(n_samples=5000, seed=RANDOM_STATE)
    df.to_csv(output_path, index=False)
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"Generated dataset saved to: {output_path} with {len(df)} records.")
    print(f"Data dictionary saved to: {dict_path}")
    print(f"Cohort Response Rate: {df['treatment_response'].mean() * 100:.2f}%")
    return df, meta

if __name__ == "__main__":
    save_generated_dataset()
