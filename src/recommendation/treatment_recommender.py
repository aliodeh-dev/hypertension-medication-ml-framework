"""
Personalized Clinical Treatment Recommender Engine for Hypertension Management
Simulates counterfactual drug regimens across all first-line classes, estimates response probabilities
and blood pressure drops, and incorporates clinical practice guidelines (ACC/AHA & ESH).
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.config import DRUG_CLASSES, MODELS_DIR
from src.feature_engineering import engineer_features

class TreatmentRecommender:
    def __init__(
        self,
        classifier=None,
        regressor_sbp=None,
        regressor_dbp=None,
        preprocessor=None
    ):
        self.classifier = classifier or joblib.load(MODELS_DIR / "best_classifier.joblib")
        self.regressor_sbp = regressor_sbp or joblib.load(MODELS_DIR / "best_regressor_sbp.joblib")
        self.regressor_dbp = regressor_dbp or joblib.load(MODELS_DIR / "best_regressor_dbp.joblib")
        self.preprocessor = preprocessor or joblib.load(MODELS_DIR / "preprocessor.joblib")

    def evaluate_clinical_guidelines(self, patient_profile: Dict[str, Any], drug_class: str) -> Dict[str, Any]:
        """
        Evaluates clinical guidelines (ACC/AHA 2017 & ESH 2023) for suitability and warnings.
        """
        age = float(patient_profile.get("age", 55))
        sbp = float(patient_profile.get("baseline_sbp", 150))
        dbp = float(patient_profile.get("baseline_dbp", 90))
        hr = float(patient_profile.get("heart_rate", 75))
        dm = str(patient_profile.get("diabetes_t2", "No")).lower() == "yes"
        ckd = str(patient_profile.get("ckd", "No")).lower() == "yes"
        cad = str(patient_profile.get("cad", "No")).lower() == "yes"
        k_val = float(patient_profile.get("serum_potassium_meq_l", 4.3))
        proteinuria = str(patient_profile.get("proteinuria", "Normal")).lower() in ["trace", "positive"]

        guideline_score = 0
        reasons = []
        cautions = []

        if drug_class in ["ACE_Inhibitor", "ARB"]:
            if dm or proteinuria:
                guideline_score += 3
                reasons.append("Guideline Strong Indication: Renoprotective and reduces microalbuminuria in Diabetes/Proteinuria.")
            if ckd and k_val < 5.0:
                guideline_score += 2
                reasons.append("Indicated in CKD for renal hemodynamics optimization.")
            if k_val >= 5.2:
                guideline_score -= 4
                cautions.append("Caution: Pre-existing hyperkalemia (K >= 5.2 mEq/L); monitor electrolytes closely.")

        elif drug_class == "CCB":
            if age >= 65:
                guideline_score += 3
                reasons.append("Guideline Preferred: Highly effective in older adults and isolated systolic hypertension.")
            if (sbp - dbp) >= 60:
                guideline_score += 2
                reasons.append("Effective in wide pulse pressure and vascular stiffness.")

        elif drug_class == "Thiazide_Diuretic":
            if age >= 60 or float(patient_profile.get("dietary_sodium_mg", 3000)) > 3500:
                guideline_score += 2
                reasons.append("Effective for volume-dependent and salt-sensitive hypertension.")
            if ckd:
                guideline_score -= 1
                cautions.append("Note: Thiazide monotherapy may exhibit diminished natriuretic efficacy in moderate-to-severe CKD.")

        elif drug_class == "Beta_Blocker":
            if cad:
                guideline_score += 4
                reasons.append("Guideline Class I Indication: Compelling indication in Coronary Artery Disease.")
            if hr >= 82:
                guideline_score += 2
                reasons.append("Favorable for sympathetic hyperactivity / elevated resting heart rate.")
            if age >= 65 and not cad:
                guideline_score -= 1
                cautions.append("Generally not first-line monotherapy in uncomplicated elderly hypertension.")

        return {
            "guideline_score": guideline_score,
            "clinical_reasons": reasons,
            "clinical_cautions": cautions
        }

    def recommend_best_regimen(
        self,
        patient_profile: Dict[str, Any],
        tested_dosage: str = "Standard",
        tested_combination: str = "Monotherapy"
    ) -> List[Dict[str, Any]]:
        """
        Simulates all candidate drug classes for the patient and ranks them.
        """
        results = []
        base_sbp = float(patient_profile.get("baseline_sbp", 150.0))
        base_dbp = float(patient_profile.get("baseline_dbp", 90.0))

        for drug in DRUG_CLASSES:
            # Create counterfactual row
            row = patient_profile.copy()
            row["drug_class"] = drug
            row["dosage_level"] = tested_dosage
            row["combination_therapy"] = tested_combination
            
            df_row = pd.DataFrame([row])
            df_eng = engineer_features(df_row)
            
            # Drop excluded columns if present
            feature_cols = [c for c in df_eng.columns if c not in [
                "patient_id", "post_treatment_sbp", "post_treatment_dbp",
                "delta_sbp", "delta_dbp", "treatment_response"
            ]]
            df_features = df_eng[feature_cols]

            # Transform features
            X_proc = self.preprocessor.transform(df_features)

            # Predict response probability
            prob_success = float(self.classifier.predict_proba(X_proc)[0, 1])
            
            # Predict BP drop
            pred_delta_sbp = float(self.regressor_sbp.predict(X_proc)[0])
            pred_delta_dbp = float(self.regressor_dbp.predict(X_proc)[0])

            est_post_sbp = max(80.0, round(base_sbp - pred_delta_sbp, 1))
            est_post_dbp = max(50.0, round(base_dbp - pred_delta_dbp, 1))

            guidelines = self.evaluate_clinical_guidelines(patient_profile, drug)

            # Composite Clinical Suitability Score: Prob * 70 + DeltaSBP * 1.5 + GuidelineScore * 5
            composite_score = round(prob_success * 70.0 + (pred_delta_sbp * 1.2) + (guidelines["guideline_score"] * 4.0), 2)

            results.append({
                "drug_class": drug,
                "response_probability_pct": round(prob_success * 100.0, 1),
                "predicted_delta_sbp_mmhg": round(pred_delta_sbp, 1),
                "predicted_delta_dbp_mmhg": round(pred_delta_dbp, 1),
                "estimated_post_treatment_bp": f"{est_post_sbp:.0f} / {est_post_dbp:.0f} mmHg",
                "composite_suitability_score": composite_score,
                "clinical_reasons": guidelines["clinical_reasons"],
                "clinical_cautions": guidelines["clinical_cautions"]
            })

        # Rank by composite suitability
        results.sort(key=lambda x: x["composite_suitability_score"], reverse=True)
        return results
