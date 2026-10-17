"""
Unit and Integration Test Suite for Hypertension ML Framework
Verifies data generation, feature engineering, preprocessing, model inference, and recommendation.
"""

import unittest
import numpy as np
import pandas as pd
import joblib

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, DRUG_CLASSES
)
from src.data_generator import generate_hypertension_cohort
from src.feature_engineering import engineer_features
from src.preprocessor import ClinicalDataPreprocessor
from src.recommendation.treatment_recommender import TreatmentRecommender

class TestHypertensionFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df, cls.meta = generate_hypertension_cohort(n_samples=200, seed=123)

    def test_01_cohort_generation_bounds(self):
        """Verifies physiological realism and value bounds in synthesized cohort."""
        self.assertEqual(len(self.raw_df), 200)
        self.assertTrue(self.raw_df["baseline_sbp"].min() >= 130)
        self.assertTrue(self.raw_df["baseline_sbp"].max() <= 220)
        self.assertTrue(self.raw_df["baseline_dbp"].min() >= 75)
        self.assertTrue(self.raw_df["baseline_dbp"].max() <= 130)
        self.assertTrue(self.raw_df["age"].min() >= 25)
        self.assertTrue(self.raw_df["age"].max() <= 90)
        self.assertTrue(set(self.raw_df["drug_class"].unique()).issubset(set(DRUG_CLASSES)))

    def test_02_feature_engineering_integrity(self):
        """Verifies computed hemodynamic and biochemical indices."""
        df_eng = engineer_features(self.raw_df)
        self.assertIn("pulse_pressure", df_eng.columns)
        self.assertIn("mean_arterial_pressure", df_eng.columns)
        self.assertIn("egfr", df_eng.columns)
        self.assertIn("framingham_risk_proxy", df_eng.columns)

        # Pulse pressure = SBP - DBP
        expected_pp = (df_eng["baseline_sbp"] - df_eng["baseline_dbp"]).round(1)
        np.testing.assert_allclose(df_eng["pulse_pressure"].values, expected_pp.values, rtol=1e-3)

        # eGFR should be positive and bounded
        self.assertTrue(df_eng["egfr"].min() > 5.0)
        self.assertTrue(df_eng["egfr"].max() < 160.0)

    def test_03_preprocessor_pipeline(self):
        """Verifies preprocessing transformation and leak-free encoding."""
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        df_eng = engineer_features(self.raw_df)
        feature_cols = [c for c in df_eng.columns if c not in [
            "patient_id", "post_treatment_sbp", "post_treatment_dbp",
            "delta_sbp", "delta_dbp", "treatment_response"
        ]]
        X_proc = preprocessor.transform(df_eng[feature_cols])
        self.assertEqual(X_proc.shape[0], 200)
        self.assertFalse(X_proc.isnull().values.any(), "Transformed array contains NaN values!")

    def test_04_model_artifacts_exist(self):
        """Verifies all required models and reports were successfully trained and persisted."""
        self.assertTrue((MODELS_DIR / "best_classifier.joblib").exists())
        self.assertTrue((MODELS_DIR / "best_regressor_sbp.joblib").exists())
        self.assertTrue((MODELS_DIR / "best_regressor_dbp.joblib").exists())
        self.assertTrue((REPORTS_DIR / "classifier_benchmark.csv").exists())
        self.assertTrue((REPORTS_DIR / "regressor_benchmark.csv").exists())
        self.assertTrue((FIGURES_DIR / "roc_curves_comparison.png").exists())
        self.assertTrue((FIGURES_DIR / "shap_summary_beeswarm.png").exists())

    def test_05_treatment_recommender(self):
        """Verifies multi-class drug simulation and ranking engine."""
        recommender = TreatmentRecommender()
        sample_patient = {
            "age": 70,
            "gender": "Male",
            "bmi": 28.0,
            "smoking_status": "Never",
            "alcohol_intake": "None",
            "physical_activity": "Moderate",
            "dietary_sodium_mg": 3200,
            "family_history_htn": "Yes",
            "hypertension_duration_years": 8.0,
            "baseline_sbp": 168.0,
            "baseline_dbp": 88.0,
            "heart_rate": 72,
            "serum_creatinine_mg_dl": 1.1,
            "fasting_glucose_mg_dl": 105.0,
            "hba1c_pct": 5.8,
            "total_cholesterol_mg_dl": 195.0,
            "ldl_mg_dl": 115.0,
            "hdl_mg_dl": 50.0,
            "triglycerides_mg_dl": 150.0,
            "serum_potassium_meq_l": 4.4,
            "serum_sodium_meq_l": 141.0,
            "proteinuria": "Normal",
            "diabetes_t2": "No",
            "ckd": "No",
            "cad": "No",
            "dyslipidemia": "No",
            "lvh": "No",
            "adherence_rate": 0.95
        }
        recs = recommender.recommend_best_regimen(sample_patient)
        self.assertEqual(len(recs), 5)
        # Check all drug classes present
        recommended_classes = [r["drug_class"] for r in recs]
        self.assertEqual(set(recommended_classes), set(DRUG_CLASSES))
        # Top recommendation should have high probability
        self.assertTrue(recs[0]["response_probability_pct"] > 50.0)

if __name__ == "__main__":
    unittest.main()
