"""
Master Execution Pipeline for Intelligent Hypertension ML Framework
Executes end-to-end data generation, preprocessing, model training, cross-validation,
evaluation, publication figure generation, SHAP explainability, and recommendation simulation.
"""

import sys
import os
import time
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_DICTIONARY_PATH,
    MODELS_DIR, FIGURES_DIR, REPORTS_DIR, RANDOM_STATE
)
from src.data_generator import save_generated_dataset
from src.preprocessor import prepare_splits
from src.models.classifier_trainer import train_and_evaluate_classifiers
from src.models.regressor_trainer import train_and_evaluate_regressors
from src.evaluation.visualizer import (
    plot_roc_curves, plot_precision_recall_curves, plot_confusion_matrix_figure,
    plot_calibration_curves, plot_feature_importance_figure,
    plot_bp_reduction_distributions, plot_correlation_matrix,
    plot_regression_actual_vs_predicted
)
from src.explainability.shap_explainer import run_shap_analysis
from src.recommendation.treatment_recommender import TreatmentRecommender

def main():
    t_start = time.time()
    print("=" * 75)
    print("  INTELLIGENT MACHINE LEARNING FRAMEWORK FOR PREDICTING PATIENT RESPONSE")
    print("      TO HYPERTENSION MEDICATIONS USING CLINICAL & PHYSIOLOGICAL DATA")
    print("=" * 75)

    # -------------------------------------------------------------
    # STAGE 1: Data Preparation & Cohort Generation
    # -------------------------------------------------------------
    print("\n[STAGE 1/7] Checking Clinical Dataset...")
    if not RAW_DATA_PATH.exists():
        print("Raw dataset not found. Synthesizing clinically realistic 5,000 patient cohort...")
        raw_df, meta = save_generated_dataset(RAW_DATA_PATH, DATA_DICTIONARY_PATH)
    else:
        print(f"Loading existing clinical cohort from: {RAW_DATA_PATH}")
        raw_df = pd.read_csv(RAW_DATA_PATH)
        print(f"Loaded {len(raw_df)} patient records with {raw_df.shape[1]} features.")

    # -------------------------------------------------------------
    # STAGE 2: Exploratory Data Analysis Visualizations
    # -------------------------------------------------------------
    print("\n[STAGE 2/7] Generating Exploratory Clinical Visualizations...")
    plot_bp_reduction_distributions(raw_df, FIGURES_DIR / "bp_reduction_by_drug_class.png")
    plot_correlation_matrix(raw_df, FIGURES_DIR / "clinical_correlation_matrix.png")

    # -------------------------------------------------------------
    # STAGE 3: Preprocessing & Leak-Free Stratified Splitting
    # -------------------------------------------------------------
    print("\n[STAGE 3/7] Performing Feature Engineering & Leak-Free Splitting...")
    splits = prepare_splits(raw_df, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE)
    
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]
    y_clf_train = splits["y_clf_train"]
    y_clf_val = splits["y_clf_val"]
    y_clf_test = splits["y_clf_test"]
    y_sbp_train = splits["y_sbp_train"]
    y_sbp_test = splits["y_sbp_test"]
    y_dbp_train = splits["y_dbp_train"]
    y_dbp_test = splits["y_dbp_test"]
    feature_names = splits["feature_names"]

    print(f"Feature set size: {len(feature_names)} features after one-hot encoding.")
    print(f"Cohort splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # -------------------------------------------------------------
    # STAGE 4: Classifier Benchmarking (5-Fold CV + Test Set)
    # -------------------------------------------------------------
    print("\n[STAGE 4/7] Training & Cross-Validating Classification Models...")
    clf_models, df_clf_benchmark, best_clf_name = train_and_evaluate_classifiers(
        X_train, y_clf_train, X_val, y_clf_val, X_test, y_clf_test, cv_folds=5
    )

    # -------------------------------------------------------------
    # STAGE 5: Continuous Blood Pressure Drop Regressors
    # -------------------------------------------------------------
    print("\n[STAGE 5/7] Training Continuous Blood Pressure Drop Regressors...")
    sbp_models, dbp_models, df_reg_benchmark, best_reg_name = train_and_evaluate_regressors(
        X_train, y_sbp_train, y_dbp_train, X_test, y_sbp_test, y_dbp_test
    )

    # -------------------------------------------------------------
    # STAGE 6: Generating Publication-Grade Evaluation Plots
    # -------------------------------------------------------------
    print("\n[STAGE 6/7] Generating Publication-Quality Model Diagnostic Figures...")
    
    # ROC and PR Curves
    plot_roc_curves(clf_models, X_test, y_clf_test, FIGURES_DIR / "roc_curves_comparison.png")
    plot_precision_recall_curves(clf_models, X_test, y_clf_test, FIGURES_DIR / "precision_recall_comparison.png")
    plot_calibration_curves(clf_models, X_test, y_clf_test, FIGURES_DIR / "calibration_reliability.png")

    # Best Model Confusion Matrix
    best_clf_model = clf_models[best_clf_name]
    y_test_pred_best = best_clf_model.predict(X_test)
    plot_confusion_matrix_figure(
        y_clf_test.values, y_test_pred_best,
        model_name=best_clf_name,
        save_path=FIGURES_DIR / "confusion_matrix.png"
    )

    # Feature Importance for Tree Models
    if hasattr(best_clf_model, "feature_importances_"):
        plot_feature_importance_figure(
            feature_names, best_clf_model.feature_importances_,
            model_name=best_clf_name, top_n=15,
            save_path=FIGURES_DIR / "feature_importance.png"
        )

    # Regressor Actual vs Predicted (Best Model for SBP)
    best_reg_sbp = sbp_models[best_reg_name]
    y_sbp_pred_best = best_reg_sbp.predict(X_test)
    plot_regression_actual_vs_predicted(
        y_sbp_test.values, y_sbp_pred_best,
        target_label="Delta_SBP (mmHg)", model_name=best_reg_name,
        save_path=FIGURES_DIR / "regression_performance.png"
    )

    # -------------------------------------------------------------
    # STAGE 7: Explainable AI (XAI) with SHAP
    # -------------------------------------------------------------
    print("\n[STAGE 7/7] Computing SHAP Tree Explanations & Personalized Decision Decompositions...")
    shap_results = run_shap_analysis(best_clf_model, X_test, feature_names, max_display=15, sample_size=400)

    # -------------------------------------------------------------
    # DEMONSTRATION: Clinical Decision Support & Recommender Test
    # -------------------------------------------------------------
    print("\n--> Verifying Personalized Clinical Treatment Recommender with a Sample Patient Profile:")
    recommender = TreatmentRecommender(
        classifier=best_clf_model,
        regressor_sbp=best_reg_sbp,
        regressor_dbp=dbp_models[best_reg_name],
        preprocessor=splits["preprocessor"]
    )

    sample_patient = {
        "age": 63,
        "gender": "Female",
        "bmi": 31.4,
        "smoking_status": "Never",
        "alcohol_intake": "Moderate",
        "physical_activity": "Sedentary",
        "dietary_sodium_mg": 3800,
        "family_history_htn": "Yes",
        "hypertension_duration_years": 7.5,
        "baseline_sbp": 165.0,
        "baseline_dbp": 94.0,
        "heart_rate": 78,
        "serum_creatinine_mg_dl": 1.15,
        "fasting_glucose_mg_dl": 138.0,
        "hba1c_pct": 7.1,
        "total_cholesterol_mg_dl": 218.0,
        "ldl_mg_dl": 134.0,
        "hdl_mg_dl": 44.0,
        "triglycerides_mg_dl": 200.0,
        "serum_potassium_meq_l": 4.2,
        "serum_sodium_meq_l": 141.0,
        "proteinuria": "Positive",
        "diabetes_t2": "Yes",
        "ckd": "No",
        "cad": "No",
        "dyslipidemia": "Yes",
        "lvh": "No",
        "adherence_rate": 0.90
    }

    recommendations = recommender.recommend_best_regimen(sample_patient)
    print(f"\nPatient Profile: 63yo Female, T2DM + Proteinuria, Baseline BP: 165/94 mmHg, High Salt Intake")
    print("-" * 75)
    for rank, rec in enumerate(recommendations, 1):
        print(f"Rank #{rank}: {rec['drug_class']:<20} | Success Prob: {rec['response_probability_pct']}% | Pred Delta_SBP: -{rec['predicted_delta_sbp_mmhg']} mmHg | Est BP: {rec['estimated_post_treatment_bp']}")
        for r in rec["clinical_reasons"]:
            print(f"       + {r}")
        for c in rec["clinical_cautions"]:
            print(f"       ! {c}")

    elapsed = round(time.time() - t_start, 2)
    print("\n" + "=" * 75)
    print(f"  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {elapsed} SECONDS!")
    print(f"  Artifacts saved in:")
    print(f"    - Models:   {MODELS_DIR}")
    print(f"    - Figures:  {FIGURES_DIR}")
    print(f"    - Reports:  {REPORTS_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
