"""
Interactive Clinical Decision Support System (CDSS) for Hypertension Medication Response
Built with Streamlit for Master's Thesis Practical Implementation.
"""

import sys
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import DRUG_CLASSES, MODELS_DIR, FIGURES_DIR, REPORTS_DIR
from src.feature_engineering import engineer_features
from src.recommendation.treatment_recommender import TreatmentRecommender

# Page Configuration
st.set_page_config(
    page_title="Hypertension AI - Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background: #F8FAFC; border-radius: 10px; padding: 1.2rem; border-left: 5px solid #2563EB; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .status-badge-success { background: #DCFCE7; color: #166534; font-weight: 600; padding: 6px 14px; border-radius: 20px; font-size: 1rem; }
    .status-badge-fail { background: #FEE2E2; color: #991B1B; font-weight: 600; padding: 6px 14px; border-radius: 20px; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_framework_artifacts():
    """Loads preprocessor, trained models, and benchmark reports."""
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    classifier = joblib.load(MODELS_DIR / "best_classifier.joblib")
    regressor_sbp = joblib.load(MODELS_DIR / "best_regressor_sbp.joblib")
    regressor_dbp = joblib.load(MODELS_DIR / "best_regressor_dbp.joblib")
    
    with open(MODELS_DIR / "best_classifier_meta.json", "r") as f:
        best_meta = json.load(f)

    df_clf_bench = pd.read_csv(REPORTS_DIR / "classifier_benchmark.csv")
    df_reg_bench = pd.read_csv(REPORTS_DIR / "regressor_benchmark.csv")
    
    recommender = TreatmentRecommender(
        classifier=classifier,
        regressor_sbp=regressor_sbp,
        regressor_dbp=regressor_dbp,
        preprocessor=preprocessor
    )
    
    return preprocessor, classifier, regressor_sbp, regressor_dbp, best_meta, df_clf_bench, df_reg_bench, recommender

# Load artifacts
try:
    preprocessor, classifier, regressor_sbp, regressor_dbp, best_meta, df_clf_bench, df_reg_bench, recommender = load_framework_artifacts()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"Error loading model artifacts: {e}. Please ensure `python run_pipeline.py` has been executed.")

# App Header
st.markdown("<div class='main-header'>🩺 Intelligent Clinical Decision Support System (CDSS)</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predicting Individual Patient Response to Hypertension Medications Using Clinical & Physiological Data | Master's Thesis Practical Framework</div>", unsafe_allow_html=True)

# Sidebar: Patient Profile Input
st.sidebar.header("📋 Patient Clinical Profile")

with st.sidebar.expander("👤 1. Demographics & Lifestyle", expanded=True):
    age = st.slider("Age (Years)", 30, 85, 58)
    gender = st.selectbox("Biological Sex", ["Male", "Female"], index=0)
    bmi = st.number_input("Body Mass Index (BMI, kg/m²)", 18.5, 48.0, 29.5, step=0.5)
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], index=0)
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"], index=1)
    activity = st.selectbox("Physical Activity", ["Sedentary", "Moderate", "Active"], index=0)
    sodium_mg = st.slider("Dietary Sodium (mg/day)", 1500, 6000, 3400, step=100)
    family_htn = st.selectbox("Family History of HTN/CVD", ["Yes", "No"], index=0)

with st.sidebar.expander("💓 2. Hemodynamics & Duration", expanded=True):
    sbp = st.slider("Baseline Systolic BP (SBP, mmHg)", 135, 215, 162)
    dbp = st.slider("Baseline Diastolic BP (DBP, mmHg)", 80, 125, 94)
    hr = st.slider("Resting Heart Rate (bpm)", 50, 115, 76)
    duration_years = st.number_input("Hypertension Duration (Years)", 0.5, 30.0, 6.0, step=0.5)

with st.sidebar.expander("🧪 3. Biochemical & Laboratory Markers", expanded=False):
    scr = st.number_input("Serum Creatinine (mg/dL)", 0.5, 3.2, 1.05, step=0.05)
    fpg = st.number_input("Fasting Plasma Glucose (mg/dL)", 70.0, 260.0, 110.0, step=5.0)
    hba1c = st.number_input("HbA1c (%)", 4.5, 13.0, 6.2, step=0.1)
    chol = st.number_input("Total Cholesterol (mg/dL)", 120.0, 340.0, 205.0, step=5.0)
    ldl = st.number_input("LDL-C (mg/dL)", 50.0, 240.0, 125.0, step=5.0)
    hdl = st.number_input("HDL-C (mg/dL)", 25.0, 95.0, 48.0, step=2.0)
    tg = st.number_input("Triglycerides (mg/dL)", 60.0, 500.0, 160.0, step=10.0)
    k_serum = st.number_input("Serum Potassium (mEq/L)", 3.2, 5.8, 4.3, step=0.1)
    na_serum = st.number_input("Serum Sodium (mEq/L)", 132.0, 148.0, 140.5, step=0.5)
    proteinuria = st.selectbox("Proteinuria / Albuminuria", ["Normal", "Trace", "Positive"], index=0)

with st.sidebar.expander("⚠️ 4. Comorbidities", expanded=False):
    t2dm = st.selectbox("Type 2 Diabetes Mellitus", ["Yes", "No"], index=0 if hba1c >= 6.5 else 1)
    ckd = st.selectbox("Chronic Kidney Disease (CKD)", ["Yes", "No"], index=0 if scr >= 1.4 else 1)
    cad = st.selectbox("Coronary Artery Disease (CAD)", ["Yes", "No"], index=1)
    dyslip = st.selectbox("Dyslipidemia", ["Yes", "No"], index=0 if ldl >= 130 else 1)
    lvh = st.selectbox("Left Ventricular Hypertrophy (LVH)", ["Yes", "No"], index=1)

with st.sidebar.expander("💊 5. Proposed Medication Regimen", expanded=True):
    drug_class = st.selectbox("Antihypertensive Drug Class", DRUG_CLASSES, index=0)
    dosage = st.selectbox("Dosage Strength", ["Standard", "High"], index=0)
    combination = st.selectbox("Regimen Type", ["Monotherapy", "Dual_Combination"], index=0)
    adherence = st.slider("Expected Adherence Rate", 0.50, 1.00, 0.90, step=0.05)

# Assemble Patient Profile Dict
patient_dict = {
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "smoking_status": smoking,
    "alcohol_intake": alcohol,
    "physical_activity": activity,
    "dietary_sodium_mg": sodium_mg,
    "family_history_htn": family_htn,
    "hypertension_duration_years": duration_years,
    "baseline_sbp": sbp,
    "baseline_dbp": dbp,
    "heart_rate": hr,
    "serum_creatinine_mg_dl": scr,
    "fasting_glucose_mg_dl": fpg,
    "hba1c_pct": hba1c,
    "total_cholesterol_mg_dl": chol,
    "ldl_mg_dl": ldl,
    "hdl_mg_dl": hdl,
    "triglycerides_mg_dl": tg,
    "serum_potassium_meq_l": k_serum,
    "serum_sodium_meq_l": na_serum,
    "proteinuria": proteinuria,
    "diabetes_t2": t2dm,
    "ckd": ckd,
    "cad": cad,
    "dyslipidemia": dyslip,
    "lvh": lvh,
    "drug_class": drug_class,
    "dosage_level": dosage,
    "combination_therapy": combination,
    "adherence_rate": adherence
}

# Main Application Tabs
tab_pred, tab_recom, tab_bench, tab_xai, tab_report = st.tabs([
    "🎯 Individual Prediction",
    "🩺 Comparative Drug Recommender",
    "📊 Model Benchmarking",
    "🔍 Explainable AI (XAI)",
    "📑 Clinical Consultation Summary"
])

if models_ready:
    # -----------------------------------------------------------------
    # TAB 1: Individual Prediction
    # -----------------------------------------------------------------
    with tab_pred:
        # Preprocess single patient
        df_single = pd.DataFrame([patient_dict])
        df_eng_single = engineer_features(df_single)
        
        feature_cols = [c for c in df_eng_single.columns if c not in [
            "patient_id", "post_treatment_sbp", "post_treatment_dbp",
            "delta_sbp", "delta_dbp", "treatment_response"
        ]]
        X_single_proc = preprocessor.transform(df_eng_single[feature_cols])

        # Inference
        prob_resp = float(classifier.predict_proba(X_single_proc)[0, 1])
        pred_delta_sbp = float(regressor_sbp.predict(X_single_proc)[0])
        pred_delta_dbp = float(regressor_dbp.predict(X_single_proc)[0])

        est_post_sbp = max(80.0, round(sbp - pred_delta_sbp, 1))
        est_post_dbp = max(50.0, round(dbp - pred_delta_dbp, 1))
        is_responsive = prob_resp >= 0.50
        reached_goal = (est_post_sbp < 130) and (est_post_dbp < 80)

        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Probability", f"{prob_resp * 100:.1f}%",
                      delta="Favorable" if is_responsive else "High Risk of Failure",
                      delta_color="normal" if is_responsive else "inverse")
        with col2:
            st.metric("Predicted ΔSBP Drop", f"-{pred_delta_sbp:.1f} mmHg", delta=f"Goal: <130 mmHg")
        with col3:
            st.metric("Predicted ΔDBP Drop", f"-{pred_delta_dbp:.1f} mmHg", delta=f"Goal: <80 mmHg")
        with col4:
            st.metric("Projected Blood Pressure", f"{est_post_sbp:.0f} / {est_post_dbp:.0f} mmHg",
                      delta="ACC/AHA Goal Met" if reached_goal else "Above Goal Target",
                      delta_color="normal" if reached_goal else "inverse")

        st.markdown("---")

        # Visual Probability Gauge and BP Comparison
        g_col1, g_col2 = st.columns([1, 1])

        with g_col1:
            st.subheader("🎯 Treatment Response Likelihood")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_resp * 100.0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Response Probability ({drug_class})"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2563EB"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FEE2E2"},
                        {'range': [50, 75], 'color': "#FEF3C7"},
                        {'range': [75, 100], 'color': "#DCFCE7"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 70.0
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g_col2:
            st.subheader("📉 Blood Pressure Trajectory (Baseline vs. Predicted)")
            fig_bar = go.Figure(data=[
                go.Bar(name='Baseline BP', x=['Systolic (SBP)', 'Diastolic (DBP)'], y=[sbp, dbp], marker_color='#EF4444'),
                go.Bar(name='Estimated Post-Treatment', x=['Systolic (SBP)', 'Diastolic (DBP)'], y=[est_post_sbp, est_post_dbp], marker_color='#10B981'),
                go.Bar(name='ACC/AHA Target Goal', x=['Systolic (SBP)', 'Diastolic (DBP)'], y=[130, 80], marker_color='#93C5FD')
            ])
            fig_bar.update_layout(barmode='group', height=280, yaxis_title="Pressure (mmHg)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 2: Comparative Drug Recommender
    # -----------------------------------------------------------------
    with tab_recom:
        st.subheader("🩺 Personalized Multidrug Simulation & Ranking Engine")
        st.write("Simulates patient physiological response across all 5 major antihypertensive classes to support evidence-based precision selection.")
        
        recs = recommender.recommend_best_regimen(patient_dict, tested_dosage=dosage, tested_combination=combination)
        df_recs = pd.DataFrame(recs)

        # Comparative Chart
        fig_recom = px.bar(
            df_recs,
            x="drug_class",
            y="response_probability_pct",
            color="composite_suitability_score",
            color_continuous_scale="Viridis",
            labels={"drug_class": "Drug Class", "response_probability_pct": "Response Probability (%)", "composite_suitability_score": "Clinical Suitability Score"},
            title="Comparison of Predicted Success Rates Across Antihypertensive Classes"
        )
        fig_recom.update_layout(height=350)
        st.plotly_chart(fig_recom, use_container_width=True)

        # Ranked Table
        st.markdown("### 🏆 Clinical Ranking & Practice Guideline Rationale")
        for idx, r in enumerate(recs, 1):
            with st.expander(f"**Rank #{idx}: {r['drug_class']}** — Success Probability: **{r['response_probability_pct']}%** | Expected BP: **{r['estimated_post_treatment_bp']}** (Suitability: {r['composite_suitability_score']})", expanded=(idx==1)):
                st.write(f"- **Predicted SBP Reduction:** -{r['predicted_delta_sbp_mmhg']} mmHg")
                st.write(f"- **Predicted DBP Reduction:** -{r['predicted_delta_dbp_mmhg']} mmHg")
                if r["clinical_reasons"]:
                    st.markdown("**Guideline Indications:**")
                    for reason in r["clinical_reasons"]:
                        st.markdown(f"  * ✅ {reason}")
                if r["clinical_cautions"]:
                    st.markdown("**Prescribing Considerations / Cautions:**")
                    for caution in r["clinical_cautions"]:
                        st.markdown(f"  * ⚠️ {caution}")

    # -----------------------------------------------------------------
    # TAB 3: Model Benchmarking
    # -----------------------------------------------------------------
    with tab_bench:
        st.subheader("📊 Model Performance & Comparative Evaluation")
        st.write(f"Empirical evaluation results across candidate machine learning algorithms on the 5,000 patient clinical benchmark cohort.")

        st.markdown("#### 1. Classification Benchmarking (Treatment Response)")
        st.dataframe(df_clf_bench.style.highlight_max(subset=["Test_ROC_AUC", "Test_Accuracy", "Test_F1_Score"], color="#DCFCE7"), use_container_width=True)

        st.markdown("#### 2. Continuous Blood Pressure Drop Regression Benchmarking")
        st.dataframe(df_reg_bench.style.highlight_min(subset=["SBP_RMSE_mmHg", "SBP_MAE_mmHg", "DBP_RMSE_mmHg"], color="#DCFCE7"), use_container_width=True)

        st.markdown("#### 3. Diagnostic Figures")
        fig_c1, fig_c2 = st.columns(2)
        with fig_c1:
            if (FIGURES_DIR / "roc_curves_comparison.png").exists():
                st.image(str(FIGURES_DIR / "roc_curves_comparison.png"), caption="ROC Curves Comparison (Holdout Test Set)", use_container_width=True)
            if (FIGURES_DIR / "confusion_matrix.png").exists():
                st.image(str(FIGURES_DIR / "confusion_matrix.png"), caption="Confusion Matrix (Normalized & Counts)", use_container_width=True)
        with fig_c2:
            if (FIGURES_DIR / "precision_recall_comparison.png").exists():
                st.image(str(FIGURES_DIR / "precision_recall_comparison.png"), caption="Precision-Recall Curves Comparison", use_container_width=True)
            if (FIGURES_DIR / "calibration_reliability.png").exists():
                st.image(str(FIGURES_DIR / "calibration_reliability.png"), caption="Calibration Reliability Curve", use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 4: Explainable AI (XAI)
    # -----------------------------------------------------------------
    with tab_xai:
        st.subheader("🔍 Explainable AI (XAI) Analysis via SHAP")
        st.write("SHAP (SHapley Additive exPlanations) decomposes model predictions into transparent, clinically auditable feature attributions.")

        xai_col1, xai_col2 = st.columns(2)
        with xai_col1:
            if (FIGURES_DIR / "shap_summary_beeswarm.png").exists():
                st.image(str(FIGURES_DIR / "shap_summary_beeswarm.png"), caption="SHAP Summary (Beeswarm) Plot: Global Directional Effects", use_container_width=True)
        with xai_col2:
            if (FIGURES_DIR / "shap_feature_importance_bar.png").exists():
                st.image(str(FIGURES_DIR / "shap_feature_importance_bar.png"), caption="SHAP Global Mean Absolute Importance", use_container_width=True)

        if (FIGURES_DIR / "shap_waterfall_patient_example.png").exists():
            st.image(str(FIGURES_DIR / "shap_waterfall_patient_example.png"), caption="Sample Patient Local Decision Attribution (Waterfall)", use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 5: Clinical Consultation Summary
    # -----------------------------------------------------------------
    with tab_report:
        st.subheader("📑 Automated Clinical Decision Consultation Report")
        top_rec = recs[0]
        report_md = f"""
### Hypertension Precision Medication Consultation Note
**Patient Age / Sex:** {age} years | {gender}  
**BMI:** {bmi} kg/m² | **Duration of HTN:** {duration_years} years  
**Baseline Blood Pressure:** {sbp} / {dbp} mmHg (Pulse Pressure: {sbp - dbp} mmHg, MAP: {round(dbp + (sbp-dbp)/3, 1)} mmHg)  
**Resting Heart Rate:** {hr} bpm | **Estimated eGFR:** {df_eng_single['egfr'].iloc[0]} mL/min/1.73m²  
**Comorbidities:** Diabetes (T2DM: {t2dm}), CKD: {ckd}, CAD: {cad}, Dyslipidemia: {dyslip}, Proteinuria: {proteinuria}  
**Estimated Adherence:** {int(adherence*100)}% | **Sodium Intake:** {sodium_mg} mg/day  

---
#### AI Decision Support Recommendation:
- **Recommended First-Line Drug Class:** **{top_rec['drug_class']}** ({dosage} Dose, {combination})
- **Predicted Response Probability:** **{top_rec['response_probability_pct']}%**
- **Projected Blood Pressure Drop:** **-{top_rec['predicted_delta_sbp_mmhg']} mmHg SBP / -{top_rec['predicted_delta_dbp_mmhg']} mmHg DBP**
- **Estimated Post-Treatment Blood Pressure:** **{top_rec['estimated_post_treatment_bp']}**
- **ACC/AHA Goal Attainment:** {'Achieved (<130/80 mmHg)' if (est_post_sbp < 130 and est_post_dbp < 80) else 'Requires Intensive Monitoring / Dual Regimen'}

#### Key Clinical Indications:
{chr(10).join(['- ' + r for r in top_rec['clinical_reasons']]) if top_rec['clinical_reasons'] else '- Standard first-line indication'}

#### Prescribing Considerations:
{chr(10).join(['- ' + c for c in top_rec['clinical_cautions']]) if top_rec['clinical_cautions'] else '- No acute contraindications identified'}
"""
        st.markdown(report_md)
        st.download_button("📥 Download Consultation Note (.txt)", report_md, file_name=f"Patient_HTN_Consultation_Report.txt")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.85rem;'>Developed for Master's Thesis: Intelligent Machine Learning Framework for Predicting Patient Response to Hypertension Medications | ACC/AHA & ESH Aligned</p>", unsafe_allow_html=True)
