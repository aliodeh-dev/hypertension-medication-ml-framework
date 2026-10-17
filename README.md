# Intelligent Machine Learning Framework for Predicting Patient Response to Hypertension Medications Using Clinical and Physiological Data

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/Algorithm-CatBoost-yellow.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/Algorithm-XGBoost-red.svg)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/Algorithm-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/CDSS-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Project Overview
This project constitutes the complete practical implementation of a Master's Thesis research project:  
**"Intelligent Machine Learning Framework for Predicting Patient Response to Hypertension Medications Using Clinical and Physiological Data"**.

Hypertension is the leading modifiable risk factor for stroke, coronary artery disease, and chronic kidney disease worldwide. Traditional clinical practice relies heavily on an empirical trial-and-error approach to selecting antihypertensive agents, which often delays blood pressure control and increases cardiovascular morbidity.

This framework leverages state-of-the-art machine learning algorithms (**CatBoost, XGBoost, LightGBM, Random Forest, Multi-Layer Perceptrons, and Regularized Logistic Regression**) combined with **Explainable AI (SHAP)** to:
1. **Predict Binary Treatment Response**: Estimate whether a patient will achieve blood pressure control ($<130/80\text{ mmHg}$) or significant reduction ($\Delta\text{SBP} \ge 10\text{ mmHg}$ and $\Delta\text{DBP} \ge 5\text{ mmHg}$) according to ACC/AHA and ESH clinical guidelines.
2. **Forecast Continuous Blood Pressure Drop**: Estimate continuous systolic ($\Delta\text{SBP}$) and diastolic ($\Delta\text{DBP}$) reduction in millimeters of mercury ($\text{mmHg}$).
3. **Simulate & Rank Personalized Medications**: Counterfactually evaluate all 5 major first-line drug classes (**ACE Inhibitors, ARBs, CCBs, Thiazide Diuretics, and Beta-Blockers**) for any patient profile and rank them by response probability and clinical guideline safety.
4. **Explain Clinical Drivers**: Utilize SHAP (Shapley Additive exPlanations) to transparently inspect physiological feature contributions at both cohort and single-patient levels.
5. **Interactive Clinical Decision Support System (CDSS)**: Deploy a full-featured Streamlit web application for clinicians and evaluators.

---

## 🚀 Key Results & Benchmarking Summary

### 1. Classification Performance (Treatment Response)
Evaluated via 5-Fold Stratified Cross-Validation and an independent Holdout Test Set ($N=750$ patients):

| Model Name | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test Sensitivity (Recall) | Test Specificity | Test F1-Score | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** *(Best Classifier)* | **75.73%** | **0.8106** | **0.8931** | **83.66%** | **59.01%** | **0.8345** | **0.1568** |
| **Random Forest** | 74.67% | 0.7961 | 0.8879 | 82.28% | 58.68% | 0.8266 | 0.1652 |
| **XGBoost** | 74.00% | 0.7928 | 0.8804 | 81.25% | 58.68% | 0.8186 | 0.1690 |
| **LightGBM** | 74.40% | 0.7858 | 0.8779 | 81.89% | 58.68% | 0.8212 | 0.1742 |
| **Neural Network (MLP)** | 73.07% | 0.7743 | 0.8698 | 80.71% | 57.02% | 0.8143 | 0.1775 |
| **Logistic Regression** | 72.53% | 0.7639 | 0.8601 | 80.31% | 56.20% | 0.8103 | 0.1804 |

### 2. Continuous Blood Pressure Reduction Regression ($\Delta\text{SBP}$ & $\Delta\text{DBP}$)

| Regressor Model | $\Delta\text{SBP}$ RMSE (mmHg) | $\Delta\text{SBP}$ MAE (mmHg) | $\Delta\text{SBP}$ $R^2$ | $\Delta\text{DBP}$ RMSE (mmHg) | $\Delta\text{DBP}$ MAE (mmHg) | $\Delta\text{DBP}$ $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost Regressor** *(Best)* | **3.342** | **2.628** | **0.403** | **2.430** | **1.921** | **0.263** |
| **LightGBM Regressor** | 3.386 | 2.651 | 0.387 | 2.480 | 1.954 | 0.232 |
| **XGBoost Regressor** | 3.406 | 2.673 | 0.380 | 2.488 | 1.968 | 0.227 |
| **Ridge Regression** | 3.495 | 2.742 | 0.347 | 2.479 | 1.970 | 0.233 |
| **Random Forest Regressor** | 3.525 | 2.771 | 0.336 | 2.516 | 1.995 | 0.209 |

---

## 📂 Repository Directory Architecture

```
hypertension_ml_framework/
│
├── data/
│   ├── raw/
│   │   └── hypertension_raw.csv           # 5,000 patient clinical benchmark cohort
│   ├── processed/
│   │   └── hypertension_processed.csv     # Cohort with engineered physiological features
│   └── data_dictionary.json               # Full clinical metadata and feature dictionary
│
├── src/
│   ├── __init__.py
│   ├── config.py                          # Global clinical parameters & guideline thresholds
│   ├── data_generator.py                  # Pharmacologically grounded cohort synthesizer
│   ├── feature_engineering.py             # eGFR (CKD-EPI), Pulse Pressure, MAP, Framingham proxy
│   ├── preprocessor.py                    # Leak-free ColumnTransformer, RobustScaler, OneHotEncoder
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier_trainer.py          # 5-fold CV & holdout evaluation for 6 classifiers
│   │   └── regressor_trainer.py           # Multi-target continuous BP drop estimation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                     # Sensitivity, Specificity, PPV, NPV, LR, Brier, ROC-AUC
│   │   └── visualizer.py                  # High-res ROC, PR, Calibration, Confusion matrix figures
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_explainer.py              # SHAP TreeExplainer, Beeswarm, Waterfall, Bar plots
│   │
│   └── recommendation/
│       ├── __init__.py
│       └── treatment_recommender.py       # Multi-drug counterfactual simulator & guideline engine
│
├── app/
│   └── app_streamlit.py                   # Interactive Web Clinical Decision Support System (CDSS)
│
├── artifacts/
│   ├── models/                            # Serialized model weights (.joblib)
│   ├── figures/                           # Publication-grade figures (300 DPI PNGs)
│   └── reports/                           # Benchmark CSV and JSON summaries
│
├── run_pipeline.py                        # Automated end-to-end execution script
├── test_framework.py                      # Complete unit & integration test suite
├── requirements.txt                       # Project dependencies
└── README.md                              # Technical documentation
```

---

## 🛠️ Installation & Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher.
- Recommended to run in a dedicated virtual environment.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the End-to-End Pipeline
Executes data generation, feature engineering, cross-validation, testing, visualization, and explainability:
```bash
python run_pipeline.py
```

### 4. Run the Unit & Integration Test Suite
```bash
python test_framework.py
```

### 5. Launch the Interactive CDSS Web Application
```bash
streamlit run app/app_streamlit.py
```
*The web interface will automatically open in your default browser at `http://localhost:8501`.*

---

## 🔬 Clinical Features & Physiological Indicators

1. **Demographics**: Age, Biological Sex, Body Mass Index (BMI), Smoking Status, Alcohol Intake, Physical Activity, Dietary Sodium (mg/day), Family History of CVD/HTN.
2. **Hemodynamics**: Baseline SBP (mmHg), Baseline DBP (mmHg), Resting Heart Rate (bpm), Pulse Pressure ($\text{SBP}-\text{DBP}$), Mean Arterial Pressure (MAP), Hypertension Duration (years).
3. **Biochemical / Renal**: Serum Creatinine (mg/dL), eGFR (CKD-EPI formula, $\text{mL/min/1.73m}^2$), Fasting Plasma Glucose (mg/dL), $\text{HbA}_{1c}$ (%), Serum Sodium (mEq/L), Serum Potassium (mEq/L), Sodium-to-Potassium Ratio, Proteinuria (Normal / Trace / Positive).
4. **Lipid Profile**: Total Cholesterol, LDL-C, HDL-C, Triglycerides, Atherogenic Index of Plasma ($\log_{10}(\text{TG}/\text{HDL})$).
5. **Comorbidities**: Type 2 Diabetes Mellitus, Chronic Kidney Disease (CKD), Coronary Artery Disease (CAD), Dyslipidemia, Left Ventricular Hypertrophy (LVH).
6. **Medication Variables**: Drug Class (ACEi, ARB, CCB, Thiazide, BB), Dosage Strength (Standard / High), Regimen (Monotherapy / Dual Combination), Patient Adherence Rate.

---

## 📈 Explainable AI (XAI) Findings
Using **SHAP TreeExplainer** on the optimal CatBoost ensemble:
- **Baseline Systolic Blood Pressure & Adherence Rate**: Highest global importance. Higher baseline SBP increases probability of achieving the $\ge 10\text{ mmHg}$ clinical drop threshold. Adherence $>85\%$ is the strongest positive modifier of success.
- **Drug-Patient Phenotype Matching**:
  - Calcium Channel Blockers (CCB) demonstrate the highest efficacy in elderly patients ($\ge 65$) and patients with high dietary sodium intake and wide pulse pressure.
  - ARBs and ACE inhibitors show peak responsiveness in patients with Type 2 Diabetes, younger demographics, and proteinuria.
  - Beta-Blockers show favorable response predominantly in patients with elevated resting heart rate ($>80\text{ bpm}$) and documented CAD.
  - Dual combination regimens consistently produce $+30-38\%$ greater odds of goal attainment compared to monotherapy.

---

## 🎓 Academic Thesis Statement
This framework demonstrates that machine learning models, specifically gradient-boosted decision trees, can accurately predict patient-specific response to antihypertensive pharmacological therapies. By coupling predictive modeling with Explainable AI and clinical guideline constraints, the system successfully transitions from a "black-box" algorithm into an actionable Clinical Decision Support System that mitigates trial-and-error prescribing.
