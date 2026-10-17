"""
Preprocessing Pipeline for Hypertension ML Framework
Provides leak-free encoding, scaling, imputing, and stratified train/val/test splitting.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_STATE, CLASSIFICATION_TARGET, REGRESSION_TARGET_SBP, REGRESSION_TARGET_DBP,
    MODELS_DIR, PROCESSED_DATA_PATH
)
from src.feature_engineering import engineer_features

# Targets and non-feature columns
EXCLUDE_COLUMNS = [
    "patient_id", "post_treatment_sbp", "post_treatment_dbp",
    "delta_sbp", "delta_dbp", "treatment_response"
]

class ClinicalDataPreprocessor:
    def __init__(self, use_robust_scaler: bool = True):
        self.use_robust_scaler = use_robust_scaler
        self.pipeline: ColumnTransformer = None
        self.feature_names: List[str] = []
        self.num_features: List[str] = []
        self.cat_features: List[str] = []

    def identify_feature_types(self, X: pd.DataFrame):
        self.num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    def build_pipeline(self) -> ColumnTransformer:
        scaler = RobustScaler() if self.use_robust_scaler else StandardScaler()
        
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler)
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.num_features),
                ("cat", cat_pipeline, self.cat_features)
            ],
            remainder="drop"
        )
        return transformer

    def fit(self, X: pd.DataFrame):
        self.identify_feature_types(X)
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X)

        # Extract encoded feature names
        ohe = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_encoded_names = list(ohe.get_feature_names_out(self.cat_features))
        self.feature_names = self.num_features + cat_encoded_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.pipeline.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, filepath: str = str(MODELS_DIR / "preprocessor.joblib")):
        joblib.dump(self, filepath)
        print(f"Preprocessor pipeline successfully saved to: {filepath}")

    @staticmethod
    def load(filepath: str = str(MODELS_DIR / "preprocessor.joblib")) -> "ClinicalDataPreprocessor":
        return joblib.load(filepath)

def prepare_splits(
    raw_df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Applies feature engineering, performs leak-free stratified splitting,
    fits the preprocessor only on train set, and returns transformed train, val, and test splits.
    """
    # 1. Feature Engineering
    df_eng = engineer_features(raw_df)
    
    # Save processed dataframe
    df_eng.to_csv(PROCESSED_DATA_PATH, index=False)

    # 2. Extract X and Targets
    feature_cols = [c for c in df_eng.columns if c not in EXCLUDE_COLUMNS]
    X = df_eng[feature_cols]
    y_clf = df_eng[CLASSIFICATION_TARGET]
    y_reg_sbp = df_eng[REGRESSION_TARGET_SBP]
    y_reg_dbp = df_eng[REGRESSION_TARGET_DBP]

    # Stratify by classification target
    strat_key = y_clf.astype(str) + "_" + df_eng["drug_class"].astype(str)

    # Train+Val vs Test split
    X_train_val, X_test, y_clf_train_val, y_clf_test, y_sbp_train_val, y_sbp_test, y_dbp_train_val, y_dbp_test, strat_train_val, _ = train_test_split(
        X, y_clf, y_reg_sbp, y_reg_dbp, strat_key,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key
    )

    # Train vs Val split
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_clf_train, y_clf_val, y_sbp_train, y_sbp_val, y_dbp_train, y_dbp_val = train_test_split(
        X_train_val, y_clf_train_val, y_sbp_train_val, y_dbp_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=strat_train_val
    )

    # 3. Fit Preprocessor ONLY on training set to avoid data leakage
    preprocessor = ClinicalDataPreprocessor(use_robust_scaler=True)
    preprocessor.fit(X_train)
    preprocessor.save()

    # 4. Transform all splits
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    return {
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names,
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "X_train": X_train_proc,
        "X_val": X_val_proc,
        "X_test": X_test_proc,
        "y_clf_train": y_clf_train,
        "y_clf_val": y_clf_val,
        "y_clf_test": y_clf_test,
        "y_sbp_train": y_sbp_train,
        "y_sbp_val": y_sbp_val,
        "y_sbp_test": y_sbp_test,
        "y_dbp_train": y_dbp_train,
        "y_dbp_val": y_dbp_val,
        "y_dbp_test": y_dbp_test,
    }

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    raw = pd.read_csv(RAW_DATA_PATH)
    splits = prepare_splits(raw)
    print("Preprocessing completed!")
    print(f"Train samples: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
    print(f"Number of encoded features: {len(splits['feature_names'])}")
    print("Sample feature names:", splits['feature_names'][:8])
