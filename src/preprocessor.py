# src/preprocessor.py
# ─────────────────────────────────────────────────────────────────────────────
# Builds a reusable sklearn Pipeline for preprocessing.
# WHY a Pipeline? It ensures that imputation statistics (medians, modes) are
# learned ONLY on training data, then applied identically to validation and
# production data. This prevents data leakage and makes deployment trivial.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer that handles:
      - Numerical: median imputation → log transform → standard scaling
      - Categorical: mode imputation → one-hot encoding
    
    WHY median imputation for numerical? Median is robust to outliers.
    Mean imputation on right-skewed income data would impute unrealistically
    high values because the mean is pulled up by extreme earners.
    
    WHY log transform? ApplicantIncome and LoanAmount are right-skewed.
    Log compression makes the distribution approximately normal, which
    helps linear models (Logistic Regression) learn better coefficients.
    Tree-based models don't need this, but it doesn't hurt them either.
    
    WHY StandardScaler? After log transform, values are still on different
    scales. Scaling to mean=0 std=1 ensures Logistic Regression doesn't
    give disproportionate weight to larger-scale features.
    """
    
    # ── Numerical pipeline ───────────────────────────────────────────────────
    numerical_pipeline = Pipeline(steps=[
        # Step 1: Fill missing values with the median of the training set
        # strategy='median' is robust to outliers; 'mean' is not
        ("imputer", SimpleImputer(strategy="median")),
        
        # Step 2: Log-transform to reduce skew
        # np.log1p = log(x + 1), handles zero values safely (log(0) = -inf)
        ("log_transform", FunctionTransformer(np.log1p, validate=True)),
        
        # Step 3: Scale to mean=0, std=1
        # Required for Logistic Regression; harmless for tree models
        ("scaler", StandardScaler()),
    ])
    
    # ── Categorical pipeline ─────────────────────────────────────────────────
    categorical_pipeline = Pipeline(steps=[
        # Step 1: Fill missing with the most frequent value (mode)
        # For Gender/Married this is MCAR — mode imputation is safe
        ("imputer", SimpleImputer(strategy="most_frequent")),
        
        # Step 2: One-hot encode
        # handle_unknown='ignore': if a new category appears at inference
        # time that wasn't in training, don't crash — just set all dummies to 0
        # drop='first': drops one dummy per feature to avoid multicollinearity
        # (the "dummy variable trap" — relevant for Logistic Regression)
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    
    # ── Combine with ColumnTransformer ───────────────────────────────────────
    # remainder='drop' means any column not listed here is dropped
    # This is intentional — Loan_ID gets dropped automatically
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")
    
    return preprocessor