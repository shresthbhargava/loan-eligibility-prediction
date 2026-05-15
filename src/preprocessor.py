# src/preprocessor.py
# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing pipeline for Loan Eligibility Prediction.
#
# Handles ALL feature types explicitly:
#   numerical    → median impute → log1p → StandardScaler
#   categorical  → mode impute  → OneHotEncoder(drop='first')
#   passthrough  → median impute → pass through as-is
#   engineered   → same pipelines applied to domain-engineered features
#
# WHY a sklearn Pipeline?
#   Statistics (median, mean, categories) are learned ONLY on training data
#   during fit(), then applied identically to validation and production data
#   during transform(). This prevents data leakage — the most common
#   silent bug in ML projects.
#
# WHY use_engineered parameter?
#   Lets you compare model performance with and without engineered features.
#   You cannot claim feature engineering helps without this baseline.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute        import SimpleImputer

from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PASSTHROUGH_FEATURES,
    ENGINEERED_NUMERICAL,
    ENGINEERED_LOG,
    ENGINEERED_BINARY,
)


# ── Sub-pipeline builders ─────────────────────────────────────────────────────
# Each returns a fresh Pipeline instance.
# Fresh instances are required because sklearn Pipelines are stateful —
# fitting the same instance twice on different data causes silent errors.

def _numerical_pipeline() -> Pipeline:
    """
    For continuous skewed features: ApplicantIncome, CoapplicantIncome,
    LoanAmount, Loan_Amount_Term, and engineered ratio features.

    Steps:
      1. SimpleImputer(median)   — robust to outliers; mean would be pulled
                                   up by high-income outliers in this dataset
      2. log1p transform         — compresses right skew so Logistic Regression
                                   gets approximately normal inputs;
                                   tree models are unaffected but not harmed
      3. StandardScaler          — zero mean, unit variance; required for
                                   Logistic Regression coefficient stability;
                                   harmless for tree-based models

    feature_names_out='one-to-one':
      Required for get_feature_names_out() compatibility in newer sklearn.
      Without it, pipeline.get_feature_names_out() raises AttributeError.
    """
    return Pipeline(steps=[
        ("imputer",       SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(
            np.log1p,
            validate=True,
            feature_names_out="one-to-one",
        )),
        ("scaler",        StandardScaler()),
    ])


def _categorical_pipeline() -> Pipeline:
    """
    For nominal string features: Gender, Married, Education,
    Self_Employed, Property_Area.

    Steps:
      1. SimpleImputer(most_frequent) — mode imputation; safe for MCAR
                                        missingness (form fields left blank)
      2. OneHotEncoder                — converts strings to binary columns

    OneHotEncoder options:
      handle_unknown='ignore' — if a category unseen during training appears
                                at inference, encode it as all zeros instead
                                of crashing. Critical for production.
      drop='first'            — drops one dummy per feature to avoid the
                                dummy variable trap (perfect multicollinearity).
                                Matters for Logistic Regression; harmless
                                for tree-based models.
      sparse_output=False     — returns a dense numpy array; required for
                                compatibility with downstream sklearn steps.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )),
    ])


def _passthrough_pipeline() -> Pipeline:
    """
    For already-numeric ordinal/binary features:
      Credit_History : -1 (unknown MNAR) / 0 (defaulted) / 1 (clean)
      Dependents     :  0 / 1 / 2 / 3   (ordinal integer after mapping)

    WHY no log transform or scaling?
      Both features have a small integer range (-1 to 3).
      Log transform would distort the ordinal meaning of -1.
      Scaling provides minimal benefit for tree models.
      For Logistic Regression the impact is minor given the small range.

    WHY median imputation?
      Dependents has 8 NaN values (confirmed in Phase 2 diagnostics).
      Credit_History NaN was already filled with -1 in data_loader,
      but median impute here acts as a safety net.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])


def _log_scaled_pipeline() -> Pipeline:
    """
    For features that are ALREADY log-transformed (LoanAmountLog,
    TotalIncomeLog). Applying log1p a second time would be wrong.
    These only need imputation and scaling.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ── Main preprocessor builder ─────────────────────────────────────────────────

def build_preprocessor(use_engineered: bool = True) -> ColumnTransformer:
    """
    Assembles the full ColumnTransformer from sub-pipelines.

    Parameters
    ----------
    use_engineered : bool, default True
        If True, includes domain-engineered features in the pipeline.
        If False, trains on base features only — used to prove that
        feature engineering adds measurable value via comparison.

    Feature groups
    ──────────────
    Base features (always included):
      numerical    → ApplicantIncome, CoapplicantIncome,
                     LoanAmount, Loan_Amount_Term
      categorical  → Gender, Married, Education,
                     Self_Employed, Property_Area
      passthrough  → Credit_History, Dependents

    Engineered features (when use_engineered=True):
      eng_num      → TotalIncome, LoanAmountPerIncome,
                     IncomePerDependent, EMI_to_Income_ratio
                     (continuous ratios → log + scale)
      eng_log      → LoanAmountLog, TotalIncomeLog
                     (already log-transformed → scale only)
      eng_bin      → IsHighIncome, IsSelfEmployedHighLoan,
                     HasCoapplicant
                     (binary flags → passthrough)

    remainder='drop':
      Any column not explicitly listed is dropped.
      This ensures Loan_ID is dropped automatically.
      It also means if you add a column to the dataframe but forget
      to add it to config.py, it will silently be excluded —
      always verify with get_feature_names_out() after fitting.

    Usage
    ─────
    >>> pre = build_preprocessor(use_engineered=True)
    >>> X_train_proc = pre.fit_transform(X_train)   # learns statistics
    >>> X_val_proc   = pre.transform(X_val)          # applies learned stats
    >>> pre.get_feature_names_out()                  # verify all features present
    """

    # Always-present base transformers
    transformers = [
        ("num",  _numerical_pipeline(),   NUMERICAL_FEATURES),
        ("cat",  _categorical_pipeline(), CATEGORICAL_FEATURES),
        ("pass", _passthrough_pipeline(), PASSTHROUGH_FEATURES),
    ]

    # Optional engineered feature transformers
    if use_engineered:
        transformers += [
            # Continuous ratio features — same treatment as base numericals
            ("eng_num", _numerical_pipeline(),   ENGINEERED_NUMERICAL),

            # Already log-transformed — skip the log step, just scale
            ("eng_log", _log_scaled_pipeline(),  ENGINEERED_LOG),

            # Binary 0/1 flags — no transformation needed
            ("eng_bin", _passthrough_pipeline(), ENGINEERED_BINARY),
        ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",    # Loan_ID and any unlisted columns are dropped
    )