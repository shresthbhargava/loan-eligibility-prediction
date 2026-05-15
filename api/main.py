# api/main.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application for loan eligibility prediction.
#
# Three endpoints:
#   GET  /health       — liveness check for deployment platforms
#   GET  /model-info   — metadata about the loaded model
#   POST /predict      — main prediction endpoint
#
# Design decisions:
#   - Model loaded ONCE at startup (not per request) — critical for performance
#   - Pydantic validation runs BEFORE model inference — invalid inputs never
#     reach the model, preventing cryptic sklearn errors
#   - Threshold applied at API level — business logic stays in the API,
#     not buried in the model artifact
#   - SHAP computed on request — expensive but necessary for explainability
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logger import get_logger
from config import MODELS_DIR
from data_loader import (clean_dependents, handle_credit_history,
                          engineer_features)

logger = get_logger("api")

# ── Global state — loaded once at startup ────────────────────────────────────
# WHY global? Loading a 10MB joblib model takes ~200ms.
# If we loaded per request at 100 req/s, that's 20 seconds of loading per second.
# Loading once at startup means every request uses the same in-memory model.
MODEL      = None
MODEL_NAME = "final_model"
THRESHOLD  = 0.77    # cost-optimized threshold from Phase 6


# ── Startup / shutdown lifecycle ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, SHAP_EXPLAINER, FEATURE_NAMES

    logger.info("API starting — loading model...")
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"

    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    MODEL = joblib.load(model_path)
    logger.info("Model loaded: %s", model_path)

    # Warm up SHAP explainer at startup — prevents 2s first-request spike
    logger.info("Warming up SHAP explainer...")
    try:
        import shap
        import pandas as pd
        from data_loader import load_raw_data, clean_dependents
        from data_loader import handle_credit_history, encode_target
        from data_loader import engineer_features, get_X_y

        df_warm, _ = load_raw_data()
        df_warm = clean_dependents(df_warm)
        df_warm = handle_credit_history(df_warm)
        df_warm = encode_target(df_warm)
        df_warm = engineer_features(df_warm)
        X_warm, _ = get_X_y(df_warm)

        preprocessor = MODEL.named_steps['preprocessor']
        classifier   = MODEL.named_steps['classifier']
        X_proc       = preprocessor.transform(X_warm.head(50))

        SHAP_EXPLAINER = shap.TreeExplainer(classifier, check_additivity=False)
        # Dummy call to fully initialize internal structures
        _ = SHAP_EXPLAINER.shap_values(X_proc[:1])

        try:
            FEATURE_NAMES = [n.split('__')[-1]
                             for n in preprocessor.get_feature_names_out()]
        except Exception:
            FEATURE_NAMES = [f"f{i}" for i in range(X_proc.shape[1])]

        logger.info("SHAP explainer warmed up. All requests will be fast.")
    except Exception as e:
        logger.warning("SHAP warmup failed (non-critical): %s", e)
        SHAP_EXPLAINER = None
        FEATURE_NAMES  = []

    yield
    logger.info("API shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Loan Eligibility Prediction API",
    description = """
Predicts loan approval probability for applicants using a Random Forest
classifier trained on historical lending data.

Returns:
- Approval decision (Approved/Rejected)
- Approval probability
- Risk category
- Top SHAP feature contributions
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allows the Streamlit frontend (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production to specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class ApplicantInput(BaseModel):
    """
    Pydantic model for incoming applicant data.

    WHY Pydantic validation?
    Without it, if someone sends Gender="XYZ" or ApplicantIncome=-5000,
    the error surfaces deep inside sklearn with a cryptic message.
    Pydantic catches invalid inputs at the API boundary with clear messages.

    Field() provides:
      - description: shown in auto-generated /docs UI
      - ge/le: greater-than-equal / less-than-equal bounds
      - pattern: regex validation for categorical fields
    """
    Gender           : str   = Field(..., description="Male or Female",
                                      pattern="^(Male|Female)$")
    Married          : str   = Field(..., description="Yes or No",
                                      pattern="^(Yes|No)$")
    Dependents       : str   = Field(..., description="0, 1, 2, or 3+",
                                      pattern="^(0|1|2|3\+)$")
    Education        : str   = Field(..., description="Graduate or Not Graduate",
                                      pattern="^(Graduate|Not Graduate)$")
    Self_Employed    : str   = Field(..., description="Yes or No",
                                      pattern="^(Yes|No)$")
    ApplicantIncome  : float = Field(..., description="Monthly income in rupees",
                                      ge=0, le=1_000_000)
    CoapplicantIncome: float = Field(0,   description="Co-applicant monthly income",
                                      ge=0, le=1_000_000)
    LoanAmount       : float = Field(..., description="Loan amount in thousands",
                                      ge=1, le=10_000)
    Loan_Amount_Term : float = Field(360, description="Loan term in months",
                                      ge=12, le=480)
    Credit_History   : float = Field(..., description="1=clean, 0=default, -1=unknown",
                                      ge=-1, le=1)
    Property_Area    : str   = Field(..., description="Urban, Semiurban, or Rural",
                                      pattern="^(Urban|Semiurban|Rural)$")

    @field_validator('Credit_History')
    @classmethod
    def validate_credit_history(cls, v):
        """Credit_History must be exactly -1, 0, or 1."""
        if v not in (-1.0, 0.0, 1.0):
            raise ValueError("Credit_History must be -1, 0, or 1")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 2000,
                "LoanAmount": 150,
                "Loan_Amount_Term": 360,
                "Credit_History": 1,
                "Property_Area": "Urban"
            }
        }
    }


# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Structured response — every field documented for API consumers."""
    decision          : str          # "Approved" or "Rejected"
    approval_probability: float      # 0.0 to 1.0
    risk_category     : str          # "Low" / "Medium" / "High"
    threshold_used    : float        # business threshold applied
    top_factors       : list[dict]   # top SHAP feature contributions
    processing_time_ms: float        # latency for monitoring
    warnings             : list[str] = []


class HealthResponse(BaseModel):
    status     : str
    model_loaded: bool
    version    : str


class ModelInfoResponse(BaseModel):
    model_name  : str
    threshold   : float
    feature_count: int
    model_type  : str


# ── Helper functions ──────────────────────────────────────────────────────────
def get_risk_category(probability: float) -> str:
    """
    Converts approval probability to a business-readable risk category.
    These thresholds are business decisions — adjust based on risk appetite.
    """
    if probability >= 0.85:
        return "Low Risk"
    elif probability >= 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


def prepare_input(data: ApplicantInput) -> pd.DataFrame:
    """
    Converts validated Pydantic input to a DataFrame that matches
    the exact column format the pipeline expects.
    """
    row = {
        "Loan_ID":          "INFERENCE",   # dropped by preprocessor
        "Gender":           data.Gender,
        "Married":          data.Married,
        "Dependents":       data.Dependents,
        "Education":        data.Education,
        "Self_Employed":    data.Self_Employed,
        "ApplicantIncome":  data.ApplicantIncome,
        "CoapplicantIncome":data.CoapplicantIncome,
        "LoanAmount":       data.LoanAmount,
        "Loan_Amount_Term": data.Loan_Amount_Term,
        "Credit_History":   data.Credit_History,
        "Property_Area":    data.Property_Area,
        "Loan_Status":      0,   # placeholder — dropped by get_X_y, not used
    }
    df = pd.DataFrame([row])
    df = clean_dependents(df)
    df = handle_credit_history(df)
    df = engineer_features(df)

    # Drop columns the pipeline doesn't expect at inference time
    df = df.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
    return df

# At module level — add alongside MODEL
SHAP_EXPLAINER = None
FEATURE_NAMES  = []

def get_shap_contributions(df: pd.DataFrame, n_top: int = 5) -> list[dict]:
    """
    Computes SHAP feature contributions for a single prediction.

    SHAP 0.51+ returns ndarray shape (n_samples, n_features, n_classes).
    Confirmed by diagnostic: shape (1, 21, 2).
    We extract index [0, :, 1] = sample 0, all features, class 1 (approval).
    """
    try:
        import shap
        import numpy as np

        preprocessor = MODEL.named_steps['preprocessor']
        classifier   = MODEL.named_steps['classifier']

        # Step 1 — preprocess the input row
        X_proc = preprocessor.transform(df)                  # shape (1, 21)

        # Step 2 — get feature names, strip sklearn prefix
        try:
            raw_names     = preprocessor.get_feature_names_out()
            feature_names = [n.split("__")[-1] for n in raw_names]
        except Exception as fe:
            logger.warning("feature_names_out failed: %s", fe)
            feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]

        # Step 3 — build explainer
        # Use pre-warmed explainer if available (set during lifespan startup)
        explainer = (SHAP_EXPLAINER
                     if SHAP_EXPLAINER is not None
                     else shap.TreeExplainer(classifier))

        # Step 4 — compute SHAP values
        shap_vals = explainer.shap_values(X_proc, check_additivity=False)

        # Step 5 — extract class-1 values for sample 0
        # Your SHAP 0.51 returns ndarray (1, 21, 2)
        # ndim=3 → [sample_idx, all_features, class_idx]
        if isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                vals = shap_vals[0, :, 1]      # ← your case: (1,21,2)
            elif shap_vals.ndim == 2:
                vals = shap_vals[0]             # (n_samples, n_features)
            else:
                vals = shap_vals.flatten()      # (n_features,)

        elif isinstance(shap_vals, list):
            # Old SHAP < 0.42: list of arrays per class
            vals = np.array(shap_vals[1])[0]   # class 1, sample 0

        else:
            logger.error("Unknown SHAP type: %s", type(shap_vals))
            return []

        # Step 6 — length safety check
        n             = min(len(vals), len(feature_names))
        vals          = np.array(vals).flatten()[:n]
        feature_names = feature_names[:n]

        # Step 7 — build sorted contribution list
        contributions = [
            {
                "feature":    name,
                "shap_value": round(float(val), 4),
            }
            for name, val in zip(feature_names, vals)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return contributions[:n_top]

    except Exception as e:
        logger.warning("SHAP failed: %s", str(e), exc_info=True)
        return []

# Training distribution bounds — computed from train.csv
# Update these when model is retrained on new data
TRAINING_BOUNDS = {
    "ApplicantIncome":   {"max": 81000,  "p99": 25000},
    "CoapplicantIncome": {"max": 41667,  "p99": 15000},
    "LoanAmount":        {"max": 700,    "p99": 400},
}

def check_distribution(data: ApplicantInput) -> list[str]:
    """
    Returns list of warning messages for OOD inputs.
    Does NOT block the prediction — warns the user instead.
    """
    warnings_list = []
    checks = [
        ("ApplicantIncome",   data.ApplicantIncome),
        ("CoapplicantIncome", data.CoapplicantIncome),
        ("LoanAmount",        data.LoanAmount),
    ]
    for field, value in checks:
        bound = TRAINING_BOUNDS.get(field, {})
        if value > bound.get("max", float("inf")):
            warnings_list.append(
                f"{field} ({value}) exceeds training maximum "
                f"({bound['max']}). Prediction may be unreliable."
            )
        elif value > bound.get("p99", float("inf")):
            warnings_list.append(
                f"{field} ({value}) is above 99th percentile of "
                f"training data. Use with caution."
            )
    return warnings_list
# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Liveness endpoint.
    Deployment platforms (Render, Railway, K8s) ping this every 30 seconds.
    If it returns non-200, the platform restarts the container.
    """
    return HealthResponse(
        status      = "healthy" if MODEL is not None else "degraded",
        model_loaded= MODEL is not None,
        version     = "1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns metadata about the currently loaded model."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    preprocessor  = MODEL.named_steps['preprocessor']
    classifier    = MODEL.named_steps['classifier']

    try:
        n_features = len(preprocessor.get_feature_names_out())
    except Exception:
        n_features = -1

    return ModelInfoResponse(
        model_name   = MODEL_NAME,
        threshold    = THRESHOLD,
        feature_count= n_features,
        model_type   = type(classifier).__name__,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(applicant: ApplicantInput):
    """
    Main prediction endpoint.

    Accepts applicant details, returns loan decision with explanation.

    The pipeline:
    1. Pydantic validates input types and ranges (automatic)
    2. prepare_input() converts to DataFrame + runs feature engineering
    3. MODEL.predict_proba() runs preprocessing + inference
    4. Threshold applied to get binary decision
    5. SHAP values computed for top contributing features
    6. Structured response returned
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    try:
        # Prepare input dataframe
        df = prepare_input(applicant)

        # Check for out-of-distribution inputs BEFORE prediction
        ood_warnings = check_distribution(applicant)
        if ood_warnings:
            for w in ood_warnings:
                logger.warning("OOD input detected: %s", w)

        # Get approval probability
        prob = float(MODEL.predict_proba(df)[0][1])

        # Apply business threshold
        decision = "Approved" if prob >= THRESHOLD else "Rejected"

        # Risk category
        risk = get_risk_category(prob)

        # SHAP explanations
        factors = get_shap_contributions(df)

        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            "Prediction: %s | prob=%.3f | risk=%s | time=%.1fms",
            decision, prob, risk, elapsed_ms
        )

        return PredictionResponse(
            decision            = decision,
            approval_probability= round(prob, 4),
            risk_category       = risk,
            threshold_used      = THRESHOLD,
            top_factors         = factors,
            processing_time_ms  = round(elapsed_ms, 2),
            warnings            = ood_warnings,        # ← only new line
        )

    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"Prediction error: {str(e)}")