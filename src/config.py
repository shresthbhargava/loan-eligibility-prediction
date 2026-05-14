# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration. ALL constants live here.
# Never hardcode paths or column names inside functions.
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"

# ── Column definitions ───────────────────────────────────────────────────────
TARGET             = "Loan_Status"
DROP_COLS          = ["Loan_ID"]
NUMERICAL_FEATURES = ["ApplicantIncome", "CoapplicantIncome",
                       "LoanAmount", "Loan_Amount_Term"]
CATEGORICAL_FEATURES = ["Gender", "Married", "Education",
                         "Self_Employed", "Property_Area"]
ORDINAL_FEATURES   = ["Dependents"]
BINARY_FEATURES    = ["Credit_History"]

# ── Model settings ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2