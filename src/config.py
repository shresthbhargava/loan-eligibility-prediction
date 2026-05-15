# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration. All constants live here.
# Never hardcode paths, column names, or model parameters inside functions.
# Change here and it propagates everywhere automatically.
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"

# ── Target ────────────────────────────────────────────────────────────────────
TARGET    = "Loan_Status"
DROP_COLS = ["Loan_ID"]

# ── Base features ─────────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
]

# Already numeric after data_loader — impute only, no encoding
PASSTHROUGH_FEATURES = [
    "Credit_History",   # -1 / 0 / 1
    "Dependents",       #  0 / 1 / 2 / 3
]

# ── Engineered features (created by engineer_features() in data_loader.py) ───
ENGINEERED_NUMERICAL = [
    "TotalIncome",
    "LoanAmountPerIncome",
    "IncomePerDependent",
    "EMI_to_Income_ratio",
]

ENGINEERED_LOG = [
    "LoanAmountLog",
    "TotalIncomeLog",
]

ENGINEERED_BINARY = [
    "IsHighIncome",
    "IsSelfEmployedHighLoan",
    "HasCoapplicant",
]

# ── Model settings ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2