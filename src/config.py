from pathlib import Path;
ROOT_DIR=Path(__file__).parent.parent
DATA_RAW=ROOT_DIR/"data"/"raw"
DATA_PROCESSED=ROOT_DIR/"data"/"processed"
MODELS_DIR=ROOT_DIR/"models"

Target="Loan_Status"
DROP_COLS=["Loan_ID"]
NUMERICAL_FEATURES=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
CATEGORICAL_FEATURES=["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"]
ORDINAL_FEATURES=["Dependents"]
BINARTY_FEATURES=["Credit_History"]
RANDOM_FEATURES=["Credit_History"]

RANDOM_STATE=42
TEST_SIZE=0.2