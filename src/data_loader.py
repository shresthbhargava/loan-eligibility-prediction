# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import DATA_RAW, TARGET, RANDOM_STATE, TEST_SIZE
from logger import get_logger
logger = get_logger(__name__)

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs. Returns (train_df, test_df)."""
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    logger.info(f"Loaded training data: {train.shape}")
    logger.info(f"Loaded test data: {test.shape}")
    return train, test

def clean_dependents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Dependents from string ordinal to integer.
    '3+' → 3, preserving ordinal meaning.
    WHY: sklearn cannot process string features in numerical pipelines.
    """
    dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    df = df.copy()   # never mutate the original dataframe
    df['Dependents'] = df['Dependents'].map(dep_map)
    return df

def handle_credit_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit_History missing is MNAR — unknown credit history is
    meaningfully different from 0 (defaulted) or 1 (clean).
    Strategy: fill with -1 as a third category, then let the model
    learn its own weight for this state.
    """
    df = df.copy()
    df['Credit_History'] = df['Credit_History'].fillna(-1)
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Loan_Status: Y → 1, N → 0."""
    df = df.copy()
    df[TARGET] = (df[TARGET] == 'Y').astype(int)
    return df

def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def get_train_val_split(X, y):
    """
    Stratified split to preserve class ratio in both sets.
    WHY stratified? With 68/32 imbalance, a random split might give
    a validation set that's 80% approved — making metrics misleading.
    stratify=y guarantees the same 68/32 ratio in both sets.
    """
    logger.info(
    f"Creating stratified train/validation split "
    f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})"
)
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # ← this is the critical parameter
    )
    
    
# Add this function to src/data_loader.py

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-informed features from raw applicant data.

    WHY each feature:

    TotalIncome:
        A co-applicant's income is equally real money available for repayment.
        Keeping them separate loses the household income picture.
        Banks always evaluate household income, not individual income.

    LoanAmountPerIncome (DTI proxy):
        Debt-to-income ratio is THE standard affordability metric in banking.
        A ₹200k loan means nothing without knowing if income is ₹50k or ₹500k.
        This ratio normalizes loan burden relative to earning power.
        Expected to be one of the top SHAP features.

    IncomePerDependent:
        Each dependent reduces disposable income available for EMI.
        Two applicants with identical incomes have different real affordability
        if one has 0 dependents and the other has 3.
        Adding 1 to Dependents avoids division by zero.

    LoanAmountLog:
        Standalone log-transformed loan amount — captures the non-linear
        relationship between loan size and default risk.
        Large loans have disproportionately higher default risk.

    EMI_estimate:
        Approximates monthly payment using a simplified formula.
        Loan_Amount_Term is in months — this gives a rough monthly burden.
        More interpretable than raw LoanAmount for downstream analysis.

    IsHighIncome:
        Binary flag: is applicant in top 25% of income?
        Tree models can find this threshold themselves, but explicit flags
        help linear models (Logistic Regression) capture non-linear jumps.

    IsSelfEmployedWithHighLoan:
        Interaction feature: self-employed AND large loan = high risk profile.
        Neither feature alone captures this compound risk.
        Banks specifically scrutinize this combination.
    """
    df = df.copy()  # never mutate input

    # ── Core financial ratios ─────────────────────────────────────────────────
    df['TotalIncome'] = (
        df['ApplicantIncome'] + df['CoapplicantIncome']
    )

    df['LoanAmountPerIncome'] = (
        df['LoanAmount'] / (df['TotalIncome'] / 1000 + 1e-6)
        # +1e-6 prevents division by zero for zero-income edge cases
        # LoanAmount is in thousands so we divide TotalIncome by 1000
    )

    df['IncomePerDependent'] = (
        df['TotalIncome'] / (df['Dependents'].fillna(0) + 1)
        # +1 so single applicants (0 dependents) don't divide by zero
    )

    # ── Log transformations (standalone, before pipeline) ─────────────────────
    df['LoanAmountLog'] = np.log1p(df['LoanAmount'].fillna(0))
    df['TotalIncomeLog'] = np.log1p(df['TotalIncome'])

    # ── EMI approximation ────────────────────────────────────────────────────
    term = df['Loan_Amount_Term'].fillna(360)   # default 30-year term
    df['EMI_estimate'] = (df['LoanAmount'].fillna(0) * 1000) / (term + 1)
    # multiply by 1000 because LoanAmount is in thousands

    df['EMI_to_Income_ratio'] = (
        df['EMI_estimate'] / (df['TotalIncome'] + 1)
    )

    # ── Binary interaction flags ──────────────────────────────────────────────
    income_75th = df['TotalIncome'].quantile(0.75)
    df['IsHighIncome'] = (df['TotalIncome'] > income_75th).astype(int)

    loan_median = df['LoanAmount'].fillna(0).median()
    df['IsSelfEmployedHighLoan'] = (
        (df['Self_Employed'] == 'Yes') &
        (df['LoanAmount'].fillna(0) > loan_median)
    ).astype(int)

    df['HasCoapplicant'] = (df['CoapplicantIncome'] > 0).astype(int)

    return df