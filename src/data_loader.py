# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import DATA_RAW, TARGET, RANDOM_STATE, TEST_SIZE

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs. Returns (train_df, test_df)."""
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
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
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # ← this is the critical parameter
    )