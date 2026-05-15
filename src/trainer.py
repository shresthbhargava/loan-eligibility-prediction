# src/trainer.py
# ─────────────────────────────────────────────────────────────────────────────
# Model definitions and training pipeline.
# Each model is configured with reasonable defaults — not final tuned values.
# Phase 7 (hyperparameter tuning) will optimize these.
# ─────────────────────────────────────────────────────────────────────────────
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from preprocessor import build_preprocessor
from config import MODELS_DIR, RANDOM_STATE

def get_models() -> dict:
    """
    Returns a dictionary of model name → sklearn estimator.

    WHY a dictionary? It lets us loop over all models with identical
    training and evaluation code — no copy-paste per model.
    This is the DRY (Don't Repeat Yourself) principle in practice.

    Default parameters are intentionally conservative:
    - max_depth limits prevent overfitting before tuning
    - random_state ensures reproducibility across runs
    - eval_metric on XGBoost suppresses a version-compatibility warning
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,          # default 100 often fails to converge
            random_state=RANDOM_STATE,
            C=1.0,                  # inverse regularization strength
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5,            # constrained to prevent memorization
            min_samples_split=10,   # a split needs at least 10 samples
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,       # 100 trees — good default
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,              # use all CPU cores
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,      # shrinkage — smaller = slower but better
            max_depth=3,            # shallow trees work better for boosting
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=RANDOM_STATE,
            eval_metric='logloss',  # suppresses warning; logloss = cross-entropy
            use_label_encoder=False,
            n_jobs=-1
        ),
    }


def build_full_pipeline(model) -> Pipeline:
    """
    Wraps preprocessor + model into a single sklearn Pipeline.

    WHY a full pipeline?
    When you call pipeline.predict(raw_df), preprocessing and inference
    happen in one step. This is critical for deployment — your FastAPI
    endpoint receives raw applicant data, not preprocessed arrays.
    The pipeline handles everything internally.

    At inference time: pipeline.predict([raw_input]) → preprocesses → predicts
    No manual preprocessing needed. No risk of forgetting a step.
    """
    preprocessor = build_preprocessor()
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def train_all_models(X_train, y_train) -> dict:
    """
    Trains all models and returns fitted pipelines.
    Each pipeline includes preprocessing + the model.

    Returns: dict of model_name → fitted_pipeline
    """
    models = get_models()
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        pipeline = build_full_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        print("done")

    return fitted_pipelines


def save_model(pipeline: Pipeline, name: str) -> Path:
    """
    Saves a fitted pipeline to disk using joblib.

    WHY joblib over pickle?
    joblib is optimized for large numpy arrays (model weights).
    It compresses them efficiently. pickle is general-purpose
    and slower for numpy-heavy objects.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    print(f"  Saved: {path}")
    return path


def load_model(name: str) -> Pipeline:
    """Loads a saved pipeline from disk."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Train first.")
    return joblib.load(path)