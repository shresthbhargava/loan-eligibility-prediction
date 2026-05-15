# src/train_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# End-to-end training pipeline. Single entry point.
#
# Run with: python src/train_pipeline.py
#
# This script replaces running notebooks manually.
# It is what your CI/CD pipeline (Phase 15) will execute on every push.
# It is what a new team member runs to reproduce your results.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger      import get_logger
from config      import MODELS_DIR, DATA_PROCESSED, RANDOM_STATE
from data_loader import (load_raw_data, clean_dependents,
                          handle_credit_history, encode_target,
                          engineer_features, get_X_y, get_train_val_split)
from preprocessor import build_preprocessor
from trainer      import get_models, build_full_pipeline
from evaluator    import evaluate_all_models

logger = get_logger("train_pipeline")


def run_pipeline() -> dict:
    """
    Executes full training pipeline end-to-end.
    Returns evaluation results dictionary.
    """
    start_time = time.time()
    logger.info("=" * 55)
    logger.info("Loan Eligibility — Training Pipeline Started")
    logger.info("=" * 55)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    logger.info("Step 1/5: Loading and cleaning data")
    try:
        df, _ = load_raw_data()
        original_shape = df.shape
        df = clean_dependents(df)
        df = handle_credit_history(df)
        df = encode_target(df)
        df = engineer_features(df)
        logger.info("  Loaded %d rows, %d columns", *original_shape)
        logger.info("  After engineering: %d columns", df.shape[1])
    except FileNotFoundError as e:
        logger.error("Data files not found: %s", e)
        logger.error("Place train.csv and test.csv in data/raw/")
        raise

    # ── Step 2: Split ────────────────────────────────────────────────────────
    logger.info("Step 2/5: Creating train/validation split")
    X, y  = get_X_y(df)
    X_train, X_val, y_train, y_val = get_train_val_split(X, y)
    logger.info("  Train: %d samples | Val: %d samples", len(X_train), len(X_val))
    logger.info("  Approval rate — train: %.1f%% | val: %.1f%%",
                y_train.mean()*100, y_val.mean()*100)

    # ── Step 3: Train all models ─────────────────────────────────────────────
    logger.info("Step 3/5: Training all models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    models = get_models()
    fitted = {}

    for name, model in models.items():
        logger.info("  Training %s ...", name)
        t0       = time.time()
        pre      = build_preprocessor(use_engineered=True)
        pipeline = build_full_pipeline(model)

        # Rebuild with correct preprocessor
        from sklearn.pipeline import Pipeline as SKPipeline
        pipeline = SKPipeline([("preprocessor", pre), ("classifier", model)])
        pipeline.fit(X_train, y_train)

        elapsed = time.time() - t0
        fitted[name] = pipeline
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, path)
        logger.info("  [OK] %s trained in %.1fs | saved to %s", name, elapsed, path)

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    logger.info("Step 4/5: Evaluating all models")
    results_df = evaluate_all_models(fitted, X_val, y_val)

    logger.info("\n%s", results_df.to_string())

    best_name = results_df['roc_auc'].idxmax()
    best_auc  = results_df.loc[best_name, 'roc_auc']
    logger.info("  Best model: %s (AUC %.4f)", best_name, best_auc)

    # ── Step 5: Save final model ─────────────────────────────────────────────
    logger.info("Step 5/5: Saving final model")
    final_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(fitted[best_name], final_path)

    # Save evaluation results for CI/CD checks
    results_path = DATA_PROCESSED / "evaluation_results.csv"
    results_df.to_csv(results_path)

    elapsed_total = time.time() - start_time
    logger.info("Pipeline completed in %.1fs", elapsed_total)
    logger.info("Final model saved: %s", final_path)
    logger.info("=" * 55)

    return {
        "best_model":  best_name,
        "best_auc":    best_auc,
        "results":     results_df,
        "fitted":      fitted,
        "X_val":       X_val,
        "y_val":       y_val,
    }


if __name__ == "__main__":
    results = run_pipeline()
    sys.exit(0)