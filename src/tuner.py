# src/tuner.py
# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter tuning with RandomizedSearchCV.
#
# Goal for this dataset specifically:
#   - Reduce overfit gap from 0.22-0.25 toward < 0.10
#   - Improve or maintain CV ROC-AUC above 0.76
#   - Constrain model complexity given only 491 training rows
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessor import build_preprocessor
from trainer      import build_full_pipeline, get_models
from config       import RANDOM_STATE, MODELS_DIR


def get_param_grids() -> dict:
    """
    Parameter search spaces. Note that for a small dataset (491 rows),
    we deliberately include aggressive regularization values.

    KEY PARAMETERS EXPLAINED:

    RandomForest:
      max_depth [3,5,7,10,None]:
        None = fully grown (causes our 0.99 train AUC).
        3-7 = constrained trees = less overfitting.
        We expect the tuner to prefer 5-8 for this dataset.

      min_samples_leaf [4,8,16,32]:
        Higher values = smoother decision boundaries = less overfitting.
        With 491 rows, leaf=32 means each leaf covers 6.5% of data — healthy.

      max_features ['sqrt','log2',0.5]:
        Controls feature diversity between trees.
        0.5 = use 50% of features at each split.

    XGBoost:
      max_depth [2,3,4]:
        Boosting works best with shallow trees (2-4).
        Deep trees in boosting = overfitting much faster than RF.

      learning_rate [0.01,0.05,0.1]:
        Lower = more conservative corrections = less overfitting.
        BUT requires more n_estimators to converge.

      subsample [0.6,0.7,0.8]:
        Row sampling per tree. Lower = more regularization.

      reg_lambda [1,2,5,10]:
        L2 regularization. Higher = stronger penalty on large weights.
        Our overfit gap of 0.25 suggests we need reg_lambda > 1.

      reg_alpha [0,0.1,0.5,1]:
        L1 regularization. Encourages sparsity — some weights → 0.
    """
    return {
        "RandomForest": {
            "classifier__n_estimators":      [100, 200, 300],
            "classifier__max_depth":         [3, 5, 7, 10, None],
            "classifier__min_samples_split":  [10, 20, 30],
            "classifier__min_samples_leaf":   [4, 8, 16, 32],
            "classifier__max_features":      ["sqrt", "log2", 0.5],
        },
        "XGBoost": {
            "classifier__n_estimators":      [100, 200, 300],
            "classifier__max_depth":         [2, 3, 4],
            "classifier__learning_rate":     [0.01, 0.05, 0.1],
            "classifier__subsample":         [0.6, 0.7, 0.8],
            "classifier__colsample_bytree":  [0.6, 0.7, 0.8],
            "classifier__reg_lambda":        [1, 2, 5, 10],
            "classifier__reg_alpha":         [0, 0.1, 0.5, 1],
            "classifier__min_child_weight":  [5, 10, 20],
        },
    }


def tune_model(model_name: str, X_train, y_train,
               n_iter: int = 60) -> tuple:
    """
    Runs RandomizedSearchCV with stratified 5-fold CV.

    WHY n_iter=60?
    RandomForest grid has ~3×5×3×4×3 = 540 combinations.
    XGBoost grid has ~3×3×3×3×3×4×4×3 = 11,664 combinations.
    Exhaustive search is impractical. 60 random samples finds
    near-optimal params in ~5-10 minutes on a laptop.

    Research by Bergstra & Bengio (2012) showed random search
    finds equally good params as grid search in 1/10 the iterations
    because most parameters have low importance — a few dominate.
    For RF, max_depth and min_samples_leaf dominate.
    For XGBoost, max_depth and learning_rate dominate.

    WHY refit=True?
    After finding best params via CV, refit=True trains one final
    model on the ENTIRE X_train using those params.
    This is the model you save and deploy — not a CV fold model.
    """
    all_models  = get_models()
    param_grids = get_param_grids()

    base_model  = all_models[model_name]
    param_grid  = param_grids[model_name]
    pipeline    = build_full_pipeline(base_model)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator           = pipeline,
        param_distributions = param_grid,
        n_iter              = n_iter,
        scoring             = "roc_auc",   # same metric we've used throughout
        cv                  = skf,
        verbose             = 1,
        random_state        = RANDOM_STATE,
        n_jobs              = -1,
        refit               = True,
    )

    print(f"\nTuning {model_name} ({n_iter} iterations × 5 folds = "
          f"{n_iter * 5} model fits)...")
    search.fit(X_train, y_train)

    print(f"\n  Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"  Best params:")
    for k, v in search.best_params_.items():
        print(f"    {k.replace('classifier__',''):<25} = {v}")

    return search.best_estimator_, search.best_params_, search.best_score_


def run_tuning_analysis(search_cv, model_name: str, top_n: int = 10):
    """
    Prints the top N parameter combinations found.
    Teaches you which parameters mattered most.
    """
    results = pd.DataFrame(search_cv.cv_results_)
    top = (results[['params', 'mean_test_score', 'std_test_score']]
           .sort_values('mean_test_score', ascending=False)
           .head(top_n))

    print(f"\nTop {top_n} parameter sets for {model_name}:")
    print(f"{'Rank':<5} {'CV AUC':<10} {'Std':<8} Key params")
    print("-" * 60)
    for i, row in top.iterrows():
        params_short = {k.replace('classifier__', ''): v
                        for k, v in row['params'].items()}
        # Show only the two most important params per model
        if model_name == 'RandomForest':
            key = f"depth={params_short.get('max_depth')}  " \
                  f"leaf={params_short.get('min_samples_leaf')}"
        else:
            key = f"depth={params_short.get('max_depth')}  " \
                  f"lr={params_short.get('learning_rate')}  " \
                  f"lambda={params_short.get('reg_lambda')}"
        print(f"{i:<5} {row['mean_test_score']:.4f}     "
              f"{row['std_test_score']:.4f}   {key}")