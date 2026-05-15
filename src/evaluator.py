# src/evaluator.py
# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation with all metrics relevant to a banking classification task.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay, classification_report
)

APPROVE_COLOR = '#00c896'
REJECT_COLOR  = '#ff4d6d'
NEUTRAL_COLOR = '#5b7fff'


def evaluate_model(pipeline, X_val, y_val, model_name: str) -> dict:
    """
    Computes all relevant metrics for a fitted pipeline.

    WHY these specific metrics?
    - accuracy:  included but lowest priority — misleading with imbalance
    - precision: of all approved, how many actually repay → bank's profit metric
    - recall:    of all who repay, how many did we approve → revenue metric
    - f1:        harmonic mean of precision and recall → overall quality
    - roc_auc:   threshold-independent — measures separation quality
    """
    y_pred      = pipeline.predict(X_val)
    y_pred_prob = pipeline.predict_proba(X_val)[:, 1]  # probability of approval

    return {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred), 4),
        "recall":    round(recall_score(y_val, y_pred), 4),
        "f1":        round(f1_score(y_val, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_val, y_pred_prob), 4),
    }


def evaluate_all_models(fitted_pipelines: dict, X_val, y_val) -> pd.DataFrame:
    """Evaluates all models and returns a sorted comparison DataFrame."""
    results = []
    for name, pipeline in fitted_pipelines.items():
        metrics = evaluate_model(pipeline, X_val, y_val, name)
        results.append(metrics)

    df = pd.DataFrame(results).set_index("model")
    # Sort by ROC-AUC — the most honest metric for this problem
    return df.sort_values("roc_auc", ascending=False)


def plot_confusion_matrices(fitted_pipelines: dict, X_val, y_val):
    """Plots confusion matrices for all models side by side."""
    n = len(fitted_pipelines)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')

    for ax, (name, pipeline) in zip(axes, fitted_pipelines.items()):
        y_pred = pipeline.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'],
                    ax=ax, cbar=False,
                    annot_kws={'size': 13, 'weight': 'bold'})
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('../data/processed/eval_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(fitted_pipelines: dict, X_val, y_val):
    """Plots ROC curves for all models on a single axes."""
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = [APPROVE_COLOR, NEUTRAL_COLOR, '#f4a261', '#e76f51', '#a8dadc']
    for (name, pipeline), color in zip(fitted_pipelines.items(), colors):
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        RocCurveDisplay.from_predictions(
            y_val, y_prob, name=f"{name} (AUC={auc:.3f})",
            ax=ax, color=color
        )

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/processed/eval_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metric_comparison(results_df: pd.DataFrame):
    """Bar chart comparing all models across all metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    colors  = [APPROVE_COLOR, NEUTRAL_COLOR, '#f4a261', '#e76f51', '#a8dadc']

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
    fig.suptitle('Model Comparison — All Metrics', fontsize=14, fontweight='bold')

    for ax, metric, color in zip(axes, metrics, colors):
        vals = results_df[metric].sort_values(ascending=True)
        bars = ax.barh(vals.index, vals.values, color=color, edgecolor='none')
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xlim(0.5, 1.0)    # zoom in — differences matter at the margins
        for bar, val in zip(bars, vals.values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('../data/processed/eval_metric_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
def cross_validate_all_models(X_train, y_train, cv_folds: int = 5) -> pd.DataFrame:
    """5-fold stratified cross-validation across all models."""
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from trainer import get_models, build_full_pipeline

    skf     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = []

    for name, model in get_models().items():
        print(f"  Cross-validating {name}...", end=" ")
        pipeline   = build_full_pipeline(model)
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=skf, scoring=scoring,
            return_train_score=True
        )
        results.append({
            "model":          name,
            "cv_roc_auc":     round(cv_results['test_roc_auc'].mean(),   4),
            "cv_roc_auc_std": round(cv_results['test_roc_auc'].std(),    4),
            "cv_f1":          round(cv_results['test_f1'].mean(),        4),
            "cv_precision":   round(cv_results['test_precision'].mean(), 4),
            "cv_recall":      round(cv_results['test_recall'].mean(),    4),
            "train_roc_auc":  round(cv_results['train_roc_auc'].mean(),  4),
            "overfit_gap":    round(
                cv_results['train_roc_auc'].mean() -
                cv_results['test_roc_auc'].mean(), 4
            ),
        })
        print("done")

    df = pd.DataFrame(results).set_index("model")
    return df.sort_values("cv_roc_auc", ascending=False)


def find_optimal_threshold(pipeline, X_val, y_val,
                            fp_cost: float = 50000,
                            fn_cost: float = 10000) -> dict:
    """Finds the decision threshold that minimises total business cost."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_probs    = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    costs, precisions, recalls, f1s = [], [], [], []

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0,1]).ravel()
        costs.append((fp * fp_cost) + (fn * fn_cost))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2*p*r / (p+r)  if (p + r)  > 0 else 0
        precisions.append(p); recalls.append(r); f1s.append(f)

    idx   = int(np.argmin(costs))
    opt_t = thresholds[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Threshold Optimisation', fontsize=14, fontweight='bold')

    axes[0].plot(thresholds, [c/1e5 for c in costs], color='#ff4d6d', lw=2)
    axes[0].axvline(opt_t, color='#00c896', ls='--', lw=2,
                    label=f'Optimal: {opt_t:.2f}')
    axes[0].set_title('Business Cost vs Threshold')
    axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Cost (₹ lakhs)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(thresholds, precisions, color='#00c896', lw=2, label='Precision')
    axes[1].plot(thresholds, recalls,    color='#ff4d6d', lw=2, label='Recall')
    axes[1].plot(thresholds, f1s,        color='#5b7fff', lw=2, label='F1')
    axes[1].axvline(opt_t, color='white', ls='--', lw=1.5, alpha=0.7,
                    label=f'Optimal: {opt_t:.2f}')
    axes[1].set_title('Precision / Recall / F1 vs Threshold')
    axes[1].set_xlabel('Threshold'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/processed/eval_threshold_optimisation.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nOptimal threshold : {opt_t:.2f}")
    print(f"Minimum cost      : ₹{costs[idx]:,.0f}")
    print(f"Precision         : {precisions[idx]:.3f}")
    print(f"Recall            : {recalls[idx]:.3f}")
    print(f"F1                : {f1s[idx]:.3f}")

    return {"optimal_threshold": opt_t, "minimum_cost": costs[idx],
            "precision": precisions[idx], "recall": recalls[idx],
            "f1": f1s[idx]}