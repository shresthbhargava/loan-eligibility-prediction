# src/explainer.py
# ─────────────────────────────────────────────────────────────────────────────
# SHAP-based explainability for the loan eligibility model.
#
# Two levels of explanation:
#   Global  — which features matter most across ALL applicants?
#             Used for model validation and business reporting.
#   Local   — why was THIS specific applicant approved or rejected?
#             Used for individual applicant explanations and compliance.
#
# WHY SHAP over simpler alternatives?
#   Feature importance (sklearn): tells you split frequency, not contribution
#   Permutation importance: tells you mean contribution, not directional
#   SHAP: tells you exact signed contribution per feature per prediction
#         and is mathematically guaranteed to be consistent and locally accurate
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

APPROVE_COLOR = '#00c896'
REJECT_COLOR  = '#ff4d6d'
NEUTRAL_COLOR = '#5b7fff'


def get_shap_explainer(pipeline, X_train: pd.DataFrame):
    """
    Creates a SHAP TreeExplainer for the fitted pipeline.

    WHY TreeExplainer specifically?
      TreeExplainer is optimized for tree-based models (Random Forest,
      XGBoost, GradientBoosting). It computes exact SHAP values in
      polynomial time using the tree structure directly.

      The alternative, KernelExplainer, works on any model but uses
      sampling approximation — much slower and less accurate.

    WHY preprocess X_train before passing to explainer?
      SHAP operates on the model's classifier step directly, not the
      full pipeline. So we must manually preprocess X_train first
      to match the format the classifier expects.

    Returns: (explainer, shap_values, feature_names)
    """
    # Extract fitted preprocessor and classifier from pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    classifier   = pipeline.named_steps['classifier']

    # Transform training data — explainer needs processed features
    X_train_proc = preprocessor.transform(X_train)

    # Get feature names from the preprocessor
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
        # Clean up sklearn's prefix format: "num__ApplicantIncome" → "ApplicantIncome"
        feature_names = [n.split('__')[-1] for n in feature_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    # Build TreeExplainer
    # check_additivity=False: suppresses floating point precision warnings
    # that occasionally appear with Random Forest ensembles
    explainer = shap.TreeExplainer(
    classifier,
    data=X_train_proc,
    feature_names=feature_names,
    )

    shap_values = explainer(
    X_train_proc,
    check_additivity=False
    )
    return explainer, shap_values, feature_names, X_train_proc


def plot_global_importance(shap_values, feature_names: list,
                           top_n: int = 15, save_path: str = None):
    """
    Bar chart of mean absolute SHAP values — global feature importance.

    WHY mean absolute SHAP?
      Mean |SHAP| = average magnitude of each feature's contribution
      across all applicants, regardless of direction.
      This is the fairest measure of feature importance because:
        - it's consistent (doesn't depend on class distribution)
        - it's additive (contributions sum to prediction)
        - it captures non-linear effects unlike correlation

    Interpretation: a feature with mean |SHAP| = 0.15 moved the average
    prediction by 15 percentage points (in log-odds space) on average.
    """
    # Handle both old and new SHAP API formats
    if hasattr(shap_values, 'values'):
        vals = shap_values.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]   # class 1 (approval)
    else:
        vals = shap_values

    mean_abs = np.abs(vals).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature':    feature_names[:len(mean_abs)],
        'importance': mean_abs
    }).sort_values('importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [APPROVE_COLOR if imp > importance_df['importance'].median()
              else NEUTRAL_COLOR
              for imp in importance_df['importance']]

    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                   color=colors, edgecolor='none')

    for bar, val in zip(bars, importance_df['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_title('Global Feature Importance (Mean |SHAP|)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Mean |SHAP Value| — average impact on prediction')
    ax.grid(True, axis='x', alpha=0.3)

    high_patch = mpatches.Patch(color=APPROVE_COLOR, label='Above median importance')
    low_patch  = mpatches.Patch(color=NEUTRAL_COLOR, label='Below median importance')
    ax.legend(handles=[high_patch, low_patch], loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    print("\nTop 5 most important features:")
    for _, row in importance_df.tail(5).iloc[::-1].iterrows():
        print(f"  {row['feature']:<30} mean|SHAP| = {row['importance']:.4f}")

    return importance_df


def plot_beeswarm(shap_values, feature_names: list, save_path: str = None):
    """
    SHAP beeswarm plot — the most information-dense SHAP visualization.

    Each dot = one applicant.
    X axis  = SHAP value (positive = pushed toward approval)
    Color   = feature value (red = high, blue = low)
    Y axis  = features ranked by importance

    What this reveals that a bar chart cannot:
      - Direction: does high Credit_History push toward approval or rejection?
      - Distribution: are most applicants clustered or spread?
      - Outliers: are there unusual applicants with extreme feature impacts?
      - Interaction hints: are there two clusters suggesting a threshold effect?
    """
    if hasattr(shap_values, 'values'):
        vals = shap_values.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        data = shap_values.data
    else:
        vals = shap_values
        data = None

    fig, ax = plt.subplots(figsize=(12, 9))

    shap.summary_plot(
        vals,
        features=data,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=15,
    )

    plt.title('SHAP Beeswarm — Feature Impact Distribution',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def explain_single_applicant(pipeline, applicant_df: pd.DataFrame,
                              explainer, feature_names: list,
                              applicant_label: str = "Applicant",
                              save_path: str = None):
    """
    Generates a waterfall explanation for one individual applicant.

    This is the LOCAL explanation — answers "why was this person
    approved or rejected?"

    Waterfall chart reads:
      Start: E[f(x)] = base value (average prediction across training set)
      Each bar: this feature pushed prediction up (+) or down (-)
      End: f(x) = final prediction for this applicant

    This is what a bank compliance officer would show to a rejected
    applicant who asks "why was my loan rejected?"
    Legally required in many jurisdictions under GDPR Article 22
    and the EU AI Act (high-risk AI systems).
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier   = pipeline.named_steps['classifier']

    # Preprocess the single applicant
    X_proc = preprocessor.transform(applicant_df)

    # Get prediction and probability
    prob    = classifier.predict_proba(X_proc)[0][1]
    decision = "APPROVED" if prob >= 0.77 else "REJECTED"
    color    = APPROVE_COLOR if decision == "APPROVED" else REJECT_COLOR

    # Compute SHAP values for this single applicant
    shap_vals_single = explainer(X_proc)

    if hasattr(shap_vals_single, 'values'):
        vals = shap_vals_single.values
        if vals.ndim == 3:
            vals = vals[0, :, 1]
        else:
            vals = vals[0]
        base = shap_vals_single.base_values
        if isinstance(base, np.ndarray):
            base = base[0] if base.ndim == 1 else base[0][1]
    else:
        vals = shap_vals_single[0]
        base = explainer.expected_value

    # Build contribution dataframe
    contrib_df = pd.DataFrame({
        'feature':      feature_names[:len(vals)],
        'shap_value':   vals,
        'feature_value': X_proc[0][:len(vals)]
    }).sort_values('shap_value', key=abs, ascending=True).tail(12)

    # ── Waterfall chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    running = float(base)
    bar_colors = []
    bar_positions = []
    bar_widths = []

    for _, row in contrib_df.iterrows():
        bar_positions.append(running + row['shap_value'] / 2)
        bar_widths.append(row['shap_value'])
        bar_colors.append(APPROVE_COLOR if row['shap_value'] > 0
                          else REJECT_COLOR)
        running += row['shap_value']

    bars = ax.barh(
        range(len(contrib_df)),
        bar_widths,
        left=[p - w/2 for p, w in zip(bar_positions, bar_widths)],
        color=bar_colors,
        edgecolor='none',
        height=0.6,
    )

    ax.set_yticks(range(len(contrib_df)))
    ax.set_yticklabels(contrib_df['feature'].tolist(), fontsize=10)

    # Add value labels
    for i, (_, row) in enumerate(contrib_df.iterrows()):
        sign = "+" if row['shap_value'] > 0 else ""
        ax.text(float(base) + sum(contrib_df['shap_value'].iloc[:i+1]),
                i, f"  {sign}{row['shap_value']:.3f}",
                va='center', fontsize=8.5)

    ax.axvline(x=float(base), color='white', linestyle='--',
               alpha=0.5, linewidth=1.5, label=f'Base: {float(base):.3f}')

    ax.set_title(
        f'{applicant_label} — Prediction: {decision} '
        f'(Probability: {prob:.1%})',
        fontsize=13, fontweight='bold', pad=15,
        color=color
    )
    ax.set_xlabel('SHAP value contribution to approval probability')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

    # ── Text summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {applicant_label} Explanation Report")
    print(f"{'='*55}")
    print(f"  Decision     : {decision}")
    print(f"  Probability  : {prob:.1%}")
    print(f"  Base rate    : {float(base):.1%}")
    print(f"\n  Top factors pushing toward APPROVAL:")
    pos = contrib_df[contrib_df['shap_value'] > 0].sort_values(
        'shap_value', ascending=False).head(3)
    for _, r in pos.iterrows():
        print(f"    ✓ {r['feature']:<28} +{r['shap_value']:.4f}")
    print(f"\n  Top factors pushing toward REJECTION:")
    neg = contrib_df[contrib_df['shap_value'] < 0].sort_values(
        'shap_value').head(3)
    for _, r in neg.iterrows():
        print(f"    ✗ {r['feature']:<28} {r['shap_value']:.4f}")
    print(f"{'='*55}")

    return contrib_df