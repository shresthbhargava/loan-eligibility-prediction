# 🏦 Loan Eligibility Prediction System

[![CI Pipeline](https://github.com/shresthbhargava/loan-eligibility-prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shresthbhargava/loan-eligibility-prediction/actions)
[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/shresth0/loan-eligibility-predictor)
[![API](https://img.shields.io/badge/API-Render-blue)](https://loan-eligibility-prediction-4nh7.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

> End-to-end ML system for loan approval prediction with explainable AI,
> production API, and interactive dashboard.

## Live Demo

| Component | URL |
|-----------|-----|
| Interactive Dashboard | [Hugging Face Spaces](https://huggingface.co/spaces/shresth0/loan-eligibility-predictor) |
| REST API | [Render](https://loan-eligibility-prediction-4nh7.onrender.com/docs) |

> Note: First request may take 30-60s (Render free tier cold start)

## Results

| Model | Val AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| **Random Forest** | **0.8548** | **0.8837** | **0.8736** | **0.8941** |
| XGBoost | 0.8421 | 0.8671 | 0.8523 | 0.8824 |
| Decision Tree | 0.8303 | 0.9029 | 0.8778 | 0.9294 |
| Gradient Boosting | 0.7926 | 0.8736 | 0.8539 | 0.8941 |
| Logistic Regression | 0.7480 | 0.7857 | 0.6937 | 0.9059 |

**Business threshold:** 0.77 (cost-optimized: FP=₹50k, FN=₹10k → saves ₹13.4L vs default)

## Architecture

```
Applicant Input
      │
      ▼
Streamlit UI (HF Spaces)
      │  HTTP POST /predict
      ▼
FastAPI Backend (Render)
      │
      ├── Pydantic validation
      ├── Feature engineering (21 features)
      ├── RandomForest inference
      ├── SHAP explanation
      └── OOD detection
```

## Features

- **5 ML models** trained and compared with cross-validation
- **21 features** including engineered financial ratios (DTI, EMI burden)
- **SHAP explainability** — every prediction explained by feature contributions
- **Cost-optimized threshold** — tuned to minimize bank's financial loss
- **OOD detection** — flags inputs outside training distribution
- **FastAPI backend** — sub-150ms inference with Pydantic validation
- **Streamlit dashboard** — probability gauge, SHAP waterfall, risk category
- **Docker** — containerized for reproducible deployment
- **CI/CD** — GitHub Actions runs tests on every push

## Project Structure

```
loan-eligibility-predictor/
├── src/
│   ├── config.py           # central configuration
│   ├── data_loader.py      # loading + feature engineering
│   ├── preprocessor.py     # sklearn pipeline
│   ├── trainer.py          # model definitions
│   ├── evaluator.py        # metrics and visualization
│   ├── explainer.py        # SHAP analysis
│   └── train_pipeline.py   # end-to-end training script
├── api/
│   └── main.py             # FastAPI application
├── app/
│   └── streamlit_app.py    # Streamlit dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_shap_explainability.ipynb
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/loan-eligibility-predictor
cd loan-eligibility-predictor
pip install -r requirements.txt

# Train model
python src/train_pipeline.py

# Start API
uvicorn api.main:app --port 8000

# Start dashboard (new terminal)
streamlit run app/streamlit_app.py
```

## Key Technical Decisions

**Why Random Forest over XGBoost?**
With 614 training rows, Random Forest's variance reduction through bagging
outperforms XGBoost's boosting advantage. CV AUC: 0.766 vs 0.762.

**Why threshold 0.77 instead of 0.5?**
Cost analysis: FP (bad loan approved) costs ₹50k, FN (good loan rejected)
costs ₹10k. Threshold optimization reduced expected validation loss from
₹19.1L to ₹5.7L — a 70% reduction.

**Why SHAP over feature importance?**
sklearn's feature importance measures split frequency, not contribution
magnitude. SHAP provides signed, per-prediction explanations required
for regulatory compliance (GDPR Article 22, EU AI Act).

## Tech Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP`
`FastAPI` `Streamlit` `Plotly` `Docker` `GitHub Actions`
`Render` `Hugging Face Spaces`

## Author

**Shresth** — [GitHub](https://github.com/shresthbhargava) · [LinkedIn](https://linkedin.com/in/shresth-bhargava/)
