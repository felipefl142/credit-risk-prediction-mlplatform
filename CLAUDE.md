# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end ML platform for credit risk prediction using the **Home Credit Default Risk** Kaggle dataset. Binary classification: predict whether a loan applicant will default. The project covers data lake patterns, feature stores (Feast), experiment tracking (MLFlow), model deployment (FastAPI), and an interactive Streamlit dashboard.

## Commands

```bash
# Activate environment
source .venv/bin/activate

# Run notebooks
jupyter lab

# Start MLFlow UI (from project root)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Start FastAPI serving endpoint
uvicorn serving.api:app --reload

# Run Streamlit dashboard
streamlit run app/streamlit_dashboard.py

# Download dataset (requires ~/.kaggle/kaggle.json)
kaggle competitions download -c home-credit-default-risk -p data/raw
```

## Architecture

### Data Lake Layers (Local Parquet-based)

- **Bronze** (`data/raw/`): Raw CSVs from Kaggle (7 tables, ~1.5 GB)
- **Silver** (`data/processed/`): Cleaned, typed Parquet files
- **Gold** (`data/features/`): Feature-engineered datasets, Feast-materialized

### Tool Stack

| Component | Tool |
|-----------|------|
| Data lake | Parquet + directory layering (bronze/silver/gold) |
| Feature store | Feast (local file-based offline store) |
| Experiment tracking | MLFlow (local tracking + SQLite registry) |
| Training | scikit-learn, LightGBM, XGBoost, CatBoost |
| Model serving | FastAPI + MLFlow model loading |
| Dashboard | Streamlit |

### Notebook Sequence (must run in order)

- **01**: Data ingestion & data lake setup (CSV → Parquet)
- **02**: Exploratory data analysis (EDA)
- **03**: Feature engineering (aggregations, domain features)
- **04**: Feature store setup with Feast (3 feature views: applicant, bureau, credit history)
- **05**: Baseline models (LogReg, RF) + first MLFlow experiment
- **06**: Model selection & comparison (LightGBM, XGBoost, CatBoost) + SHAP
- **07**: Hyperparameter tuning (Optuna + MLFlow)
- **08**: Evaluation & insights (SHAP, error analysis, business metrics)
- **09**: Model registry & deployment prep

### Key Files

- `feature_store/feature_store.yaml` — Feast repo config
- `feature_store/definitions.py` — Feature view & entity definitions
- `serving/api.py` — FastAPI scoring endpoint
- `app/streamlit_dashboard.py` — Interactive dashboard

## Key Technical Conventions

- **Target**: `TARGET` column (1 = default, 0 = no default)
- **Class imbalance**: ~8% positive class — use stratified splits, appropriate metrics (AUC-ROC, PR-AUC)
- **No leakage**: features derived only from data available at application time
- **Data formats**: raw as CSV, processed/features as Parquet
- **MLFlow tracking**: local file store at `mlruns/`, SQLite backend for model registry
- **Current progress**: NB01-NB07 complete, NB08+ planned
