# Credit Risk Prediction - ML Platform

End-to-end Machine Learning platform for credit risk prediction, built as a portfolio project showcasing modern MLOps practices: data lake patterns, feature stores, experiment tracking, model serving, and interactive dashboards.

**Task**: Binary classification -- predict whether a loan applicant will default on their credit.

## Data Source

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) -- Kaggle competition dataset.

- 7 relational tables (~1.5 GB raw)
- 307,511 loan applications
- ~8% positive class (defaults) -- significant class imbalance

Download (requires `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
```

## Architecture

### Data Lake (Local Parquet-based)

| Layer | Path | Description |
|-------|------|-------------|
| Bronze | `data/raw/` | Raw CSVs from Kaggle |
| Silver | `data/processed/` | Cleaned, typed Parquet files |
| Gold | `data/features/` | Feature-engineered datasets |

### Tool Stack

| Component | Tool |
|-----------|------|
| Data lake | Parquet + directory layering (bronze / silver / gold) |
| Feature store | Feast (local file-based offline store + SQLite online store) |
| Experiment tracking | MLFlow (local tracking + SQLite model registry) |
| Training | scikit-learn, LightGBM, XGBoost, CatBoost |
| Explainability | SHAP |
| Hyperparameter tuning | Optuna |
| Model serving | FastAPI + MLFlow model loading |
| Dashboard | Streamlit |

## Notebook Sequence

Notebooks must be run in order. Each builds on artifacts from the previous step.

| # | Notebook | Description | Status |
|---|----------|-------------|--------|
| 01 | `01_data_ingestion.ipynb` | Data ingestion & data lake setup (CSV to Parquet bronze/silver layers) | Done |
| 02 | `02_eda.ipynb` | Exploratory data analysis -- target distribution, missing values, correlations, categorical/numeric analysis, bureau data exploration | Done |
| 03 | `03_feature_engineering.ipynb` | Feature engineering -- bureau/balance/previous app/POS/credit card/installment aggregations, domain features, binary `_IS_MISSING` indicators | Done |
| 04 | `04_feature_store.ipynb` | Feature store setup with Feast -- entity/feature view registration, materialization, historical & online retrieval | Done |
| 05 | `05_baseline_models.ipynb` | Baseline models (LogReg, Random Forest) + first MLFlow experiment | Planned |
| 06 | `06_model_selection.ipynb` | Model selection & comparison (LightGBM, XGBoost, CatBoost) + SHAP | Planned |
| 07 | `07_hyperparameter_tuning.ipynb` | Hyperparameter tuning with Optuna + MLFlow | Planned |
| 08 | `08_evaluation.ipynb` | Evaluation & insights -- SHAP, error analysis, business metrics | Planned |
| 09 | `09_deployment.ipynb` | Model registry & deployment prep | Planned |

## Project Structure

```
credit-risk-mlplatform/
├── app/
│   └── streamlit_dashboard.py      # Interactive dashboard
├── data/
│   ├── raw/                         # Bronze layer (CSV)
│   ├── processed/                   # Silver layer (Parquet)
│   ├── features/                    # Gold layer (Parquet)
│   └── feast/                       # Feast data + registry
├── feature_store/
│   ├── feature_store.yaml           # Feast repo config
│   └── definitions.py              # Entity & feature view definitions
├── figures/                         # EDA plots from NB02
├── mlruns/                          # MLFlow tracking store
├── models/                          # Serialized models
├── notebooks/                       # Jupyter notebooks (01-09)
├── serving/
│   └── api.py                       # FastAPI scoring endpoint
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
kaggle competitions download -c home-credit-default-risk -p data/raw

# Run notebooks
jupyter lab
```

## Key Design Decisions

- **No data leakage**: all features are derived strictly from data available at application time
- **Missingness as signal**: binary `_IS_MISSING` flags for the 6 most relevant features (>5% missing rate) -- missingness in credit data often correlates with default risk
- **Stratified splits**: all train/test splits preserve the ~8% positive class ratio
- **Metrics**: AUC-ROC and PR-AUC as primary metrics given class imbalance
