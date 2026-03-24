"""
FastAPI serving endpoint for Credit Risk prediction model.

Usage:
    uvicorn serving.api:app --reload

Expects model artifacts in the `models/` directory:
    - best_lgbm.txt          (LightGBM model)
    - preprocessor.pkl        (fitted OrdinalEncoder)
    - feature_columns.json    (ordered feature column list)
    - categorical_columns.json (categorical column names)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths (relative to project root — works when launched from project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class LoanApplication(BaseModel):
    """Input schema for a single loan application."""

    # Core financial features
    AMT_INCOME_TOTAL: float = Field(..., description="Total income of the applicant")
    AMT_CREDIT: float = Field(..., description="Credit amount of the loan")
    AMT_ANNUITY: float = Field(..., description="Loan annuity")
    AMT_GOODS_PRICE: float = Field(..., description="Price of the goods for which the loan is given")

    # Demographic features
    AGE_YEARS: float = Field(..., description="Age in years")
    NAME_CONTRACT_TYPE: str = Field(..., description="Type of loan contract (Cash loans / Revolving loans)")
    CODE_GENDER: str = Field(..., description="Gender (M / F)")
    FLAG_OWN_CAR: str = Field(..., description="Owns a car (Y / N)")
    FLAG_OWN_REALTY: str = Field(..., description="Owns real estate (Y / N)")
    NAME_EDUCATION_TYPE: str = Field(..., description="Education level")
    NAME_FAMILY_STATUS: str = Field(..., description="Family status")
    NAME_INCOME_TYPE: str = Field(..., description="Income type (Working, Commercial associate, etc.)")

    # Family
    CNT_CHILDREN: int = Field(0, description="Number of children")
    CNT_FAM_MEMBERS: int = Field(1, description="Number of family members")

    # Employment / external scores (optional)
    DAYS_EMPLOYED: Optional[float] = Field(None, description="Days employed (negative = before application)")
    EXT_SOURCE_1: Optional[float] = Field(None, description="External source score 1")
    EXT_SOURCE_2: Optional[float] = Field(None, description="External source score 2")
    EXT_SOURCE_3: Optional[float] = Field(None, description="External source score 3")

    # Additional optional features — default to None, filled as NaN
    REGION_POPULATION_RELATIVE: Optional[float] = None
    DAYS_REGISTRATION: Optional[float] = None
    DAYS_ID_PUBLISH: Optional[float] = None
    OWN_CAR_AGE: Optional[float] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    REGION_RATING_CLIENT: Optional[int] = None
    REGION_RATING_CLIENT_W_CITY: Optional[int] = None
    HOUR_APPR_PROCESS_START: Optional[int] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    DAYS_EMPLOYED_ANOMALY: Optional[int] = None

    # Engineered features (optional — will be derived if missing)
    CREDIT_INCOME_RATIO: Optional[float] = None
    ANNUITY_INCOME_RATIO: Optional[float] = None
    CREDIT_TERM: Optional[float] = None
    INCOME_PER_PERSON: Optional[float] = None
    CHILDREN_RATIO: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "AGE_YEARS": 35.0,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_INCOME_TYPE": "Working",
                "CNT_CHILDREN": 0,
                "CNT_FAM_MEMBERS": 2,
                "DAYS_EMPLOYED": -3000.0,
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.5,
            }
        }


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Probability of default (0-1)")
    prediction: int = Field(..., description="Binary prediction (0 = no default, 1 = default)")
    risk_level: str = Field(..., description="Risk category (Low / Medium / High / Very High)")


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    feature_columns: list[str]
    categorical_columns: list[str]
    artifacts_path: str


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_state: dict = {}


def _classify_risk(probability: float) -> str:
    """Map default probability to a human-readable risk level."""
    if probability < 0.10:
        return "Low"
    elif probability < 0.25:
        return "Medium"
    elif probability < 0.50:
        return "High"
    return "Very High"


def _derive_features(row: dict) -> dict:
    """Compute engineered features when they are not provided."""
    income = row.get("AMT_INCOME_TOTAL") or 1.0
    credit = row.get("AMT_CREDIT") or 0.0
    annuity = row.get("AMT_ANNUITY") or 0.0
    goods = row.get("AMT_GOODS_PRICE") or 0.0
    children = row.get("CNT_CHILDREN") or 0
    fam = row.get("CNT_FAM_MEMBERS") or 1

    if row.get("CREDIT_INCOME_RATIO") is None:
        row["CREDIT_INCOME_RATIO"] = credit / income if income else np.nan
    if row.get("ANNUITY_INCOME_RATIO") is None:
        row["ANNUITY_INCOME_RATIO"] = annuity / income if income else np.nan
    if row.get("CREDIT_TERM") is None:
        row["CREDIT_TERM"] = annuity / credit if credit else np.nan
    if row.get("INCOME_PER_PERSON") is None:
        row["INCOME_PER_PERSON"] = income / fam if fam else np.nan
    if row.get("CHILDREN_RATIO") is None:
        row["CHILDREN_RATIO"] = children / fam if fam else np.nan

    return row


# ---------------------------------------------------------------------------
# Lifespan — load model & artifacts once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load LightGBM model
    model_path = MODELS_DIR / "best_lgbm.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    _state["model"] = lgb.Booster(model_file=str(model_path))

    # Load preprocessor (OrdinalEncoder)
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if preprocessor_path.exists():
        with open(preprocessor_path, "rb") as f:
            _state["preprocessor"] = pickle.load(f)
    else:
        _state["preprocessor"] = None

    # Load feature columns
    feature_cols_path = MODELS_DIR / "feature_columns.json"
    with open(feature_cols_path) as f:
        _state["feature_columns"] = json.load(f)

    # Load categorical columns
    cat_cols_path = MODELS_DIR / "categorical_columns.json"
    with open(cat_cols_path) as f:
        _state["categorical_columns"] = json.load(f)

    print(f"Model loaded from {model_path}")
    print(f"Features: {len(_state['feature_columns'])} total, {len(_state['categorical_columns'])} categorical")

    yield

    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict the probability that a loan applicant will default.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return model metadata."""
    return ModelInfoResponse(
        model_type="LightGBM",
        n_features=len(_state["feature_columns"]),
        feature_columns=_state["feature_columns"],
        categorical_columns=_state["categorical_columns"],
        artifacts_path=str(MODELS_DIR),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    """Score a single loan application."""
    try:
        # Convert to dict and derive any missing engineered features
        row = application.model_dump()
        row = _derive_features(row)

        # Build a single-row DataFrame with the exact columns the model expects
        feature_cols = _state["feature_columns"]
        cat_cols = _state["categorical_columns"]
        preprocessor = _state["preprocessor"]

        df = pd.DataFrame([row])

        # Ensure all expected columns are present (fill missing with NaN)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[feature_cols]

        # Preprocess categorical columns
        if preprocessor is not None and cat_cols:
            cat_present = [c for c in cat_cols if c in df.columns]
            # Fill NaN for categoricals before encoding
            df[cat_present] = df[cat_present].fillna("missing")
            df[cat_present] = preprocessor.transform(df[cat_present])

        # Predict
        model = _state["model"]
        probability = float(model.predict(df)[0])
        prediction = int(probability >= 0.5)
        risk_level = _classify_risk(probability)

        return PredictionResponse(
            probability=round(probability, 6),
            prediction=prediction,
            risk_level=risk_level,
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
