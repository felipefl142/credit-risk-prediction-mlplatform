"""
Credit Risk Assessment Dashboard — Streamlit application.

Usage:
    streamlit run app/streamlit_dashboard.py

Expects model artifacts in the `models/` directory and gold layer data
in `data/features/`.
"""

import json
import pickle
from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk ML Platform",
    page_icon=":bar_chart:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the XGBoost model from disk."""
    model_path = MODELS_DIR / "best_xgb.json"
    if not model_path.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


@st.cache_resource
def load_preprocessor():
    """Load the fitted OrdinalEncoder."""
    path = MODELS_DIR / "preprocessor.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_columns():
    path = MODELS_DIR / "feature_columns.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_categorical_columns():
    path = MODELS_DIR / "categorical_columns.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_training_data():
    path = FEATURES_DIR / "train_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_risk(probability: float) -> tuple[str, str]:
    """Return (risk_level, color)."""
    if probability < 0.10:
        return "Low", "#28a745"
    elif probability < 0.25:
        return "Medium", "#ffc107"
    elif probability < 0.50:
        return "High", "#fd7e14"
    return "Very High", "#dc3545"


def derive_features(row: dict) -> dict:
    """Compute engineered features when they are not provided."""
    income = row.get("AMT_INCOME_TOTAL") or 1.0
    credit = row.get("AMT_CREDIT") or 0.0
    annuity = row.get("AMT_ANNUITY") or 0.0
    children = row.get("CNT_CHILDREN", 0)
    fam = row.get("CNT_FAM_MEMBERS", 1) or 1

    row.setdefault("CREDIT_INCOME_RATIO", credit / income if income else np.nan)
    row.setdefault("ANNUITY_INCOME_RATIO", annuity / income if income else np.nan)
    row.setdefault("CREDIT_TERM", annuity / credit if credit else np.nan)
    row.setdefault("INCOME_PER_PERSON", income / fam if fam else np.nan)
    row.setdefault("CHILDREN_RATIO", children / fam if fam else np.nan)
    return row


def make_prediction(row: dict, model, preprocessor, feature_cols, cat_cols):
    """Run a single prediction through the model."""
    row = derive_features(row)
    df = pd.DataFrame([row])

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_cols]

    if preprocessor is not None and cat_cols:
        cat_present = [c for c in cat_cols if c in df.columns]
        df[cat_present] = df[cat_present].fillna("missing")
        df[cat_present] = preprocessor.transform(df[cat_present])

    probability = float(model.predict(xgb.DMatrix(df.astype(float)))[0])
    return probability


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Credit Risk ML Platform")
page = st.sidebar.radio(
    "Navigate",
    ["Model Overview", "Make Prediction", "Data Explorer"],
)

# ---------------------------------------------------------------------------
# PAGE: Model Overview
# ---------------------------------------------------------------------------
if page == "Model Overview":
    st.title("Model Overview")
    st.markdown("Performance metrics and feature importances for the production credit-risk model.")

    metrics = load_metrics()
    model = load_model()

    # --- Metrics row ---
    if metrics:
        st.subheader("Validation Metrics")
        cols = st.columns(4)
        metric_display = [
            ("AUC-ROC", metrics.get("auc_roc", "N/A")),
            ("PR-AUC", metrics.get("pr_auc", "N/A")),
            ("F1 Score", metrics.get("f1", "N/A")),
            ("Accuracy", metrics.get("accuracy", "N/A")),
        ]
        for col, (label, value) in zip(cols, metric_display):
            if isinstance(value, (int, float)):
                col.metric(label, f"{value:.4f}")
            else:
                col.metric(label, value)
    else:
        st.info(
            "No saved metrics found at `models/metrics.json`. "
            "Run notebooks 05-08 to generate metrics, or the metrics will be "
            "computed from the model artifacts once available."
        )

    # --- Feature importance ---
    st.subheader("Feature Importance (Top 25)")
    if model is not None:
        scores = model.get_score(importance_type="gain")
        imp_df = (
            pd.DataFrame(list(scores.items()), columns=["feature", "importance"])
            .sort_values("importance", ascending=False)
            .head(25)
        )

        fig = px.bar(
            imp_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="XGBoost Feature Importance (gain)",
            labels={"importance": "Importance (gain)", "feature": "Feature"},
        )
        fig.update_layout(height=600, yaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model not found. Please ensure `models/best_xgb.json` exists.")

# ---------------------------------------------------------------------------
# PAGE: Make Prediction
# ---------------------------------------------------------------------------
elif page == "Make Prediction":
    st.title("Make a Prediction")
    st.markdown("Enter loan application details to get a default-risk assessment.")

    model = load_model()
    preprocessor = load_preprocessor()
    feature_cols = load_feature_columns()
    cat_cols = load_categorical_columns()

    if model is None:
        st.error("Model not loaded. Ensure `models/best_xgb.json` exists.")
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Applicant Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            amt_income = st.number_input("Total Income", value=202500.0, min_value=0.0, step=1000.0)
            amt_credit = st.number_input("Credit Amount", value=406597.5, min_value=0.0, step=1000.0)
            amt_annuity = st.number_input("Annuity", value=24700.5, min_value=0.0, step=100.0)
            amt_goods = st.number_input("Goods Price", value=351000.0, min_value=0.0, step=1000.0)
            age = st.number_input("Age (years)", value=35.0, min_value=18.0, max_value=80.0, step=1.0)

        with col2:
            contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
            gender = st.selectbox("Gender", ["M", "F"])
            own_car = st.selectbox("Owns Car", ["Y", "N"])
            own_realty = st.selectbox("Owns Realty", ["Y", "N"])
            education = st.selectbox(
                "Education",
                [
                    "Higher education",
                    "Secondary / secondary special",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
            )

        with col3:
            family_status = st.selectbox(
                "Family Status",
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
            )
            income_type = st.selectbox(
                "Income Type",
                ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
            )
            children = st.number_input("Number of Children", value=0, min_value=0, max_value=20, step=1)
            fam_members = st.number_input("Family Members", value=2, min_value=1, max_value=20, step=1)
            days_employed = st.number_input("Days Employed (negative)", value=-3000.0, step=100.0)

        st.subheader("External Scores (optional)")
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            ext1 = st.number_input("EXT_SOURCE_1", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
        with ecol2:
            ext2 = st.number_input("EXT_SOURCE_2", value=0.6, min_value=0.0, max_value=1.0, step=0.01)
        with ecol3:
            ext3 = st.number_input("EXT_SOURCE_3", value=0.5, min_value=0.0, max_value=1.0, step=0.01)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": amt_goods,
            "AGE_YEARS": age,
            "NAME_CONTRACT_TYPE": contract_type,
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": own_realty,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_INCOME_TYPE": income_type,
            "CNT_CHILDREN": children,
            "CNT_FAM_MEMBERS": fam_members,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
        }

        probability = make_prediction(row, model, preprocessor, feature_cols, cat_cols)
        risk_level, color = classify_risk(probability)

        st.divider()

        # Results
        r1, r2, r3 = st.columns(3)
        r1.metric("Default Probability", f"{probability:.2%}")
        r2.metric("Prediction", "DEFAULT" if probability >= 0.5 else "NO DEFAULT")
        r3.metric("Risk Level", risk_level)

        # Probability gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Default Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 10], "color": "#d4edda"},
                        {"range": [10, 25], "color": "#fff3cd"},
                        {"range": [25, 50], "color": "#ffe0b2"},
                        {"range": [50, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # SHAP explanation (if shap is available)
        try:
            import shap

            row_derived = derive_features(row.copy())
            df_row = pd.DataFrame([row_derived])
            for col in feature_cols:
                if col not in df_row.columns:
                    df_row[col] = np.nan
            df_row = df_row[feature_cols]
            if preprocessor is not None and cat_cols:
                cat_present = [c for c in cat_cols if c in df_row.columns]
                df_row[cat_present] = df_row[cat_present].fillna("missing")
                df_row[cat_present] = preprocessor.transform(df_row[cat_present])

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_row)

            st.subheader("SHAP Explanation")
            import matplotlib.pyplot as plt

            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=df_row.iloc[0].values,
                    feature_names=feature_cols,
                ),
                max_display=15,
                show=False,
            )
            st.pyplot(fig_shap)
            plt.close(fig_shap)

        except ImportError:
            st.info("Install `shap` to see feature-level explanations: `pip install shap`")
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

# ---------------------------------------------------------------------------
# PAGE: Data Explorer
# ---------------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("Browse the training data and inspect distributions.")

    df = load_training_data()

    if df is None:
        st.error("Training data not found at `data/features/train_features.parquet`.")
        st.stop()

    st.subheader("Dataset Overview")
    ocol1, ocol2, ocol3, ocol4 = st.columns(4)
    ocol1.metric("Rows", f"{df.shape[0]:,}")
    ocol2.metric("Columns", f"{df.shape[1]}")
    ocol3.metric("Default Rate", f"{df['TARGET'].mean()*100:.2f}%")
    ocol4.metric("Missing Cells", f"{df.isnull().sum().sum():,}")

    st.subheader("Sample Data")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

    # Distribution plot for a selected numeric column
    st.subheader("Feature Distribution")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("SK_ID_CURR", "TARGET")]

    selected_col = st.selectbox("Select feature", numeric_cols)
    if selected_col:
        fig = px.histogram(
            df,
            x=selected_col,
            color="TARGET",
            nbins=60,
            barmode="overlay",
            opacity=0.6,
            title=f"Distribution of {selected_col} by Target",
            color_discrete_map={0: "#636EFA", 1: "#EF553B"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap of top features
    st.subheader("Top Feature Correlations with Target")
    corr_with_target = (
        df[numeric_cols + ["TARGET"]]
        .corr()["TARGET"]
        .drop("TARGET", errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(15)
    )
    fig_corr = px.bar(
        x=corr_with_target.values,
        y=corr_with_target.index,
        orientation="h",
        title="Top 15 Features by |Correlation| with Target",
        labels={"x": "|Correlation|", "y": "Feature"},
    )
    fig_corr.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_corr, use_container_width=True)
