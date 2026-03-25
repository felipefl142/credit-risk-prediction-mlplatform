"""
Feast feature definitions for the Credit Risk ML Platform.

Defines entities, data sources, and feature views for serving
engineered features from the gold layer.
"""

from datetime import timedelta

from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
applicant = Entity(
    name="applicant",
    join_keys=["SK_ID_CURR"],
    description="Loan applicant identified by SK_ID_CURR",
)

# ---------------------------------------------------------------------------
# Data source  (Feast-prepared parquet with event_timestamp column)
# ---------------------------------------------------------------------------
train_source = FileSource(
    name="train_features_source",
    path="../data/feast/train_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

# 1. Core application features
applicant_features = FeatureView(
    name="applicant_features",
    entities=[applicant],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="AMT_INCOME_TOTAL", dtype=Float32),
        Field(name="AMT_CREDIT", dtype=Float32),
        Field(name="AMT_ANNUITY", dtype=Float32),
        Field(name="AMT_GOODS_PRICE", dtype=Float32),
        Field(name="AGE_YEARS", dtype=Float32),
        Field(name="DAYS_EMPLOYED", dtype=Float32),
        Field(name="DAYS_REGISTRATION", dtype=Float32),
        Field(name="DAYS_ID_PUBLISH", dtype=Float32),
        Field(name="DAYS_EMPLOYED_ANOMALY", dtype=Int64),
        Field(name="EXT_SOURCE_1", dtype=Float32),
        Field(name="EXT_SOURCE_2", dtype=Float32),
        Field(name="EXT_SOURCE_3", dtype=Float32),
    ],
    source=train_source,
    online=True,
)

# 2. Bureau aggregation features (from credit bureau data)
bureau_features = FeatureView(
    name="bureau_features",
    entities=[applicant],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="BUREAU_CREDIT_COUNT", dtype=Float32),
        Field(name="BUREAU_AMT_CREDIT_SUM_SUM", dtype=Float32),
        Field(name="BUREAU_AMT_CREDIT_SUM_MEAN", dtype=Float32),
        Field(name="BUREAU_AMT_CREDIT_SUM_DEBT_SUM", dtype=Float32),
        Field(name="BUREAU_AMT_CREDIT_SUM_DEBT_MEAN", dtype=Float32),
        Field(name="BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN", dtype=Float32),
        Field(name="BUREAU_DAYS_CREDIT_MEAN", dtype=Float32),
        Field(name="BUREAU_DAYS_CREDIT_MIN", dtype=Float32),
        Field(name="BUREAU_DAYS_CREDIT_MAX", dtype=Float32),
        Field(name="BUREAU_DAYS_CREDIT_ENDDATE_MEAN", dtype=Float32),
        Field(name="BUREAU_OVERDUE_CREDIT_PROPORTION", dtype=Float32),
    ],
    source=train_source,
    online=True,
)

# 3. Credit history features (previous applications, POS, credit card, installments)
credit_history_features = FeatureView(
    name="credit_history_features",
    entities=[applicant],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="PREV_APPLICATION_COUNT", dtype=Float32),
        Field(name="PREV_AMT_APPLICATION_MEAN", dtype=Float32),
        Field(name="PREV_AMT_CREDIT_MEAN", dtype=Float32),
        Field(name="PREV_STATUS_APPROVED_COUNT", dtype=Float32),
        Field(name="PREV_STATUS_REFUSED_COUNT", dtype=Float32),
        Field(name="PREV_APPROVAL_RATE", dtype=Float32),
        Field(name="POS_MONTHS_BALANCE_MEAN", dtype=Float32),
        Field(name="POS_SK_DPD_MAX", dtype=Float32),
        Field(name="POS_SK_DPD_DEF_MAX", dtype=Float32),
        Field(name="POS_COMPLETED_COUNT", dtype=Float32),
        Field(name="CC_AMT_BALANCE_MEAN", dtype=Float32),
        Field(name="CC_AMT_BALANCE_MAX", dtype=Float32),
        Field(name="CC_AMT_DRAWINGS_CURRENT_MEAN", dtype=Float32),
        Field(name="CC_AMT_PAYMENT_CURRENT_MEAN", dtype=Float32),
        Field(name="INST_AMT_PAYMENT_MEAN", dtype=Float32),
        Field(name="INST_AMT_INSTALMENT_MEAN", dtype=Float32),
        Field(name="INST_PAYMENT_DIFF_MEAN", dtype=Float32),
        Field(name="INST_PAYMENT_DIFF_SUM", dtype=Float32),
    ],
    source=train_source,
    online=True,
)

