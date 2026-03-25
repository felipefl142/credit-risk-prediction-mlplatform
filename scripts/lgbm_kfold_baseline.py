"""
Home Credit Default Risk — LightGBM KFold Baseline
===================================================
Feature engineering across all 7 tables + KFold LightGBM training.

Key ideas:
- Divide/subtract important features to get rates (annuity/income, etc.)
- Bureau: separate aggregations for Active vs Closed credits
- Previous Applications: separate aggregations for Approved vs Refused
- One-hot encoding for all categorical features
- KFold (or StratifiedKFold) cross-validation with LightGBM

Usage (from project root):
    source .venv/bin/activate
    python scripts/lgbm_kfold_baseline.py           # full run
    python scripts/lgbm_kfold_baseline.py --debug   # 10 000-row smoke test
"""

import argparse
import gc
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SUBMISSION_FILE = MODELS / "lgbm_kfold_submission.csv"
IMPORTANCE_PLOT = FIGURES / "08_lgbm_feature_importance.png"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@contextmanager
def timer(title: str):
    t0 = time.time()
    yield
    print(f"{title} — done in {time.time() - t0:.0f}s")


def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True):
    original_columns = list(df.columns)
    categorical_columns = [c for c in df.columns if df[c].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    # pandas 2.x returns bool dtype for dummies; cast to int8 for LightGBM compatibility
    bool_cols = [c for c in new_columns if df[c].dtype == bool]
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(np.int8)
    return df, new_columns


# ---------------------------------------------------------------------------
# Table processors
# ---------------------------------------------------------------------------

def application_train_test(num_rows=None, nan_as_category=False):
    df = pd.read_csv(DATA_RAW / "application_train.csv", nrows=num_rows)
    test_df = pd.read_csv(DATA_RAW / "application_test.csv", nrows=num_rows)
    print(f"Train samples: {len(df)}, test samples: {len(test_df)}")

    df = pd.concat([df, test_df], ignore_index=True)

    # Remove 4 applications with XNA CODE_GENDER (train set only)
    df = df[df["CODE_GENDER"] != "XNA"]

    # Binary-encode two-category features
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col], _ = pd.factorize(df[col])

    df, _ = one_hot_encoder(df, nan_as_category)
    df = df.copy()  # defragment after get_dummies

    # DAYS_EMPLOYED sentinel → NaN
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    # Rate features
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    del test_df
    gc.collect()
    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv(DATA_RAW / "bureau.csv", nrows=num_rows)
    bb = pd.read_csv(DATA_RAW / "bureau_balance.csv", nrows=num_rows)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # bureau_balance → aggregate per SK_ID_BUREAU, then join
    bb_agg_spec = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_agg_spec[col] = ["mean"]
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_agg_spec)
    bb_agg.columns = pd.Index([f"{e[0]}_{e[1].upper()}" for e in bb_agg.columns])
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    num_agg = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    cat_agg = {cat: ["mean"] for cat in bureau_cat}
    for cat in bb_cat:
        cat_agg[f"{cat}_MEAN"] = ["mean"]

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_agg, **cat_agg})
    bureau_agg.columns = pd.Index(
        [f"BURO_{e[0]}_{e[1].upper()}" for e in bureau_agg.columns]
    )

    # Active credits
    active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_agg)
    active_agg.columns = pd.Index(
        [f"ACTIVE_{e[0]}_{e[1].upper()}" for e in active_agg.columns]
    )
    bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
    del active, active_agg
    gc.collect()

    # Closed credits
    closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_agg)
    closed_agg.columns = pd.Index(
        [f"CLOSED_{e[0]}_{e[1].upper()}" for e in closed_agg.columns]
    )
    bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv(DATA_RAW / "previous_application.csv", nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    prev = prev.copy()  # defragment after get_dummies

    for col in [
        "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE", "DAYS_TERMINATION",
    ]:
        prev[col].replace(365243, np.nan, inplace=True)

    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    num_agg = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    cat_agg = {cat: ["mean"] for cat in cat_cols}

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_agg, **cat_agg})
    prev_agg.columns = pd.Index(
        [f"PREV_{e[0]}_{e[1].upper()}" for e in prev_agg.columns]
    )

    # Approved applications
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("SK_ID_CURR").agg(num_agg)
    approved_agg.columns = pd.Index(
        [f"APPROVED_{e[0]}_{e[1].upper()}" for e in approved_agg.columns]
    )
    prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")

    # Refused applications
    refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("SK_ID_CURR").agg(num_agg)
    refused_agg.columns = pd.Index(
        [f"REFUSED_{e[0]}_{e[1].upper()}" for e in refused_agg.columns]
    )
    prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")

    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv(DATA_RAW / "POS_CASH_balance.csv", nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    agg = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for cat in cat_cols:
        agg[cat] = ["mean"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(agg)
    pos_agg.columns = pd.Index(
        [f"POS_{e[0]}_{e[1].upper()}" for e in pos_agg.columns]
    )
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    del pos
    gc.collect()
    return pos_agg


def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv(DATA_RAW / "installments_payments.csv", nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)

    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    agg = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for cat in cat_cols:
        agg[cat] = ["mean"]

    ins_agg = ins.groupby("SK_ID_CURR").agg(agg)
    ins_agg.columns = pd.Index(
        [f"INSTAL_{e[0]}_{e[1].upper()}" for e in ins_agg.columns]
    )
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
    del ins
    gc.collect()
    return ins_agg


def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv(DATA_RAW / "credit_card_balance.csv", nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)

    cc_agg = cc.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = pd.Index(
        [f"CC_{e[0]}_{e[1].upper()}" for e in cc_agg.columns]
    )
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    del cc
    gc.collect()
    return cc_agg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def display_importances(feature_importance_df: pd.DataFrame):
    cols = (
        feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values("importance", ascending=False)
        .head(40)
        .index
    )
    best = feature_importance_df[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance", y="feature",
        data=best.sort_values("importance", ascending=False),
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT)
    print(f"Feature importance plot saved → {IMPORTANCE_PLOT}")


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace special JSON characters in column names that LightGBM rejects."""
    import re
    df.columns = [re.sub(r'[{}\[\]":,\s]+', "_", c).strip("_") for c in df.columns]
    return df


def kfold_lightgbm(df: pd.DataFrame, num_folds: int, stratified: bool = False, debug: bool = False):
    df = _sanitize_columns(df)
    train_df = df[df["TARGET"].notnull()]
    test_df = df[df["TARGET"].isnull()]
    print(f"Starting LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}")
    del df
    gc.collect()

    folds = (
        StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        if stratified
        else KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    )

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    exclude = {"TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"}
    feats = [f for f in train_df.columns if f not in exclude]

    mlflow.set_experiment("lgbm_kfold_baseline")
    with mlflow.start_run(run_name=f"kfold_{num_folds}{'_strat' if stratified else ''}"):
        mlflow.log_params({
            "num_folds": num_folds,
            "stratified": stratified,
            "n_features": len(feats),
            "debug": debug,
        })

        for n_fold, (train_idx, valid_idx) in enumerate(
            folds.split(train_df[feats], train_df["TARGET"])
        ):
            train_x = train_df[feats].iloc[train_idx]
            train_y = train_df["TARGET"].iloc[train_idx]
            valid_x = train_df[feats].iloc[valid_idx]
            valid_y = train_df["TARGET"].iloc[valid_idx]

            clf = LGBMClassifier(
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=34,
                colsample_bytree=0.9497036,
                subsample=0.8715623,
                max_depth=8,
                reg_alpha=0.041545473,
                reg_lambda=0.0735294,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                device="gpu",
                n_jobs=4,
                verbose=-1,
            )

            clf.fit(
                train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric="auc",
                callbacks=[
                    early_stopping(stopping_rounds=200, verbose=False),
                    log_evaluation(period=200),
                ],
            )

            oof_preds[valid_idx] = clf.predict_proba(
                valid_x, num_iteration=clf.best_iteration_
            )[:, 1]
            sub_preds += (
                clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1]
                / folds.n_splits
            )

            fold_imp = pd.DataFrame({
                "feature": feats,
                "importance": clf.feature_importances_,
                "fold": n_fold + 1,
            })
            feature_importance_df = pd.concat([feature_importance_df, fold_imp], axis=0)

            fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
            print(f"Fold {n_fold + 1:2d} AUC: {fold_auc:.6f}")
            mlflow.log_metric("fold_auc", fold_auc, step=n_fold + 1)

            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        full_auc = roc_auc_score(train_df["TARGET"], oof_preds)
        print(f"Full OOF AUC: {full_auc:.6f}")
        mlflow.log_metric("oof_auc", full_auc)

        if not debug:
            test_df["TARGET"] = sub_preds
            test_df[["SK_ID_CURR", "TARGET"]].to_csv(SUBMISSION_FILE, index=False)
            print(f"Submission saved → {SUBMISSION_FILE}")
            mlflow.log_artifact(str(SUBMISSION_FILE))

    display_importances(feature_importance_df)
    mlflow.log_artifact(str(IMPORTANCE_PLOT))
    return feature_importance_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(debug: bool = False):
    num_rows = 10_000 if debug else None

    with timer("Process application train+test"):
        df = application_train_test(num_rows)
        print(f"Application df shape: {df.shape}")

    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print(f"Bureau df shape: {bureau.shape}")
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()

    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print(f"Previous applications df shape: {prev.shape}")
        df = df.join(prev, how="left", on="SK_ID_CURR")
        del prev
        gc.collect()

    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print(f"Pos-cash balance df shape: {pos.shape}")
        df = df.join(pos, how="left", on="SK_ID_CURR")
        del pos
        gc.collect()

    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print(f"Installments payments df shape: {ins.shape}")
        df = df.join(ins, how="left", on="SK_ID_CURR")
        del ins
        gc.collect()

    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print(f"Credit card balance df shape: {cc.shape}")
        df = df.join(cc, how="left", on="SK_ID_CURR")
        del cc
        gc.collect()

    with timer("Run LightGBM with KFold"):
        kfold_lightgbm(df, num_folds=10, stratified=False, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM KFold baseline for Home Credit")
    parser.add_argument(
        "--debug", action="store_true",
        help="Run with 10 000 rows per table for a quick smoke test",
    )
    args = parser.parse_args()

    with timer("Full model run"):
        main(debug=args.debug)
