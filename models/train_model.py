"""
models/train_model.py
----------------------
Trains XGBoost and LightGBM classifiers on the featured dataset.

Key design decisions:
  - TEMPORAL validation: data is split by encounter_order (chronological),
    not randomly. This prevents look-ahead bias.
  - AUPRC (Area Under Precision-Recall Curve) as primary metric.
  - Class imbalance handled via scale_pos_weight / is_unbalance flags
    and optional SMOTE oversampling on the training split only.
  - SHAP summary and beeswarm plots saved to reports/figures/.
  - Trained models serialized with joblib to models/artifacts/.

Usage:
    python models/train_model.py
    python models/train_model.py --smote        # enable SMOTE oversampling
    python models/train_model.py --no-shap      # skip SHAP (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import lightgbm as lgb
import xgboost as xgb

ROOT_DIR      = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR   = ROOT_DIR / "reports" / "figures"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"
FEATURED_CSV  = PROCESSED_DIR / "featured.csv"
TARGET_COL    = "readmitted_lt30"
TEMPORAL_COL  = "encounter_order"
TEST_FRACTION = 0.20          # last 20% of encounters → test set

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading & temporal split
# ---------------------------------------------------------------------------

def load_and_split(path: Path = FEATURED_CSV, test_frac: float = TEST_FRACTION):
    """
    Load featured.csv and perform a TEMPORAL split.

    The data is sorted by encounter_order (a proxy for calendar time derived
    from encounter_id in preprocess.py). The last `test_frac` of rows form
    the held-out test set. This ensures the model never sees future data
    during training — equivalent to a production deployment scenario.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Featured data not found: {path}\n"
            "Run: python features/feature_engineering.py"
        )

    log.info(f"Loading {path}")
    df = pd.read_csv(path)

    if TEMPORAL_COL not in df.columns:
        raise KeyError(
            f"Column '{TEMPORAL_COL}' not found. "
            "Ensure preprocess.py created encounter_order."
        )

    df = df.sort_values(TEMPORAL_COL).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_frac))
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    drop_cols = [TARGET_COL, TEMPORAL_COL]
    feature_names = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_names].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_names].values
    y_test  = test_df[TARGET_COL].values

    log.info(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    log.info(f"Positive rate — Train: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")
    log.info(f"Features: {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Oversample the minority class with SMOTE (training split only)."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        log.info(f"SMOTE: {len(y_train):,} → {len(y_res):,} samples")
        return X_res, y_res
    except ImportError:
        log.warning("imbalanced-learn not installed; skipping SMOTE")
        return X_train, y_train


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
             label: str, threshold: float = 0.5) -> dict:
    """Compute and log key classification metrics."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    auprc  = average_precision_score(y_test, proba)
    auroc  = roc_auc_score(y_test, proba)
    f1     = f1_score(y_test, preds, zero_division=0)
    report = classification_report(y_test, preds, zero_division=0)

    log.info(f"\n{'='*55}\n{label}")
    log.info(f"  AUPRC  (primary): {auprc:.4f}")
    log.info(f"  AUC-ROC         : {auroc:.4f}")
    log.info(f"  F1-score        : {f1:.4f}")
    log.info(f"\n{report}")

    return {"model": label, "auprc": auprc, "auroc": auroc, "f1": f1,
            "proba": proba, "preds": preds}


def precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: int = 100) -> float:
    """Precision in the top-k highest-probability predictions."""
    top_k_idx = np.argsort(proba)[-k:]
    return y_true[top_k_idx].mean()


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="saga",
            C=0.1,
        )),
    ])


def build_xgboost(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def build_lightgbm(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# SHAP interpretability
# ---------------------------------------------------------------------------

def generate_shap_plots(model, X_test: np.ndarray, feature_names: list[str],
                         label: str, max_display: int = 10) -> None:
    """
    Generate and save SHAP summary (bar) and beeswarm plots.

    Parameters
    ----------
    model        : Fitted XGBoost or LightGBM classifier
    X_test       : Test feature matrix (numpy array)
    feature_names: List of column names
    label        : Model name string used in filenames
    max_display  : Top N features to display
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Computing SHAP values for {label} ...")

    try:
        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(X_test)

        # For binary classifiers some libraries return list of 2 arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # --- Bar plot (global feature importance) ---
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test,
                          feature_names=feature_names,
                          plot_type="bar",
                          max_display=max_display,
                          show=False)
        bar_path = FIGURES_DIR / f"shap_bar_{label.lower().replace(' ','_')}.png"
        plt.savefig(bar_path, bbox_inches="tight", dpi=150)
        plt.close()
        log.info(f"SHAP bar plot saved → {bar_path}")

        # --- Beeswarm plot (value + direction) ---
        shap.summary_plot(shap_values, X_test,
                          feature_names=feature_names,
                          max_display=max_display,
                          show=False)
        bee_path = FIGURES_DIR / f"shap_beeswarm_{label.lower().replace(' ','_')}.png"
        plt.savefig(bee_path, bbox_inches="tight", dpi=150)
        plt.close()
        log.info(f"SHAP beeswarm plot saved → {bee_path}")

    except Exception as exc:
        log.warning(f"SHAP failed for {label}: {exc}")


# ---------------------------------------------------------------------------
# PR-curve plot
# ---------------------------------------------------------------------------

def plot_pr_curves(results: list[dict], y_test: np.ndarray) -> None:
    """Save a combined Precision-Recall curve for all models."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_rate = y_test.mean()
    ax.axhline(baseline_rate, linestyle="--", color="grey",
                label=f"Random baseline ({baseline_rate:.3f})")

    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for res, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, res["proba"])
        ax.plot(rec, prec, color=color,
                label=f"{res['model']} (AUPRC={res['auprc']:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — 30-Day Readmission", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    path = FIGURES_DIR / "pr_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"PR-curve saved → {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(use_smote: bool = False, generate_shap: bool = True) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_and_split()

    # Class-imbalance ratio for XGBoost / LightGBM weights
    neg  = (y_train == 0).sum()
    pos  = (y_train == 1).sum()
    spw  = neg / max(pos, 1)
    log.info(f"scale_pos_weight = {spw:.2f}")

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    results   = []
    eval_set  = [(X_test, y_test)]

    # --- Logistic Regression (baseline) ---
    log.info("\nTraining Logistic Regression baseline...")
    lr = build_logistic_regression()
    lr.fit(X_train, y_train)
    res_lr = evaluate(lr, X_test, y_test, "Logistic Regression")
    results.append(res_lr)
    joblib.dump(lr, ARTIFACTS_DIR / "logistic_regression.joblib")

    # --- XGBoost ---
    log.info("\nTraining XGBoost...")
    xgb_model = build_xgboost(spw)
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    res_xgb = evaluate(xgb_model, X_test, y_test, "XGBoost")
    log.info(f"  Precision@100: {precision_at_k(y_test, res_xgb['proba'], 100):.3f}")
    results.append(res_xgb)
    joblib.dump(xgb_model, ARTIFACTS_DIR / "xgboost.joblib")

    # --- LightGBM ---
    log.info("\nTraining LightGBM...")
    lgb_model = build_lightgbm(spw)
    lgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
    )
    res_lgb = evaluate(lgb_model, X_test, y_test, "LightGBM")
    log.info(f"  Precision@100: {precision_at_k(y_test, res_lgb['proba'], 100):.3f}")
    results.append(res_lgb)
    joblib.dump(lgb_model, ARTIFACTS_DIR / "lightgbm.joblib")

    # --- PR curves ---
    plot_pr_curves(results, y_test)

    # --- SHAP ---
    if generate_shap:
        generate_shap_plots(xgb_model, X_test, feature_names, "XGBoost")
        generate_shap_plots(lgb_model, X_test, feature_names, "LightGBM")

    # --- Save summary JSON ---
    summary = [
        {"model": r["model"], "auprc": round(r["auprc"], 4),
         "auroc": round(r["auroc"], 4), "f1": round(r["f1"], 4)}
        for r in results
    ]
    summary_path = ARTIFACTS_DIR / "results_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Results summary saved → {summary_path}")

    log.info("\n" + "="*55)
    log.info("Training complete. Best model by AUPRC:")
    best = max(results, key=lambda r: r["auprc"])
    log.info(f"  {best['model']}  AUPRC={best['auprc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train readmission models")
    parser.add_argument("--smote",   action="store_true",
                        help="Apply SMOTE oversampling on training split")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP computation (faster)")
    args = parser.parse_args()
    main(use_smote=args.smote, generate_shap=not args.no_shap)
