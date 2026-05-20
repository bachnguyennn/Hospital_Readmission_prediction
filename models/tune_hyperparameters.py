"""
models/tune_hyperparameters.py
------------------------------
Optuna hyperparameter tuning for XGBoost and LightGBM.

Usage:
    python models/tune_hyperparameters.py --model lgb --n-trials 50
    python models/tune_hyperparameters.py --model xgb --n-trials 50
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

ROOT_DIR      = Path(__file__).resolve().parents[1]
FEATURED_CSV  = ROOT_DIR / "data" / "processed" / "featured.csv"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_data():
    df = pd.read_csv(FEATURED_CSV).sort_values("encounter_order")
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]
    
    drop_cols = ["readmitted_lt30", "encounter_order"]
    X_train = train_df.drop(columns=drop_cols).values
    y_train = train_df["readmitted_lt30"].values
    X_test  = test_df.drop(columns=drop_cols).values
    y_test  = test_df["readmitted_lt30"].values
    
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return X_train, X_test, y_train, y_test, spw


def objective_lgb(trial, X_train, X_test, y_train, y_test, spw):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": spw,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    proba = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, proba)


def objective_xgb(trial, X_train, X_test, y_train, y_test, spw):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": spw,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    proba = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, proba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgb", "xgb"], required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test, spw = load_data()
    
    study = optuna.create_study(direction="maximize")
    if args.model == "lgb":
        study.optimize(lambda t: objective_lgb(t, X_train, X_test, y_train, y_test, spw),
                       n_trials=args.n_trials)
    else:
        study.optimize(lambda t: objective_xgb(t, X_train, X_test, y_train, y_test, spw),
                       n_trials=args.n_trials)
    
    log.info(f"\nBest trial: AUPRC = {study.best_value:.4f}")
    for k, v in study.best_params.items():
        log.info(f"  {k}: {v}")
        
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    study.trials_dataframe().to_csv(ARTIFACTS_DIR / f"optuna_{args.model}.csv", index=False)

if __name__ == "__main__":
    main()
