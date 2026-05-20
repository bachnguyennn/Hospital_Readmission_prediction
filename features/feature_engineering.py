"""
features/feature_engineering.py
---------------------------------
Derives domain-specific clinical features from the cleaned dataset.

New features engineered:
  - total_medications          : count of distinct prescribed medications
  - num_changed_medications    : medications with dose change (Up/Down)
  - num_lab_procedures_per_day : lab intensity relative to length of stay
  - num_procedures_per_day     : procedure intensity per day
  - has_primary_diabetes_diag  : whether primary diagnosis is diabetes
  - has_any_diabetes_diag      : any of diag_1/2/3 is diabetes
  - comorbidity_count          : number of distinct ICD-9 categories across diag_1/2/3
  - admission_complexity       : composite score (meds + labs + diagnoses)
  - is_elderly                 : age >= 65
  - insulin_dose_changed       : insulin was up- or down-titrated
  - a1c_tested                 : A1C was tested at this encounter
  - glucose_tested             : serum glucose was tested
  - age_x_num_medications      : interaction feature

Usage:
    python features/feature_engineering.py
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR      = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INPUT_CSV     = PROCESSED_DIR / "cleaned.csv"
OUTPUT_CSV    = PROCESSED_DIR / "featured.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MED_COLS = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
    "insulin","glyburide-metformin","glipizide-metformin",
    "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone",
]


def _present_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived clinical features in-place.

    Parameters
    ----------
    df : Cleaned DataFrame from preprocess.py

    Returns
    -------
    DataFrame with additional engineered feature columns.
    """
    present_meds = _present_cols(df, MED_COLS)

    # ------------------------------------------------------------------
    # Medication features
    # ------------------------------------------------------------------
    # Total distinct medications prescribed (any value > 0)
    df["total_medications"] = df[present_meds].gt(0).sum(axis=1).astype(np.int16)

    # Medications with a dose change (coded 2 in preprocess.py)
    df["num_changed_medications"] = df[present_meds].eq(2).sum(axis=1).astype(np.int16)

    # Insulin-specific flag
    if "insulin" in df.columns:
        df["insulin_dose_changed"] = (df["insulin"] == 2).astype(np.int8)

    # ------------------------------------------------------------------
    # Lab and procedure intensity
    # ------------------------------------------------------------------
    time_col = "time_in_hospital"
    days = df[time_col].clip(lower=1) if time_col in df.columns else pd.Series(1, index=df.index)

    if "num_lab_procedures" in df.columns:
        df["num_lab_procedures_per_day"] = (df["num_lab_procedures"] / days).round(3)

    if "num_procedures" in df.columns:
        df["num_procedures_per_day"] = (df["num_procedures"] / days).round(3)

    # ------------------------------------------------------------------
    # Diagnosis-based features
    # ------------------------------------------------------------------
    diag_cat_cols = _present_cols(df, ["diag_1_cat","diag_2_cat","diag_3_cat"])

    if diag_cat_cols:
        # Whether primary diagnosis is diabetes
        if "diag_1_cat" in df.columns:
            df["has_primary_diabetes_diag"] = (
                df["diag_1_cat"] == "diabetes"
            ).astype(np.int8)

        # Whether ANY diagnosis is diabetes
        df["has_any_diabetes_diag"] = (
            df[diag_cat_cols]
            .apply(lambda row: int("diabetes" in row.values), axis=1)
            .astype(np.int8)
        )

        # Number of distinct ICD-9 categories (comorbidity count)
        df["comorbidity_count"] = (
            df[diag_cat_cols]
            .apply(lambda row: len(set(row.dropna()) - {"other"}), axis=1)
            .astype(np.int8)
        )

    # ------------------------------------------------------------------
    # Composite complexity score
    # ------------------------------------------------------------------
    score = pd.Series(0.0, index=df.index)
    if "total_medications" in df.columns:
        score += df["total_medications"]
    if "num_lab_procedures" in df.columns:
        score += df["num_lab_procedures"] * 0.1
    if "comorbidity_count" in df.columns:
        score += df["comorbidity_count"] * 2
    if "num_diagnoses" in df.columns:
        score += df["num_diagnoses"]
    df["admission_complexity"] = score.round(3)

    # ------------------------------------------------------------------
    # Demographic features
    # ------------------------------------------------------------------
    if "age" in df.columns:
        df["is_elderly"] = (df["age"] >= 65).astype(np.int8)

    # ------------------------------------------------------------------
    # Lab test flags
    # ------------------------------------------------------------------
    if "A1Cresult" in df.columns:
        df["a1c_tested"] = (df["A1Cresult"] > 0).astype(np.int8)

    if "max_glu_serum" in df.columns:
        df["glucose_tested"] = (df["max_glu_serum"] > 0).astype(np.int8)

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    if "age" in df.columns and "total_medications" in df.columns:
        df["age_x_num_medications"] = (df["age"] * df["total_medications"]).astype(np.int32)

    # ------------------------------------------------------------------
    # Re-admission history proxy
    # ------------------------------------------------------------------
    if "number_outpatient" in df.columns and "number_inpatient" in df.columns:
        df["prior_admissions"] = df["number_outpatient"] + df["number_inpatient"]
        if "number_emergency" in df.columns:
            df["prior_admissions"] += df["number_emergency"]
        df["prior_admissions"] = df["prior_admissions"].astype(np.int16)

    log.info(f"Feature engineering complete. New shape: {df.shape}")
    return df


def run_feature_engineering(input_path: Path = INPUT_CSV,
                             output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    """
    Loads the cleaned CSV, engineers new clinical features, and saves the result.
    
    Args:
        input_path (Path): Path to the `cleaned.csv` file.
        output_path (Path): Path to save the resulting `featured.csv` file.

    Returns:
        pd.DataFrame: The dataframe enriched with new clinical features.
        
    Raises:
        FileNotFoundError: If the cleaned data CSV is not found.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {input_path}\n"
            "Run: python features/preprocess.py"
        )

    log.info(f"Loading {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        log.error(f"Failed to load cleaned data from {input_path}: {e}")
        raise e

    try:
        df = engineer_features(df)
    except Exception as e:
        log.error(f"Error during feature engineering: {e}")
        raise e

    df.to_csv(output_path, index=False)
    log.info(f"Saved featured data → {output_path}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
