"""
features/preprocess.py
-----------------------
Cleans the raw Diabetes 130-US Hospitals CSV and saves a structured
cleaned.csv ready for feature engineering.

Steps:
  1. Replace '?' sentinel with NaN
  2. Drop administrative / high-missingness columns
  3. Remove duplicate patient encounters (keep first — preserves temporal order)
  4. Binary target: readmitted_lt30 (1 = <30 days, 0 = otherwise)
  5. Encode age brackets, gender, binary flags, A1C/glucose, medications
  6. Map ICD-9 diagnosis codes to broad categories
  7. One-hot encode remaining categoricals
  8. Median-impute residual numeric NaNs

Usage:
    python features/preprocess.py
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR      = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_CSV       = RAW_DIR / "diabetic_data.csv"
OUTPUT_CSV    = PROCESSED_DIR / "cleaned.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DROP_COLS = [
    "encounter_id", "patient_nbr",
    "examide", "citoglipton",
    "weight", "payer_code", "medical_specialty",
]

AGE_MAP = {
    "[0-10)":5,"[10-20)":15,"[20-30)":25,"[30-40)":35,"[40-50)":45,
    "[50-60)":55,"[60-70)":65,"[70-80)":75,"[80-90)":85,"[90-100)":95,
}

MED_COLS = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
    "insulin","glyburide-metformin","glipizide-metformin",
    "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone",
]

DIAG_CATS = {
    "circulatory":(390,459), "respiratory":(460,519), "digestive":(520,579),
    "diabetes":(250,250), "injury":(800,999), "musculoskeletal":(710,739),
    "genitourinary":(580,629), "neoplasms":(140,239),
}


def categorize_icd9(code: str | float) -> str:
    if pd.isna(code):
        return "other"
    s = str(code).strip().upper()
    if s.startswith(("V","E")):
        return "external"
    try:
        num = float(s.split(".")[0])
    except ValueError:
        return "other"
    for cat, (lo, hi) in DIAG_CATS.items():
        if lo <= num <= hi:
            return cat
    return "other"


def run_preprocessing(input_path: Path = RAW_CSV,
                      output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    """
    Executes the full preprocessing pipeline on the raw diabetes dataset.
    
    This function handles missing values, removes duplicates, encodes categorical
    variables, and prepares the dataset for feature engineering.

    Args:
        input_path (Path): Path to the raw `diabetic_data.csv` file.
        output_path (Path): Path to save the processed `cleaned.csv` file.

    Returns:
        pd.DataFrame: The cleaned and preprocessed dataframe.
        
    Raises:
        FileNotFoundError: If the raw data CSV is not found.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {input_path}\n"
            "Run: python data/download_data.py"
        )

    log.info(f"Loading {input_path}")
    try:
        df = pd.read_csv(input_path, na_values=["?"])
    except pd.errors.EmptyDataError:
        log.error(f"The file {input_path} is empty.")
        sys.exit(1)
    except Exception as e:
        log.error(f"Failed to read CSV at {input_path}: {e}")
        sys.exit(1)

    log.info(f"Raw shape: {df.shape}")

    # --- Dedup (keep first encounter per patient, sorted by encounter_id) ---
    df = df.sort_values("encounter_id")
    n_before = len(df)
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    log.info(f"Removed {n_before - len(df):,} duplicate patient encounters")

    # --- Encounter order proxy ---
    df["encounter_order"] = df["encounter_id"].rank(method="first").astype(int)

    # --- Drop low-value columns ---
    drop_present = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_present)
    log.info(f"Dropped: {drop_present}")

    # --- Binary target ---
    df["readmitted_lt30"] = (df["readmitted"] == "<30").astype(np.int8)
    df = df.drop(columns=["readmitted"])
    pos = df["readmitted_lt30"].sum()
    log.info(f"Target → pos={pos:,} ({100*pos/len(df):.1f}%), neg={len(df)-pos:,}")

    # --- Age ---
    df["age"] = df["age"].map(AGE_MAP)

    # --- Gender ---
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    # --- Binary flags ---
    for col in ["change", "diabetesMed"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes":1,"Ch":1,"No":0}).fillna(0).astype(np.int8)

    # --- A1C / glucose ---
    df["A1Cresult"] = df["A1Cresult"].map(
        {"None":0,"Norm":1,">7":2,">8":3}).fillna(0).astype(np.int8)
    df["max_glu_serum"] = df["max_glu_serum"].map(
        {"None":0,"Norm":1,">200":2,">300":3}).fillna(0).astype(np.int8)

    # --- Medications ---
    med_map = {"No":0,"Steady":1,"Up":2,"Down":2}
    for col in [c for c in MED_COLS if c in df.columns]:
        df[col] = df[col].map(med_map).fillna(0).astype(np.int8)

    # --- Diagnoses ---
    for col in ["diag_1","diag_2","diag_3"]:
        if col in df.columns:
            df[f"{col}_cat"] = df[col].apply(categorize_icd9)
            df = df.drop(columns=[col])

    # --- One-hot encode categoricals ---
    cat_cols = df.select_dtypes("object").columns.tolist()
    if cat_cols:
        log.info(f"One-hot encoding: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=np.int8)

    # --- Impute numeric NaNs ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    log.info(f"Cleaned shape: {df.shape}")
    df.to_csv(output_path, index=False)
    log.info(f"Saved → {output_path}")
    return df


if __name__ == "__main__":
    run_preprocessing()
