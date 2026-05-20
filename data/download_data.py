"""
data/download_data.py
---------------------
Downloads the Diabetes 130-US Hospitals dataset (1999-2008).

Primary source : Kaggle  (requires `kaggle` CLI configured)
Fallback source: UCI ML Repository (direct CSV download)

Usage:
    python data/download_data.py                # tries Kaggle, then UCI
    python data/download_data.py --source uci   # UCI only
    python data/download_data.py --source kaggle

Outputs (in data/raw/):
    diabetic_data.csv          - Main encounters table (~100K rows, 50 cols)
    IDs_mapping.csv            - Code-to-description mapping
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT_DIR / "data" / "raw"

KAGGLE_DATASET  = "brandao/diabetes"
UCI_DATA_URL    = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00296/dataset_diabetes.zip"
)
EXPECTED_FILES  = ["diabetic_data.csv", "IDs_mapping.csv"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def already_downloaded() -> bool:
    """Return True if all expected files already exist in data/raw/."""
    return all((RAW_DIR / f).exists() for f in EXPECTED_FILES)


def download_from_kaggle() -> bool:
    """
    Attempt to download via the Kaggle CLI.

    Requires ~/.kaggle/kaggle.json with a valid API token.
    """
    if shutil.which("kaggle") is None:
        log.warning("Kaggle CLI not found. Install with: pip install kaggle")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading '{KAGGLE_DATASET}' via Kaggle API → {RAW_DIR}")
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(RAW_DIR),
                "--unzip",
                "--force",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info(result.stdout.strip())
        return already_downloaded()
    except subprocess.CalledProcessError as exc:
        log.warning(f"Kaggle download failed: {exc.stderr.strip()}")
        return False


def download_from_uci() -> bool:
    """
    Fallback: download the zip directly from the UCI ML Repository
    and extract into data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "dataset_diabetes.zip"

    log.info(f"Downloading dataset from UCI ML Repository → {zip_path}")
    try:
        resp = requests.get(UCI_DATA_URL, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        chunk_size = 8192

        with open(zip_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))

        log.info("Extracting zip archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Flatten into raw/ (strip any sub-folder)
                filename = Path(member).name
                if filename:
                    target = RAW_DIR / filename
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

        zip_path.unlink(missing_ok=True)
        log.info("UCI download and extraction complete.")
        return already_downloaded()

    except Exception as exc:
        log.error(f"UCI download failed: {exc}")
        return False


def verify_download() -> None:
    """Log basic stats for each expected file."""
    for fname in EXPECTED_FILES:
        fpath = RAW_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1_048_576
            log.info(f"  ✓ {fname}  ({size_mb:.2f} MB)")
        else:
            log.error(f"  ✗ {fname} — NOT FOUND")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Diabetes 130 dataset")
    parser.add_argument(
        "--source",
        choices=["auto", "kaggle", "uci"],
        default="auto",
        help="Data source (default: auto — tries Kaggle first, then UCI)",
    )
    args = parser.parse_args()

    if already_downloaded():
        log.info("Dataset already present in data/raw/. Skipping download.")
        verify_download()
        return

    success = False

    if args.source in ("auto", "kaggle"):
        log.info("=== Attempting Kaggle download ===")
        success = download_from_kaggle()

    if not success and args.source in ("auto", "uci"):
        log.info("=== Falling back to UCI ML Repository ===")
        success = download_from_uci()

    if success:
        log.info("✓ Dataset download successful.")
        verify_download()
    else:
        log.error("✗ All download attempts failed.")
        log.error(
            "Manual option: download 'diabetic_data.csv' and 'IDs_mapping.csv' "
            "from https://www.kaggle.com/datasets/brandao/diabetes "
            "and place them in data/raw/"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
