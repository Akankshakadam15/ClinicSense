"""
ClinicSense – run_all.py
========================
Executes all project notebooks in sequence using nbconvert.
This script is useful for a full end-to-end run after the dataset is in place.

Usage
-----
    python run_all.py              # execute all notebooks
    python run_all.py --nb 01 03   # execute only notebooks 01 and 03

Requirements
------------
- pip install nbformat nbconvert ipykernel
- The Kaggle dataset must be present at data/KaggleV2-May-2016.csv

Notes
-----
Each notebook must pass without errors for the next one to run.
Intermediate artefacts (splits.joblib, best_model_output.joblib, etc.) are
written to data/ by the notebooks and consumed by later ones.
"""

import argparse
import subprocess
import sys
import os
import time

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "notebooks")

NOTEBOOKS = [
    "01_data_collection_eda.ipynb",
    "02_preprocessing_feature_engineering.ipynb",
    "03_predictive_modelling.ipynb",
    "04_model_evaluation_optimization.ipynb",
    "05_prediction_interpretation.ipynb",
    "06_overbooking_simulator.ipynb",
    "07_advanced_analytics.ipynb",
    "08_reinforcement_learning.ipynb",
]


def run_notebook(nb_path: str) -> bool:
    """Execute a single notebook in-place via nbconvert."""
    print(f"\n{'='*65}")
    print(f"  Running: {os.path.basename(nb_path)}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=1800",
            "--ExecutePreprocessor.kernel_name=python3",
            nb_path,
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓ Done in {elapsed:.1f}s")
        return True
    else:
        print(f"  ✗ FAILED after {elapsed:.1f}s")
        print(result.stderr[-3000:])  # print last 3 KB of error
        return False


def main():
    parser = argparse.ArgumentParser(description="Run ClinicSense notebooks end-to-end.")
    parser.add_argument(
        "--nb", nargs="*", metavar="ID",
        help="Run only notebooks matching these prefixes (e.g. --nb 01 03)"
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", default=True,
        help="Stop immediately if a notebook fails (default: True)"
    )
    args = parser.parse_args()

    # Check dataset
    data_csv = os.path.join(os.path.dirname(__file__), "data", "KaggleV2-May-2016.csv")
    if not os.path.exists(data_csv):
        print("\n⚠  WARNING: Dataset not found at data/KaggleV2-May-2016.csv")
        print("   Download from https://www.kaggle.com/datasets/joniarroba/noshowappointments")
        print("   Notebooks that require the dataset will fail.\n")

    # Filter notebooks
    notebooks = NOTEBOOKS
    if args.nb:
        notebooks = [nb for nb in NOTEBOOKS
                     if any(nb.startswith(f"{prefix}_") or nb.startswith(prefix)
                            for prefix in args.nb)]

    total, passed, failed = len(notebooks), 0, 0
    for nb_name in notebooks:
        nb_path = os.path.join(NOTEBOOK_DIR, nb_name)
        if not os.path.exists(nb_path):
            print(f"\n  ✗ Not found: {nb_path}")
            failed += 1
            if args.stop_on_error:
                break
            continue
        ok = run_notebook(nb_path)
        if ok:
            passed += 1
        else:
            failed += 1
            if args.stop_on_error:
                print("\nStopped due to notebook failure.")
                break

    print(f"\n{'='*65}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*65}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
