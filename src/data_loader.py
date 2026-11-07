"""
ClinicSense – Data Loader
Reads the Kaggle 'Medical Appointment No-Shows' CSV, standardises column names,
fixes data types and performs basic integrity checks.
"""

import os
import pandas as pd
import numpy as np
from src.config import RAW_DATA_FILE, TARGET


# ── Column name map ────────────────────────────────────────────────────────────
RENAME_MAP = {
    "PatientId":       "patient_id",
    "AppointmentID":   "appointment_id",
    "Gender":          "gender",
    "ScheduledDay":    "scheduled_day",
    "AppointmentDay":  "appointment_day",
    "Age":             "age",
    "Neighbourhood":   "neighbourhood",
    "Scholarship":     "scholarship",
    "Hipertension":    "hypertension",
    "Diabetes":        "diabetes",
    "Alcoholism":      "alcoholism",
    "Handcap":         "handicap",
    "SMS_received":    "sms_received",
    "No-show":         "no_show",
}


def load_raw(filepath: str = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load the raw CSV and return a cleaned DataFrame with standardised types.

    Steps
    -----
    1. Read CSV.
    2. Rename columns to snake_case.
    3. Parse datetime columns.
    4. Map target to integer (1 = no-show, 0 = showed).
    5. Cast boolean-like columns to int8.
    6. Remove known data-quality issues:
       - negative ages
       - appointments scheduled *after* the visit date
       - duplicate appointment IDs
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please download 'KaggleV2-May-2016.csv' from\n"
            "https://www.kaggle.com/datasets/joniarroba/noshowappointments\n"
            "and place it in the data/ folder."
        )

    df = pd.read_csv(filepath)
    df.rename(columns=RENAME_MAP, inplace=True)

    # ── datetime ───────────────────────────────────────────────────────────────
    df["scheduled_day"]   = pd.to_datetime(df["scheduled_day"],   utc=True).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc=True).dt.tz_localize(None)

    # ── target ─────────────────────────────────────────────────────────────────
    df[TARGET] = (df[TARGET].str.strip().str.lower() == "yes").astype(int)

    # ── boolean columns ────────────────────────────────────────────────────────
    bool_cols = ["scholarship", "hypertension", "diabetes", "alcoholism",
                 "handicap", "sms_received"]
    for col in bool_cols:
        df[col] = df[col].astype("int8")

    # ── data quality ──────────────────────────────────────────────────────────
    before = len(df)

    # negative ages
    df = df[df["age"] >= 0]

    # age > 115 is likely error
    df = df[df["age"] <= 115]

    # scheduled after appointment
    df = df[df["scheduled_day"] <= df["appointment_day"]]

    # duplicate appointment IDs
    df = df.drop_duplicates(subset="appointment_id")

    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"[DataLoader] Removed {removed} invalid records ({before} → {after}).")

    df = df.reset_index(drop=True)
    return df


def dataset_summary(df: pd.DataFrame) -> None:
    """Print a concise dataset summary."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Date range     : {df['appointment_day'].min().date()} → "
          f"{df['appointment_day'].max().date()}")
    print(f"Unique patients: {df['patient_id'].nunique():,}")
    target_counts = df[TARGET].value_counts()
    no_show_pct   = df[TARGET].mean() * 100
    print(f"No-show count  : {target_counts.get(1, 0):,}  "
          f"({no_show_pct:.1f}%)")
    print(f"Show count     : {target_counts.get(0, 0):,}  "
          f"({100-no_show_pct:.1f}%)")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None")
    print("=" * 60)
