"""
ClinicSense – Feature Engineering
Derives all domain-relevant features from the cleaned appointment data.
"""

import pandas as pd
import numpy as np
from src.config import TARGET


# ── Age groups ────────────────────────────────────────────────────────────────
AGE_BINS   = [0, 12, 17, 35, 60, 115]
AGE_LABELS = ["child", "teen", "young_adult", "adult", "senior"]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS,
                             right=True, include_lowest=True)
    # ordinal encoding
    age_map = {l: i for i, l in enumerate(AGE_LABELS)}
    df["age_group_enc"] = df["age_group"].map(age_map).astype("int8")
    return df


# ── Lead time ─────────────────────────────────────────────────────────────────
def add_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """Days between booking and appointment."""
    df["lead_time_days"] = (
        (df["appointment_day"] - df["scheduled_day"])
        .dt.total_seconds() / 86400
    ).clip(lower=0).astype("int16")
    return df


# ── Temporal features ─────────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["appt_weekday"]    = df["appointment_day"].dt.dayofweek.astype("int8")
    df["appt_month"]      = df["appointment_day"].dt.month.astype("int8")
    df["sched_hour"]      = df["scheduled_day"].dt.hour.astype("int8")
    df["is_monday"]       = (df["appt_weekday"] == 0).astype("int8")
    df["is_friday"]       = (df["appt_weekday"] == 4).astype("int8")
    return df


# ── Patient history features ──────────────────────────────────────────────────
def add_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each appointment, count:
    - prior_appointments : number of appointments this patient had before this one
    - prior_no_shows     : number of no-shows before this appointment
    - prior_no_show_rate : historic no-show rate (0 if no prior appointments)

    Appointments are sorted by appointment_day to ensure temporal ordering.
    """
    df = df.sort_values(["patient_id", "appointment_day"]).copy()

    # cumulative counts excluding current row
    df["prior_appointments"] = (
        df.groupby("patient_id").cumcount().astype("int16")
    )
    df["prior_no_shows"] = (
        df.groupby("patient_id")[TARGET]
        .apply(lambda x: x.shift(1).cumsum())
        .fillna(0)
        .astype("int16")
        .values
    )
    df["prior_no_show_rate"] = (
        df["prior_no_shows"] / df["prior_appointments"].replace(0, np.nan)
    ).fillna(0).astype("float32")

    return df


# ── Chronic condition count ───────────────────────────────────────────────────
def add_condition_count(df: pd.DataFrame) -> pd.DataFrame:
    cond_cols = ["hypertension", "diabetes", "alcoholism", "handicap"]
    df["condition_count"] = df[cond_cols].sum(axis=1).astype("int8")
    return df


# ── Same-day booking flag ─────────────────────────────────────────────────────
def add_same_day_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["same_day_booking"] = (df["lead_time_days"] == 0).astype("int8")
    return df


# ── Master function ────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in correct order."""
    df = add_lead_time(df)
    df = add_temporal_features(df)
    df = add_age_group(df)
    df = add_patient_history(df)
    df = add_condition_count(df)
    df = add_same_day_flag(df)
    return df


# ── Feature columns used for modelling ────────────────────────────────────────
MODEL_FEATURES = [
    "age",
    "age_group_enc",
    "gender",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "sms_received",
    "lead_time_days",
    "appt_weekday",
    "appt_month",
    "sched_hour",
    "is_monday",
    "is_friday",
    "prior_appointments",
    "prior_no_shows",
    "prior_no_show_rate",
    "condition_count",
    "same_day_booking",
    "neighbourhood_freq",
]
