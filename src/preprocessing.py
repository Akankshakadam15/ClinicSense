"""
ClinicSense – Data Preprocessing
Handles outlier treatment, categorical encoding and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import TARGET, TEST_SIZE, RANDOM_STATE, TIME_SPLIT_MONTH


# ── Outlier capping (IQR method) ───────────────────────────────────────────────
def cap_outliers(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    """Cap extreme values using IQR × factor."""
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ── Categorical encoding ───────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode gender as binary (F=0, M=1).
    Handles both string ('F'/'M') and integer (0/1) gender columns.
    Neighbourhood is high-cardinality; use frequency encoding.
    """
    df = df.copy()

    # gender: F → 0, M → 1  (safe for both string and already-encoded int columns)
    if df["gender"].dtype == object:
        df["gender"] = (df["gender"].str.upper() == "M").astype("int8")
    else:
        # Already encoded as 0/1 integer — ensure int8 dtype
        df["gender"] = df["gender"].astype("int8")

    # neighbourhood → frequency encoding (only if column is present)
    if "neighbourhood" in df.columns:
        freq = df["neighbourhood"].value_counts(normalize=True)
        df["neighbourhood_freq"] = df["neighbourhood"].map(freq).astype("float32")
        df.drop(columns=["neighbourhood"], inplace=True)

    return df


# ── Standard stratified split ─────────────────────────────────────────────────
def stratified_split(df: pd.DataFrame):
    """
    80/20 stratified split preserving class ratio.
    Returns (X_train, X_test, y_train, y_test).
    """
    exclude = {TARGET, "patient_id", "appointment_id",
               "scheduled_day", "appointment_day", "age_group"}
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype != object]
    X = df[feature_cols]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            stratify=y, random_state=RANDOM_STATE)


# ── Time-based split ──────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame):
    """
    Train on appointments before TIME_SPLIT_MONTH,
    test on appointments from that month onward.
    """
    cutoff = pd.Timestamp(TIME_SPLIT_MONTH)
    train_df = df[df["appointment_day"] < cutoff].copy()
    test_df  = df[df["appointment_day"] >= cutoff].copy()

    exclude = {TARGET, "patient_id", "appointment_id",
               "scheduled_day", "appointment_day", "age_group"}
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype != object]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET]

    print(f"[TemporalSplit] Train: {len(X_train):,}  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ── Full preprocessing pipeline ────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps and return a model-ready DataFrame."""
    df = df.copy()
    df = encode_categoricals(df)
    df = cap_outliers(df, "age", factor=3.0)
    return df
