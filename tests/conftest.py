"""
Shared pytest fixtures for ClinicSense tests.
All fixtures produce synthetic data so that tests run without the Kaggle CSV.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Make the project root importable when running pytest from any directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Synthetic raw DataFrame (mimics the Kaggle CSV after load_raw()) ──────────
@pytest.fixture(scope="session")
def raw_df():
    """
    Returns a synthetic DataFrame with the same column names and dtypes
    that load_raw() produces, but entirely generated (no Kaggle CSV needed).
    500 rows is enough for all unit tests to be fast.
    """
    rng = np.random.default_rng(42)
    n   = 500

    scheduled_base = pd.Timestamp("2016-02-01")
    appointment_offsets = rng.integers(0, 90, n)  # 0–89 days after scheduled
    scheduled_days  = [scheduled_base + pd.Timedelta(days=int(d))
                       for d in rng.integers(0, 60, n)]
    appointment_days = [
        sd + pd.Timedelta(days=int(off))
        for sd, off in zip(scheduled_days, appointment_offsets)
    ]

    df = pd.DataFrame({
        "patient_id":      rng.integers(10000, 99999, n).astype(str),
        "appointment_id":  np.arange(1000, 1000 + n).astype(str),
        "gender":          rng.integers(0, 2, n).astype("int8"),   # already encoded
        "scheduled_day":   scheduled_days,
        "appointment_day": appointment_days,
        "age":             rng.integers(0, 90, n),
        "scholarship":     rng.integers(0, 2, n).astype("int8"),
        "hypertension":    rng.integers(0, 2, n).astype("int8"),
        "diabetes":        rng.integers(0, 2, n).astype("int8"),
        "alcoholism":      rng.integers(0, 2, n).astype("int8"),
        "handicap":        rng.integers(0, 2, n).astype("int8"),
        "sms_received":    rng.integers(0, 2, n).astype("int8"),
        "no_show":         rng.integers(0, 2, n).astype(int),
        "neighbourhood_freq": rng.uniform(0.001, 0.05, n).astype("float32"),
    })
    return df


# ── Engineered DataFrame ──────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def engineered_df(raw_df):
    from src.feature_engineering import engineer_features
    return engineer_features(raw_df.copy())


# ── Train/test split from engineered data ─────────────────────────────────────
@pytest.fixture(scope="session")
def split_data(engineered_df):
    from sklearn.model_selection import train_test_split
    from src.feature_engineering import MODEL_FEATURES
    from src.config import TARGET

    feats = [f for f in MODEL_FEATURES if f in engineered_df.columns]
    X = engineered_df[feats].fillna(0)
    y = engineered_df[TARGET]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# ── Daily pools for simulator tests ──────────────────────────────────────────
@pytest.fixture(scope="session")
def daily_pools():
    rng = np.random.default_rng(42)
    return [rng.uniform(0.6, 0.95, 40) for _ in range(100)]
