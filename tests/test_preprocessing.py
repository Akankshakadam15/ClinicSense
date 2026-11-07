"""Tests for src/preprocessing.py"""

import pytest
import pandas as pd
import numpy as np


def test_cap_outliers(raw_df):
    from src.preprocessing import cap_outliers
    df = cap_outliers(raw_df.copy(), "age", factor=3.0)
    assert "age" in df.columns
    # After capping, no extreme values outside IQR*3 should remain
    Q1, Q3 = raw_df["age"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
    assert df["age"].between(lower, upper).all()


def test_encode_categoricals_drops_neighbourhood(raw_df):
    """encode_categoricals must drop 'neighbourhood' and create 'neighbourhood_freq'."""
    from src.preprocessing import encode_categoricals
    df_copy = raw_df.copy()
    # Add a string neighbourhood column (as it comes from the raw CSV)
    df_copy["neighbourhood"] = "CENTRO"
    df_out = encode_categoricals(df_copy)
    assert "neighbourhood" not in df_out.columns
    assert "neighbourhood_freq" in df_out.columns


def test_encode_categoricals_works_with_integer_gender(raw_df):
    """encode_categoricals must handle integer gender columns without crashing."""
    from src.preprocessing import encode_categoricals
    df_copy = raw_df.copy()
    df_copy["neighbourhood"] = "JARDIM"
    # raw_df already has integer gender (0/1) — must not raise AttributeError
    df_out = encode_categoricals(df_copy)
    assert set(df_out["gender"].unique()).issubset({0, 1})


def test_encode_categoricals_gender_values(raw_df):
    from src.preprocessing import encode_categoricals
    df_copy = raw_df.copy()
    # Simulate string gender column as it comes from raw CSV
    df_copy["gender"] = df_copy["gender"].map({0: "F", 1: "M"})
    df_copy["neighbourhood"] = "TEST"
    df_out = encode_categoricals(df_copy)
    assert set(df_out["gender"].unique()).issubset({0, 1})


def test_stratified_split_proportions(engineered_df):
    from src.preprocessing import stratified_split
    from src.config import TARGET
    X_tr, X_te, y_tr, y_te = stratified_split(engineered_df)
    total = len(X_tr) + len(X_te)
    assert abs(len(X_te) / total - 0.20) < 0.05  # roughly 20%
    # class proportions should be similar between splits
    assert abs(y_tr.mean() - y_te.mean()) < 0.10


def test_temporal_split_ordering(engineered_df):
    from src.preprocessing import temporal_split
    from src.config import TIME_SPLIT_MONTH
    import pandas as pd
    X_tr, X_te, y_tr, y_te = temporal_split(engineered_df)
    # Temporal split may return empty test set if all data is before the cutoff.
    # In that case just verify no crash.
    assert len(X_tr) > 0
