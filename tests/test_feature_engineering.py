"""Tests for src/feature_engineering.py"""

import pytest
import pandas as pd
import numpy as np


def test_add_lead_time(raw_df):
    from src.feature_engineering import add_lead_time
    df = add_lead_time(raw_df.copy())
    assert "lead_time_days" in df.columns
    assert (df["lead_time_days"] >= 0).all(), "lead_time_days must be non-negative"
    assert df["lead_time_days"].dtype in [np.int16, np.int32, np.int64, "int16"]


def test_add_age_group(raw_df):
    from src.feature_engineering import add_age_group
    df = add_age_group(raw_df.copy())
    assert "age_group" in df.columns
    assert "age_group_enc" in df.columns
    # ordinal encoding must be 0..4
    assert df["age_group_enc"].between(0, 4).all()


def test_add_temporal_features(raw_df):
    from src.feature_engineering import add_temporal_features
    df = add_temporal_features(raw_df.copy())
    for col in ["appt_weekday", "appt_month", "sched_hour", "is_monday", "is_friday"]:
        assert col in df.columns, f"Missing column: {col}"
    assert df["appt_weekday"].between(0, 6).all()
    assert df["appt_month"].between(1, 12).all()
    assert df["is_monday"].isin([0, 1]).all()
    assert df["is_friday"].isin([0, 1]).all()


def test_add_patient_history(raw_df):
    from src.feature_engineering import add_lead_time, add_patient_history
    df = add_lead_time(raw_df.copy())
    df = add_patient_history(df)
    for col in ["prior_appointments", "prior_no_shows", "prior_no_show_rate"]:
        assert col in df.columns
    assert (df["prior_appointments"] >= 0).all()
    assert (df["prior_no_shows"] >= 0).all()
    # rate must be in [0, 1]
    assert df["prior_no_show_rate"].between(0.0, 1.0).all()


def test_add_condition_count(raw_df):
    from src.feature_engineering import add_condition_count
    df = add_condition_count(raw_df.copy())
    assert "condition_count" in df.columns
    assert df["condition_count"].between(0, 4).all()


def test_add_same_day_flag(raw_df):
    from src.feature_engineering import add_lead_time, add_same_day_flag
    df = add_lead_time(raw_df.copy())
    df = add_same_day_flag(df)
    assert "same_day_booking" in df.columns
    assert df["same_day_booking"].isin([0, 1]).all()
    # verify: same_day_booking == 1 iff lead_time_days == 0
    assert (df["same_day_booking"] == (df["lead_time_days"] == 0).astype("int8")).all()


def test_engineer_features_output_shape(raw_df):
    from src.feature_engineering import engineer_features, MODEL_FEATURES
    df = engineer_features(raw_df.copy())
    # Must have at least as many columns as original
    assert df.shape[0] == raw_df.shape[0], "Row count must not change"
    # All MODEL_FEATURES (that exist in the df) must be present
    for feat in MODEL_FEATURES:
        assert feat in df.columns, f"Expected MODEL_FEATURES col missing: {feat}"


def test_engineer_features_no_nulls_in_model_features(raw_df):
    from src.feature_engineering import engineer_features, MODEL_FEATURES
    df = engineer_features(raw_df.copy())
    feats = [f for f in MODEL_FEATURES if f in df.columns]
    # neighbourhood_freq comes from preprocessing, may be absent in raw_df fixture
    feats = [f for f in feats if f != "neighbourhood_freq"]
    null_counts = df[feats].isnull().sum()
    bad = null_counts[null_counts > 0]
    assert bad.empty, f"Unexpected NaN in model features: {bad}"
