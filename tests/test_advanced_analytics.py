"""Tests for src/advanced_analytics.py"""

import pytest
import numpy as np
import pandas as pd


# ── PSI ───────────────────────────────────────────────────────────────────────
def test_psi_identical_distributions():
    from src.advanced_analytics import psi
    arr = np.random.default_rng(0).normal(0, 1, 1000)
    score = psi(arr, arr.copy())
    assert score < 0.01, f"PSI of identical distributions should be ~0, got {score}"


def test_psi_different_distributions():
    from src.advanced_analytics import psi
    rng = np.random.default_rng(0)
    expected = rng.normal(0, 1, 2000)
    actual   = rng.normal(5, 1, 2000)  # shifted distribution
    score = psi(expected, actual)
    assert score > 0.25, f"PSI for very different distributions should be >0.25, got {score}"


def test_compute_psi_all_features_returns_dataframe(engineered_df):
    from src.advanced_analytics import compute_psi_all_features
    from src.feature_engineering import MODEL_FEATURES
    feats = [f for f in MODEL_FEATURES if f in engineered_df.columns]
    n = len(engineered_df)
    train = engineered_df.iloc[: n // 2]
    test  = engineered_df.iloc[n // 2 :]
    result = compute_psi_all_features(train, test, feats)
    assert isinstance(result, pd.DataFrame)
    assert "PSI" in result.columns
    assert "drift" in result.columns


# ── Patient Segmentation ──────────────────────────────────────────────────────
def test_segment_patients_adds_columns(engineered_df):
    from src.advanced_analytics import segment_patients
    df_out, model, scaler = segment_patients(engineered_df, n_clusters=3, method="kmeans")
    assert "cluster" in df_out.columns
    assert "cluster_name" in df_out.columns
    assert df_out["cluster"].between(0, 2).all()


def test_segment_patients_gmm(engineered_df):
    from src.advanced_analytics import segment_patients
    df_out, model, scaler = segment_patients(engineered_df, n_clusters=3, method="gmm")
    assert "cluster" in df_out.columns
    assert df_out["cluster"].nunique() <= 3


# ── T-Learner Uplift ─────────────────────────────────────────────────────────
def test_t_learner_adds_uplift_score(engineered_df):
    from src.advanced_analytics import t_learner_uplift
    from src.feature_engineering import MODEL_FEATURES
    feats = [f for f in MODEL_FEATURES if f in engineered_df.columns]
    df_out, clf1, clf0 = t_learner_uplift(
        engineered_df, feature_cols=feats,
        treatment_col="sms_received", outcome_col="no_show"
    )
    assert "uplift_score" in df_out.columns
    assert df_out["uplift_score"].between(-1.0, 1.0).all()


def test_t_learner_no_dead_import():
    """Regression test: cross_val_predict must NOT be imported inside t_learner_uplift."""
    from src.advanced_analytics import t_learner_uplift
    import inspect
    src = inspect.getsource(t_learner_uplift)
    assert "cross_val_predict" not in src, \
        "Dead import 'cross_val_predict' should have been removed from t_learner_uplift"


# ── SHAP helpers ─────────────────────────────────────────────────────────────
def test_shap_bar_returns_dataframe(engineered_df, split_data):
    """Integration test: SHAP bar chart helper returns correctly shaped DataFrame."""
    from src.advanced_analytics import plot_shap_bar
    import shap
    from src.models import build_lightgbm
    from src.feature_engineering import MODEL_FEATURES

    X_tr, X_te, y_tr, y_te = split_data
    model = build_lightgbm()
    model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)
    X_sample = X_te.iloc[:50]
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]

    imp_df = plot_shap_bar(sv, X_sample, "LightGBM_test", save=False)
    assert isinstance(imp_df, pd.DataFrame)
    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns
    assert len(imp_df) <= 15
