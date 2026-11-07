"""Tests for src/models.py – verify models build and predict without errors."""

import pytest
import numpy as np


def test_build_logistic_regression(split_data):
    from src.models import build_logistic_regression
    X_tr, X_te, y_tr, y_te = split_data
    model = build_logistic_regression()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_build_random_forest(split_data):
    from src.models import build_random_forest
    X_tr, X_te, y_tr, y_te = split_data
    model = build_random_forest()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    assert set(preds).issubset({0, 1})


def test_build_xgboost_no_use_label_encoder(split_data):
    """Regression test: build_xgboost must not pass use_label_encoder (removed in XGBoost 2.x)."""
    from src.models import build_xgboost
    import inspect
    src = inspect.getsource(build_xgboost)
    assert "use_label_encoder" not in src, \
        "use_label_encoder was removed in XGBoost 2.x – must not appear in build_xgboost()"


def test_build_xgboost_trains(split_data):
    from src.models import build_xgboost
    X_tr, X_te, y_tr, y_te = split_data
    pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    model = build_xgboost(scale_pos_weight=pos_weight)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)
    assert probs.shape[1] == 2


def test_build_lightgbm(split_data):
    from src.models import build_lightgbm
    X_tr, X_te, y_tr, y_te = split_data
    model = build_lightgbm()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2)


def test_save_and_load_model(split_data, tmp_path, monkeypatch):
    from src.models import build_random_forest, save_model, load_model
    import src.models as models_module

    # Redirect MODELS_DIR to a temporary directory for this test
    monkeypatch.setattr(models_module, "MODELS_DIR", str(tmp_path))

    X_tr, X_te, y_tr, y_te = split_data
    model = build_random_forest()
    model.fit(X_tr, y_tr)

    path = save_model(model, "test_rf")
    assert path.endswith("test_rf.joblib")

    loaded = load_model("test_rf")
    preds_original = model.predict(X_te)
    preds_loaded   = loaded.predict(X_te)
    np.testing.assert_array_equal(preds_original, preds_loaded)
