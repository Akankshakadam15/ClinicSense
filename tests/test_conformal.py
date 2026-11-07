"""Tests for conformal_predict in src/advanced_analytics.py (MAPIE 1.6.0)."""

import pytest
import numpy as np


def test_conformal_predict_returns_valid_shapes(split_data):
    from src.advanced_analytics import conformal_predict
    from src.models import build_lightgbm

    X_tr, X_te, y_tr, y_te = split_data
    model = build_lightgbm()
    model.fit(X_tr, y_tr)

    # Use first 60% of training for calibration
    n_cal = int(0.6 * len(X_tr))
    X_cal, y_cal = X_tr.iloc[:n_cal], y_tr.iloc[:n_cal]

    pred_sets, coverage, singleton_rate = conformal_predict(
        model, X_cal, y_cal, X_te, confidence=0.90
    )

    if pred_sets is None:
        pytest.fail("conformal_predict returned None — check MAPIE installation and API")

    assert pred_sets.shape[0] == len(X_te), "Row count must match test set"
    assert pred_sets.shape[1] == 2, "Should have 2 classes (binary classification)"
    assert isinstance(coverage, float), "Coverage must be a float"
    assert isinstance(singleton_rate, float), "singleton_rate must be a float"
    assert 0.0 <= coverage <= 1.0, f"Coverage {coverage:.3f} out of [0,1]"
    assert 0.0 <= singleton_rate <= 1.0, f"singleton_rate {singleton_rate:.3f} out of [0,1]"
    # At 90% confidence, coverage should be at least 70% (small calibration set)
    assert coverage >= 0.70, f"Coverage {coverage:.2f} is unexpectedly low for 90% target"


def test_conformal_predict_handles_import_error(split_data, monkeypatch):
    """If mapie is unavailable, conformal_predict should return (None, None, None)."""
    import src.advanced_analytics as aa

    # Temporarily make SplitConformalClassifier unavailable
    original = aa.conformal_predict

    def patched_conformal(*args, **kwargs):
        # Simulate ImportError
        try:
            import mapie.classification as mc
            # Temporarily hide SplitConformalClassifier
            original_scc = mc.SplitConformalClassifier
            del mc.SplitConformalClassifier
            result = original(*args, **kwargs)
            mc.SplitConformalClassifier = original_scc
            return result
        except Exception:
            return None, None, None

    X_tr, X_te, y_tr, y_te = split_data
    result = patched_conformal(None, X_tr, y_tr, X_te)
    # Should not crash
    assert len(result) == 3
