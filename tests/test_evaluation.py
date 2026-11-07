"""Tests for src/evaluation.py"""

import pytest
import numpy as np


@pytest.fixture
def simple_predictions():
    """Simple binary classification results for metric verification."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def test_compute_metrics_keys(simple_predictions):
    from src.evaluation import compute_metrics
    y_true, y_pred, y_prob = simple_predictions
    m = compute_metrics(y_true, y_pred, y_prob, "test_model")
    expected_keys = {"model", "accuracy", "precision", "recall",
                     "f1", "roc_auc", "pr_auc", "brier_score"}
    assert set(m.keys()) == expected_keys


def test_compute_metrics_ranges(simple_predictions):
    from src.evaluation import compute_metrics
    y_true, y_pred, y_prob = simple_predictions
    m = compute_metrics(y_true, y_pred, y_prob, "test_model")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert 0.0 <= m[key] <= 1.0, f"{key} out of [0,1]: {m[key]}"
    assert 0.0 <= m["brier_score"] <= 1.0


def test_find_cost_optimal_threshold_returns_valid(simple_predictions):
    from src.evaluation import find_cost_optimal_threshold
    y_true, _, y_prob = simple_predictions
    thresh, cost = find_cost_optimal_threshold(y_true, y_prob)
    assert 0.01 <= thresh <= 0.99
    assert cost >= 0


def test_find_cost_optimal_threshold_uses_config_default():
    """cost_fn default should equal COST_IDLE_SLOT from config."""
    from src.evaluation import find_cost_optimal_threshold
    from src.config import COST_IDLE_SLOT
    import inspect
    src = inspect.getsource(find_cost_optimal_threshold)
    # The function body must reference COST_IDLE_SLOT (not a hard-coded 150)
    assert "COST_IDLE_SLOT" in src


def test_comparison_table_shape():
    from src.evaluation import comparison_table, compute_metrics
    import numpy as np
    rng = np.random.default_rng(1)
    metrics_list = []
    for name in ["LR", "RF", "XGB"]:
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics_list.append(compute_metrics(y_true, y_pred, y_prob, name))
    table = comparison_table(metrics_list)
    assert table.shape[0] == 3
    assert "f1" in table.columns
    assert "roc_auc" in table.columns
