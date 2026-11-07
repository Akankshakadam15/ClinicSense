"""Tests for src/report_writer.py"""

import pytest
import os
import pandas as pd
import numpy as np


@pytest.fixture
def sample_comparison_df():
    """Minimal model comparison DataFrame."""
    data = {
        "accuracy":    [0.80, 0.82, 0.84],
        "precision":   [0.55, 0.57, 0.59],
        "recall":      [0.60, 0.62, 0.64],
        "f1":          [0.57, 0.59, 0.61],
        "roc_auc":     [0.78, 0.80, 0.82],
        "pr_auc":      [0.45, 0.47, 0.49],
        "brier_score": [0.14, 0.13, 0.12],
    }
    return pd.DataFrame(data, index=["Logistic Regression", "Random Forest", "LightGBM"])


@pytest.fixture
def sample_simulation_df():
    """Minimal simulation summary DataFrame."""
    data = {
        "Avg Total Cost":    [1500.0, 1200.0, 900.0],
        "Avg Idle Slots":    [4.0, 2.5, 1.5],
        "Avg Overflow":      [0.0, 0.5, 0.8],
        "Avg Utilisation %": [75.0, 85.0, 90.0],
        "Avg Wait (min)":    [0.0, 5.0, 8.0],
        "Avg Overtime (min)":[0.0, 5.0, 8.0],
        "Std Total Cost":    [200.0, 180.0, 160.0],
    }
    return pd.DataFrame(data, index=["No Overbooking (0)", "Fixed +3", "Risk-Based Dynamic"])


@pytest.fixture
def sample_eval_df():
    """Minimal RL eval DataFrame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "episode": range(50),
        "reward":  rng.normal(2000, 300, 50),
        "n_booked": rng.integers(20, 25, 50),
    })


def test_write_model_report_creates_file(tmp_path, monkeypatch, sample_comparison_df):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    path = report_writer.write_model_report(
        comparison_df=sample_comparison_df,
        best_model_name="LightGBM",
        optimal_threshold=0.35,
        timestamp=False,
    )
    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert "LightGBM" in content
    assert "0.35" in content
    assert "ROC-AUC" in content or "roc_auc" in content.lower()


def test_write_model_report_also_saves_csv(tmp_path, monkeypatch, sample_comparison_df):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    report_writer.write_model_report(
        comparison_df=sample_comparison_df,
        best_model_name="Random Forest",
        optimal_threshold=0.40,
        timestamp=False,
    )
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) >= 1


def test_write_model_report_empty_raises(tmp_path, monkeypatch):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        report_writer.write_model_report(
            comparison_df=pd.DataFrame(),
            best_model_name="X",
            optimal_threshold=0.5,
        )


def test_write_simulation_report_creates_file(tmp_path, monkeypatch, sample_simulation_df):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    path = report_writer.write_simulation_report(
        summary_df=sample_simulation_df,
        best_strategy="Risk-Based Dynamic",
        timestamp=False,
    )
    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert "Risk-Based Dynamic" in content
    assert "Strategy" in content or "strategy" in content.lower()


def test_write_simulation_report_savings_shown(tmp_path, monkeypatch, sample_simulation_df):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    path = report_writer.write_simulation_report(
        summary_df=sample_simulation_df,
        best_strategy="Risk-Based Dynamic",
        baseline_name="No Overbooking (0)",
        timestamp=False,
    )
    content = open(path, encoding="utf-8").read()
    # savings calculation should appear
    assert "₹" in content or "saves" in content


def test_write_simulation_report_empty_raises(tmp_path, monkeypatch):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        report_writer.write_simulation_report(
            summary_df=pd.DataFrame(),
            best_strategy="X",
        )


def test_write_rl_report_creates_file(tmp_path, monkeypatch, sample_eval_df):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    comparison = {"No Overbooking": 1800.0, "Fixed +3": 2100.0, "Q-Learning": 2300.0}
    path = report_writer.write_rl_report(
        eval_df=sample_eval_df,
        comparison_dict=comparison,
        n_episodes_trained=100,
        timestamp=False,
    )
    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert "Q-Learning" in content
    assert "reward" in content.lower()


def test_write_rl_report_empty_raises(tmp_path, monkeypatch):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        report_writer.write_rl_report(
            eval_df=pd.DataFrame(),
            comparison_dict={},
            n_episodes_trained=0,
        )


def test_write_fairness_report_creates_file(tmp_path, monkeypatch):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)

    fake_results = {
        "gender": {
            "demographic_parity_diff": 0.05,
            "equalized_odds_diff": 0.07,
        },
        "age_group": {
            "demographic_parity_diff": 0.12,
            "equalized_odds_diff": 0.09,
        },
    }
    path = report_writer.write_fairness_report(fake_results, timestamp=False)
    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert "gender" in content
    assert "age_group" in content
    assert "0.05" in content


def test_write_fairness_report_empty_raises(tmp_path, monkeypatch):
    from src import report_writer
    monkeypatch.setattr(report_writer, "REPORTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        report_writer.write_fairness_report({})
