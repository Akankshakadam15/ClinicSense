"""Tests for src/simulator.py"""

import pytest
import numpy as np
import pandas as pd


def test_simulate_day_no_overflow(daily_pools):
    from src.simulator import simulate_day
    # High attendance probs → almost no overflow
    probs = np.ones(20) * 0.99  # 20 patients almost certain to show
    result = simulate_day(probs, n_slots=20)
    assert "total_cost" in result
    assert "idle_slots" in result
    assert "utilisation" in result
    assert result["n_booked"] == 20
    assert result["overflow"] >= 0
    assert result["idle_slots"] >= 0
    assert 0.0 <= result["utilisation"] <= 1.0


def test_simulate_day_all_noshows():
    from src.simulator import simulate_day
    probs = np.zeros(20)  # everyone is a no-show
    result = simulate_day(probs, n_slots=20)
    # All slots idle
    assert result["idle_slots"] == 20
    assert result["overflow"] == 0
    assert result["utilisation"] == 0.0


def test_simulate_day_cost_non_negative():
    from src.simulator import simulate_day
    rng = np.random.default_rng(42)
    for _ in range(20):
        probs = rng.uniform(0.1, 0.9, 25)
        result = simulate_day(probs, n_slots=20)
        assert result["total_cost"] >= 0, "Total cost must be non-negative"
        assert result["idle_cost"] >= 0
        assert result["waiting_cost"] >= 0
        assert result["overtime_cost"] >= 0


def test_no_overbooking():
    from src.simulator import no_overbooking
    assert no_overbooking(np.array([0.5, 0.6])) == 0


def test_fixed_overbooking():
    from src.simulator import fixed_overbooking
    fn = fixed_overbooking(3)
    assert fn(np.array([0.5])) == 3
    fn0 = fixed_overbooking(0)
    assert fn0(np.array([0.5])) == 0


def test_risk_based_overbooking():
    from src.simulator import risk_based_overbooking
    # 10 patients each with 0.2 attendance prob → 8 expected no-shows
    probs = np.full(10, 0.2)
    extra = risk_based_overbooking(probs)
    assert extra == 8


def test_monte_carlo_returns_dataframe(daily_pools):
    from src.simulator import monte_carlo, no_overbooking
    df = monte_carlo(daily_pools, no_overbooking, n_slots=20, n_days=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "total_cost" in df.columns
    assert "utilisation" in df.columns


def test_compare_strategies_has_all_strategies(daily_pools):
    from src.simulator import compare_strategies
    summary, raw = compare_strategies(daily_pools, n_slots=20, n_days=50)
    assert "No Overbooking (0)" in summary.index
    assert "Risk-Based Dynamic" in summary.index
    assert "Avg Total Cost" in summary.columns
    # All costs must be positive
    assert (summary["Avg Total Cost"] >= 0).all()


def test_compare_strategies_risk_based_exists(daily_pools):
    from src.simulator import compare_strategies
    summary, raw = compare_strategies(daily_pools, n_slots=20, n_days=50)
    assert "Risk-Based Dynamic" in raw
    assert isinstance(raw["Risk-Based Dynamic"], pd.DataFrame)
