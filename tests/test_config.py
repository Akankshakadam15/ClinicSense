"""Tests for src/config.py – verify all constants exist and are sane."""

import os
import pytest


def test_constants_exist():
    from src.config import (
        BASE_DIR, DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR,
        REPORTS_DIR, RAW_DATA_FILE, RANDOM_STATE, TARGET,
        TEST_SIZE, TIME_SPLIT_MONTH, CLASS_WEIGHT,
        COST_IDLE_SLOT, COST_WAITING_MINUTE, COST_OVERTIME_MINUTE,
        SLOT_DURATION_MIN, SLOTS_PER_DAY, SIM_DAYS,
        RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD,
        N_CLUSTERS, SHAP_SAMPLE_SIZE, FIXED_OB_LEVELS,
    )
    # basic sanity checks
    assert TARGET == "no_show"
    assert 0 < TEST_SIZE < 1
    assert RANDOM_STATE >= 0
    assert COST_IDLE_SLOT > 0
    assert COST_WAITING_MINUTE > 0
    assert COST_OVERTIME_MINUTE > 0
    assert 0 < RISK_LOW_THRESHOLD < RISK_HIGH_THRESHOLD < 1
    assert SLOTS_PER_DAY > 0
    assert SIM_DAYS >= 100
    assert N_CLUSTERS >= 2
    assert isinstance(FIXED_OB_LEVELS, list)
    assert 0 in FIXED_OB_LEVELS


def test_paths_are_absolute():
    from src.config import BASE_DIR, DATA_DIR, FIGURES_DIR, MODELS_DIR
    for p in [BASE_DIR, DATA_DIR, FIGURES_DIR, MODELS_DIR]:
        assert os.path.isabs(p), f"Expected absolute path, got: {p}"
