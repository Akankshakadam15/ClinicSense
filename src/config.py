"""
ClinicSense – Central Configuration
All paths, hyperparameter grids, cost assumptions and random seeds live here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR  = os.path.join(OUTPUTS_DIR, "figures")
MODELS_DIR   = os.path.join(OUTPUTS_DIR, "models")
REPORTS_DIR  = os.path.join(OUTPUTS_DIR, "reports")

RAW_DATA_FILE = os.path.join(DATA_DIR, "KaggleV2-May-2016.csv")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Target column ──────────────────────────────────────────────────────────────
TARGET = "no_show"

# ── Train / test split ─────────────────────────────────────────────────────────
TEST_SIZE        = 0.20   # 20 % hold-out
TIME_SPLIT_MONTH = "2016-05"  # months after this go to the test set for temporal split

# ── Class-imbalance ────────────────────────────────────────────────────────────
CLASS_WEIGHT = "balanced"

# ── Simulation cost assumptions (in currency units) ────────────────────────────
COST_IDLE_SLOT       = 150   # revenue lost per unused slot
COST_WAITING_MINUTE  =   2   # penalty per patient waiting minute
COST_OVERTIME_MINUTE =   5   # penalty per doctor overtime minute
SLOT_DURATION_MIN    =  20   # minutes per slot
SLOTS_PER_DAY        =  20   # total slots in a simulated clinic day
SIM_DAYS             = 5000  # Monte-Carlo replications

# ── Risk thresholds ────────────────────────────────────────────────────────────
RISK_LOW_THRESHOLD    = 0.30
RISK_HIGH_THRESHOLD   = 0.55

# ── Clustering ─────────────────────────────────────────────────────────────────
N_CLUSTERS = 4

# ── SHAP ───────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 2000   # rows for SHAP calculation to save time

# ── Overbooking levels to test ─────────────────────────────────────────────────
FIXED_OB_LEVELS = [0, 1, 2, 3, 4, 5]   # extra bookings above base capacity
