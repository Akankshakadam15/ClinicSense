"""
ClinicSense – Dynamic Overbooking Simulator (Monte Carlo)
Implements three strategies:
  A. No overbooking
  B. Fixed overbooking (constant extra bookings)
  C. Risk-based dynamic overbooking (extra bookings = round(sum of risk scores))

Cost model:
  total_cost = idle_slot_cost * idle_slots
             + waiting_cost_per_min * total_patient_wait_minutes
             + overtime_cost_per_min * doctor_overtime_minutes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os
from typing import Tuple, Dict

from src.config import (
    COST_IDLE_SLOT, COST_WAITING_MINUTE, COST_OVERTIME_MINUTE,
    SLOT_DURATION_MIN, SLOTS_PER_DAY, SIM_DAYS, FIGURES_DIR,
    FIXED_OB_LEVELS, RANDOM_STATE,
)

os.makedirs(FIGURES_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


# ── Simulate a single clinic day ──────────────────────────────────────────────
def simulate_day(show_probs: np.ndarray,
                 n_slots: int,
                 slot_duration: int = SLOT_DURATION_MIN) -> dict:
    """
    show_probs : array of attendance probabilities for each booked patient
    n_slots    : base slots available
    Returns a dict of metrics for that day.
    """
    n_booked   = len(show_probs)
    show_flags = rng.random(n_booked) < show_probs
    n_showed   = show_flags.sum()

    idle_slots = max(0, n_slots - n_showed)
    overflow   = max(0, n_showed - n_slots)   # patients beyond capacity

    # waiting time model: if n_showed > n_slots, patients queue up
    # simplified: each overflow patient waits slot_duration minutes extra on average
    waiting_minutes   = overflow * slot_duration
    overtime_minutes  = overflow * slot_duration   # doctor works overtime too

    idle_cost    = idle_slots      * COST_IDLE_SLOT
    waiting_cost = waiting_minutes * COST_WAITING_MINUTE
    ot_cost      = overtime_minutes * COST_OVERTIME_MINUTE

    return {
        "n_booked":          n_booked,
        "n_showed":          int(n_showed),
        "idle_slots":        int(idle_slots),
        "overflow":          int(overflow),
        "waiting_minutes":   float(waiting_minutes),
        "overtime_minutes":  float(overtime_minutes),
        "idle_cost":         float(idle_cost),
        "waiting_cost":      float(waiting_cost),
        "overtime_cost":     float(ot_cost),
        "total_cost":        float(idle_cost + waiting_cost + ot_cost),
        "utilisation":       float(min(n_showed, n_slots) / n_slots),
    }


# ── Monte-Carlo simulation for one strategy ──────────────────────────────────
def monte_carlo(daily_pools: list,
                extra_bookings_fn,
                n_slots: int = SLOTS_PER_DAY,
                n_days: int = SIM_DAYS) -> pd.DataFrame:
    """
    daily_pools       : list of arrays of (show_prob) per simulated day
    extra_bookings_fn : callable(show_probs_for_day) → int  (how many extra to add)
    """
    results = []
    for day_idx in range(n_days):
        pool = daily_pools[day_idx % len(daily_pools)]
        extra = extra_bookings_fn(pool)
        # select n_slots + extra patients from the pool (if pool is large enough)
        n_book = min(len(pool), n_slots + extra)
        chosen_probs = rng.choice(pool, size=n_book, replace=False)
        day_result = simulate_day(chosen_probs, n_slots)
        results.append(day_result)
    return pd.DataFrame(results)


# ── Extra-bookings functions ──────────────────────────────────────────────────
def no_overbooking(_probs):
    return 0

def fixed_overbooking(n: int):
    return lambda _probs: n

def risk_based_overbooking(probs: np.ndarray) -> int:
    """Extra bookings = round(expected no-shows for the day's patients)."""
    expected_no_shows = np.sum(1 - probs)   # expected number who won't attend
    return max(0, round(expected_no_shows))


# ── Run all strategies and compare ───────────────────────────────────────────
def compare_strategies(daily_pools: list,
                       n_slots: int = SLOTS_PER_DAY,
                       n_days: int = SIM_DAYS) -> pd.DataFrame:
    """
    Runs all strategies and returns a summary DataFrame.
    """
    strategies = {"No Overbooking (0)": no_overbooking}

    for ob in FIXED_OB_LEVELS[1:]:   # skip 0 already added above
        strategies[f"Fixed +{ob}"] = fixed_overbooking(ob)

    strategies["Risk-Based Dynamic"] = risk_based_overbooking

    summary_rows = []
    raw_results  = {}

    for name, fn in strategies.items():
        print(f"  Running strategy: {name} …", end=" ", flush=True)
        df = monte_carlo(daily_pools, fn, n_slots, n_days)
        raw_results[name] = df
        row = {
            "Strategy":            name,
            "Avg Total Cost":      df["total_cost"].mean(),
            "Avg Idle Slots":      df["idle_slots"].mean(),
            "Avg Overflow":        df["overflow"].mean(),
            "Avg Utilisation %":   df["utilisation"].mean() * 100,
            "Avg Wait (min)":      df["waiting_minutes"].mean(),
            "Avg Overtime (min)":  df["overtime_minutes"].mean(),
            "Std Total Cost":      df["total_cost"].std(),
        }
        summary_rows.append(row)
        print("done")

    summary = pd.DataFrame(summary_rows).set_index("Strategy").round(2)
    return summary, raw_results


# ── Cost heatmap (sensitivity analysis) ───────────────────────────────────────
def cost_heatmap(daily_pools: list,
                 slot_range: range = range(15, 26),
                 extra_range: range = range(0, 8),
                 n_days: int = 500,
                 save: bool = True):
    """
    Vary SLOTS_PER_DAY and fixed extra bookings; show total cost as heatmap.
    """
    matrix = np.zeros((len(slot_range), len(extra_range)))
    for i, n_slots in enumerate(slot_range):
        for j, extra in enumerate(extra_range):
            fn = fixed_overbooking(extra)
            df = monte_carlo(daily_pools, fn, n_slots=n_slots, n_days=n_days)
            matrix[i, j] = df["total_cost"].mean()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(matrix,
                xticklabels=[f"+{e}" for e in extra_range],
                yticklabels=list(slot_range),
                annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Extra Bookings (Fixed Overbooking)")
    ax.set_ylabel("Base Slots Per Day")
    ax.set_title("Sensitivity Analysis – Avg Total Cost Heatmap", fontsize=13)
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "cost_heatmap.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return matrix


# ── Plot strategy comparison ──────────────────────────────────────────────────
def plot_strategy_comparison(summary: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    summary["Avg Total Cost"].plot(kind="bar", ax=axes[0],
                                   color="steelblue", edgecolor="black")
    axes[0].set_title("Avg Total Cost per Day")
    axes[0].set_ylabel("Cost (₹)")
    axes[0].tick_params(axis="x", rotation=45)

    summary["Avg Idle Slots"].plot(kind="bar", ax=axes[1],
                                   color="tomato", edgecolor="black")
    axes[1].set_title("Avg Idle Slots per Day")
    axes[1].set_ylabel("Slots")
    axes[1].tick_params(axis="x", rotation=45)

    summary["Avg Utilisation %"].plot(kind="bar", ax=axes[2],
                                       color="mediumseagreen", edgecolor="black")
    axes[2].set_title("Avg Slot Utilisation (%)")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=45)

    plt.suptitle("Overbooking Strategy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "strategy_comparison.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
