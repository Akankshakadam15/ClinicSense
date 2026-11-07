"""
ClinicSense – Report Writer
Generates reproducible Markdown and CSV reports in outputs/reports/.
All functions accept real results from project execution; they never
fabricate or hardcode metric values.

Output files are named with a timestamp to avoid overwriting prior runs.

Usage (from a notebook or script):
    from src.report_writer import write_model_report, write_simulation_report, write_rl_report
"""

import os
import json
import datetime
import pandas as pd
import numpy as np

from src.config import REPORTS_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── helpers ──────────────────────────────────────────────────────────────────
def _ts() -> str:
    """Short timestamp string for file naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _fig_md(filename: str, caption: str) -> str:
    """Relative Markdown image reference from the reports/ directory."""
    rel_path = os.path.join("..", "figures", filename).replace("\\", "/")
    return f"![{caption}]({rel_path})\n\n*{caption}*\n"


def _save(content: str, filename: str) -> str:
    """Write content to REPORTS_DIR and return absolute path."""
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[ReportWriter] Saved → {path}")
    return path


# ─── 1. Model Report ──────────────────────────────────────────────────────────
def write_model_report(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    optimal_threshold: float,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown model-comparison report.

    Parameters
    ----------
    comparison_df     : DataFrame produced by evaluation.comparison_table()
                        (indexed by model name; columns: accuracy, precision,
                         recall, f1, roc_auc, pr_auc, brier_score)
    best_model_name   : Name of the model chosen for deployment
    optimal_threshold : Cost-optimal classification threshold (float in [0,1])
    timestamp         : Whether to add a timestamp suffix to the filename

    Returns
    -------
    Absolute path to the written Markdown file.
    Raises ValueError if comparison_df is empty.
    """
    if comparison_df is None or len(comparison_df) == 0:
        raise ValueError("comparison_df is empty – no metrics to report.")

    now_str  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname    = f"model_report_{_ts()}.md" if timestamp else "model_report.md"

    # Also save raw metrics as CSV (for easy import/diff)
    csv_path = fname.replace(".md", ".csv")
    comparison_df.round(4).to_csv(os.path.join(REPORTS_DIR, csv_path))

    lines = [
        "# ClinicSense – Model Comparison Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Purpose",
        "",
        "This report documents the training and evaluation of five classification",
        "models for predicting patient no-shows.",
        "Class imbalance (~20% no-shows vs 80% shows) was handled with SMOTE on",
        "the training set and `class_weight='balanced'` parameters.",
        "Accuracy alone is **not used** as the primary metric.",
        "",
        "## 2. Model Comparison Table",
        "",
        comparison_df.round(4).to_markdown(),
        "",
        f"**Selected model:** {best_model_name}",
        "",
        f"**Cost-optimal classification threshold:** `{optimal_threshold:.4f}`",
        "(Chosen to minimise `cost_FN × FN + cost_FP × FP` where FN = missed",
        "no-show costs more than FP = unnecessary alert.)",
        "",
        "## 3. Key Metrics Explained",
        "",
        "| Metric | Why it matters here |",
        "|--------|---------------------|",
        "| ROC-AUC | Ranking quality across all thresholds |",
        "| PR-AUC | Precision-recall trade-off for the minority (no-show) class |",
        "| Recall | Fraction of actual no-shows correctly flagged |",
        "| Precision | Of flagged no-shows, fraction that truly are |",
        "| F1 | Harmonic mean of Precision and Recall |",
        "| Brier Score | Calibration quality of predicted probabilities |",
        "",
        "## 4. Plots",
        "",
        _fig_md("roc_curves.png",          "ROC Curves – All Models"),
        _fig_md("pr_curves.png",            "Precision-Recall Curves – All Models"),
        _fig_md("calibration_curve.png",    "Calibration Curve – Best Model"),
        _fig_md("threshold_tuning.png",     "Cost-Sensitive Threshold Selection"),
        _fig_md("shap_bar.png",             "Top-15 Features – Mean |SHAP| Value"),
        _fig_md("shap_summary.png",         "SHAP Beeswarm Summary Plot"),
        _fig_md("risk_stratification.png",  "Patient Risk Stratification"),
        "",
        "## 5. Known Limitations",
        "",
        "- Dataset from a single city (Vitória, Brazil); generalisability is limited.",
        "- Cost parameters (idle slot, waiting, overtime) are assumed values.",
        "- No reason for individual no-shows is recorded in the dataset.",
        "",
        "---",
        f"*Raw metrics also saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 2. Simulation Report ─────────────────────────────────────────────────────
def write_simulation_report(
    summary_df: pd.DataFrame,
    best_strategy: str,
    baseline_name: str = "No Overbooking (0)",
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown + CSV simulation/prescriptive-analytics report.

    Parameters
    ----------
    summary_df     : DataFrame from simulator.compare_strategies()
                     (indexed by strategy name)
    best_strategy  : Name of the cost-minimising strategy
    baseline_name  : Row name of the no-overbooking baseline
    timestamp      : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    Raises ValueError if summary_df is empty.
    """
    if summary_df is None or len(summary_df) == 0:
        raise ValueError("summary_df is empty – no simulation results to report.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"simulation_report_{_ts()}.md" if timestamp else "simulation_report.md"

    csv_path = fname.replace(".md", ".csv")
    summary_df.round(2).to_csv(os.path.join(REPORTS_DIR, csv_path))

    # Compute savings vs baseline
    saving_str = ""
    if baseline_name in summary_df.index and best_strategy in summary_df.index:
        baseline_cost = summary_df.loc[baseline_name, "Avg Total Cost"]
        best_cost     = summary_df.loc[best_strategy,  "Avg Total Cost"]
        saving        = baseline_cost - best_cost
        saving_pct    = saving / baseline_cost * 100 if baseline_cost > 0 else 0
        annual        = saving * 250
        saving_str = (
            f"The recommended strategy (**{best_strategy}**) saves "
            f"approximately **₹{saving:.0f}/day** ({saving_pct:.1f}%) "
            f"versus no overbooking — about **₹{annual:,.0f}/year** assuming "
            f"250 working days."
        )
    else:
        saving_str = f"Recommended strategy: **{best_strategy}**"

    lines = [
        "# ClinicSense – Overbooking Simulation Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Simulation Setup",
        "",
        "Monte Carlo simulation comparing three overbooking policy types:",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Replications per strategy | 5,000 |",
        "| Base slots per day | 20 |",
        "| Idle slot cost | ₹150 |",
        "| Patient waiting cost | ₹2/minute |",
        "| Doctor overtime cost | ₹5/minute |",
        "| Slot duration | 20 minutes |",
        "",
        "## 2. Strategy Comparison",
        "",
        summary_df.round(2).to_markdown(),
        "",
        saving_str,
        "",
        "## 3. Risk-Based Dynamic Overbooking – Method",
        "",
        "```",
        "extra_bookings = round( Σ (1 − P(attend_i)) for all booked patient i )",
        "               = round( expected no-shows for the day )",
        "```",
        "",
        "This adapts daily to the actual risk composition of the schedule,",
        "unlike a fixed percentage rule.",
        "",
        "## 4. Plots",
        "",
        _fig_md("strategy_comparison.png",        "Strategy Comparison – Cost, Idle Slots, Utilisation"),
        _fig_md("cost_distributions.png",          "Daily Cost Distributions by Strategy"),
        _fig_md("cost_heatmap.png",                "Sensitivity Analysis – Cost Heatmap"),
        _fig_md("dynamic_overbooking_detail.png",  "Risk-Based Dynamic Strategy Detail"),
        "",
        "## 5. Limitations",
        "",
        "- Waiting and overtime costs are proportional to overflow count × slot duration.",
        "  A full discrete-event simulation would model exact patient arrival times.",
        "- Cost parameters are assumptions; results should be validated with real clinic data.",
        "",
        "---",
        f"*Raw results also saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 3. RL Report ─────────────────────────────────────────────────────────────
def write_rl_report(
    eval_df: pd.DataFrame,
    comparison_dict: dict,
    n_episodes_trained: int,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown report for the Q-Learning agent evaluation.

    Parameters
    ----------
    eval_df             : DataFrame from rl_agent.evaluate_agent()
                          (columns: episode, reward, n_booked)
    comparison_dict     : {strategy_name: avg_reward} for all strategies
    n_episodes_trained  : number of training episodes
    timestamp           : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if eval_df is None or len(eval_df) == 0:
        raise ValueError("eval_df is empty – no RL evaluation results.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"rl_report_{_ts()}.md" if timestamp else "rl_report.md"

    avg_reward = eval_df["reward"].mean()
    std_reward = eval_df["reward"].std()
    avg_booked = eval_df["n_booked"].mean()

    # Build comparison table
    if comparison_dict:
        comp_df = pd.Series(comparison_dict, name="Avg Daily Reward (₹)").to_frame()
        comp_md = comp_df.round(2).to_markdown()
    else:
        comp_md = "*No rule-based comparison available.*"

    # CSV backup
    csv_path = fname.replace(".md", ".csv")
    eval_df.describe().round(2).to_csv(os.path.join(REPORTS_DIR, csv_path))

    lines = [
        "# ClinicSense – Reinforcement Learning Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Agent Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Algorithm | Tabular Q-Learning |",
        f"| Training episodes | {n_episodes_trained:,} |",
        "| State | (free_slots_bucket, expected_noshows_bucket) |",
        "| Actions | 0 = stop booking, 1 = accept next patient |",
        "| Reward | Revenue from attended appointments − idle/wait/overtime costs |",
        "| Exploration | ε-greedy (ε: 1.0 → 0.05) |",
        "",
        "## 2. Evaluation Results (Greedy Policy)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Avg reward per day | ₹{avg_reward:.2f} |",
        f"| Std reward | ₹{std_reward:.2f} |",
        f"| Avg bookings per day | {avg_booked:.2f} |",
        f"| Evaluation episodes | {len(eval_df):,} |",
        "",
        "## 3. Comparison with Rule-Based Strategies",
        "",
        comp_md,
        "",
        "## 4. Plots",
        "",
        _fig_md("rl_training_curve.png",     "Q-Learning Training Curve"),
        _fig_md("rl_vs_rules.png",            "RL Agent vs Rule-Based Strategies"),
        _fig_md("rl_reward_distribution.png", "RL Agent Reward Distribution"),
        "",
        "## 5. Interpretation",
        "",
        "- The Q-table has a small state space (MAX_EXTRA × ENS_BUCKETS = 11 × 5 = 55 states).",
        "- The learned policy accepts patients when free slots remain and expected",
        "  no-shows are high; it stops when overcrowding risk exceeds the revenue gain.",
        "- Rule-based Risk-Based Dynamic overbooking is recommended operationally",
        "  due to its simplicity and comparable performance.",
        "",
        "---",
        f"*Summary statistics saved to `{csv_path}`.*",
    ]
    return _save("\n".join(lines) + "\n", fname)


# ─── 4. Fairness Report ───────────────────────────────────────────────────────
def write_fairness_report(
    fairness_results: dict,
    timestamp: bool = True,
) -> str:
    """
    Write a Markdown fairness audit report.

    Parameters
    ----------
    fairness_results : dict from advanced_analytics.fairness_audit()
                       {attr: {demographic_parity_diff, equalized_odds_diff, metric_frame}}
    timestamp        : Add timestamp suffix to filename

    Returns
    -------
    Absolute path to the written Markdown file.
    """
    if not fairness_results:
        raise ValueError("fairness_results is empty – run fairness_audit() first.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname   = f"fairness_report_{_ts()}.md" if timestamp else "fairness_report.md"

    rows = []
    for attr, res in fairness_results.items():
        dpd = res.get("demographic_parity_diff", "N/A")
        eod = res.get("equalized_odds_diff", "N/A")
        rows.append({
            "Attribute":                    attr,
            "Demographic Parity Diff":      dpd,
            "Equalized Odds Diff":          eod,
            "DPD Verdict":  ("Fair" if abs(dpd) < 0.1 else
                             "Moderate" if abs(dpd) < 0.2 else "Unfair")
                            if isinstance(dpd, float) else "N/A",
        })
    table_md = pd.DataFrame(rows).set_index("Attribute").to_markdown()

    lines = [
        "# ClinicSense – Fairness Audit Report",
        f"*Generated: {now_str}*",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "The fairness audit checks whether the no-show risk model treats",
        "patient groups equitably across gender, age group, and scholarship status.",
        "",
        "**Thresholds used:**",
        "- |DPD| < 0.10 → Fair",
        "- 0.10 ≤ |DPD| < 0.20 → Moderate disparity",
        "- |DPD| ≥ 0.20 → Unfair",
        "",
        "## 2. Results",
        "",
        table_md,
        "",
        "## 3. Metrics Explained",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| Demographic Parity Difference | Difference in positive prediction rates across groups |",
        "| Equalized Odds Difference | Max difference in TPR and FPR across groups |",
        "",
        "## 4. Plot",
        "",
        _fig_md("fairness_audit.png", "Predicted vs Actual No-Show Rates by Group"),
        "",
        "## 5. Limitations",
        "",
        "- The audit uses observational data; disparities may reflect real differences",
        "  in no-show behaviour rather than model bias.",
        "- Only gender, age group, and scholarship status are examined.",
        "",
        "---",
    ]
    return _save("\n".join(lines) + "\n", fname)
