"""
ClinicSense – Model Evaluation
Computes all required metrics, plots confusion matrices, ROC and PR curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import os

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    brier_score_loss,
)
from src.config import FIGURES_DIR, COST_IDLE_SLOT, COST_WAITING_MINUTE

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Core metrics ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, model_name: str = "") -> dict:
    """Return a dict of all classification metrics."""
    metrics = {
        "model":        model_name,
        "accuracy":     accuracy_score(y_true, y_pred),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":      roc_auc_score(y_true, y_prob),
        "pr_auc":       average_precision_score(y_true, y_prob),
        "brier_score":  brier_score_loss(y_true, y_prob),
    }
    return metrics


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str = "", save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Show", "No-Show"],
                yticklabels=["Show", "No-Show"], ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# ── ROC curve ─────────────────────────────────────────────────────────────────
def plot_roc_curves(models_results: dict, save: bool = True):
    """
    models_results : {model_name: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_prob) in models_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "roc_curves.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Precision-Recall curve ────────────────────────────────────────────────────
def plot_pr_curves(models_results: dict, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_prob) in models_results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves – All Models", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "pr_curves.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Comparison table ─────────────────────────────────────────────────────────
def comparison_table(metrics_list: list) -> pd.DataFrame:
    df = pd.DataFrame(metrics_list)
    df = df.set_index("model")
    df = df.round(4)
    return df


# ── Cost-sensitive threshold tuning ──────────────────────────────────────────
def find_cost_optimal_threshold(y_true, y_prob,
                                 cost_fn: float = None,
                                 cost_fp: float = 20) -> float:
    """
    Sweep thresholds and find the one minimising expected cost:
      cost = cost_fn * FN + cost_fp * FP
    cost_fn : cost of missing a no-show (defaults to COST_IDLE_SLOT from config)
    cost_fp : cost of a false alarm (unnecessary overbook)
    """
    if cost_fn is None:
        cost_fn = COST_IDLE_SLOT
    thresholds = np.linspace(0.01, 0.99, 200)
    best_cost, best_thresh = np.inf, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_cost = cost_fn * fn + cost_fp * fp
        if total_cost < best_cost:
            best_cost   = total_cost
            best_thresh = t
    return best_thresh, best_cost


# ── Print full report ─────────────────────────────────────────────────────────
def print_report(y_true, y_pred, model_name: str = ""):
    print(f"\n{'='*60}")
    print(f"  Classification Report – {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Show", "No-Show"]))
