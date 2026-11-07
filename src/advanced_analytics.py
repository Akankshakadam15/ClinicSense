"""
ClinicSense – Advanced Analytics Module
Covers:
  1. SHAP explanations
  2. Patient segmentation (K-Means / Gaussian Mixture)
  3. SMS uplift modelling (T-Learner)
  4. Population Stability Index (feature drift)
  5. Fairness audit helpers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import warnings
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, RANDOM_STATE, N_CLUSTERS, SHAP_SAMPLE_SIZE

os.makedirs(FIGURES_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SHAP Explanations
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap_values(model, X_sample: pd.DataFrame, model_type: str = "tree"):
    """
    Compute SHAP values. model_type: 'tree' for RF/XGB/LGBM, 'linear' for LR.
    Returns shap_values array and explainer object.
    """
    try:
        import shap
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        # For binary classifiers shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return explainer, shap_values
    except Exception as e:
        print(f"[SHAP] Error: {e}")
        return None, None


def plot_shap_summary(shap_values, X_sample: pd.DataFrame,
                      model_name: str = "", save: bool = True):
    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
        plt.title(f"SHAP Summary Plot – {model_name}", fontsize=13)
        plt.tight_layout()
        if save:
            path = os.path.join(FIGURES_DIR,
                                f"shap_summary_{model_name.lower().replace(' ', '_')}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        print(f"[SHAP plot] Error: {e}")


def plot_shap_bar(shap_values, X_sample: pd.DataFrame,
                  model_name: str = "", save: bool = True):
    """Bar chart of mean absolute SHAP values (global feature importance)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    X_sample.columns,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=importance_df, x="importance", y="feature",
                palette="viridis", ax=ax)
    ax.set_title(f"Top 15 Features – Mean |SHAP| – {model_name}", fontsize=12)
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR,
                            f"shap_bar_{model_name.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return importance_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Patient Segmentation
# ══════════════════════════════════════════════════════════════════════════════
CLUSTER_FEATURES = [
    "age", "lead_time_days", "prior_no_show_rate",
    "prior_appointments", "sms_received", "condition_count",
]

CLUSTER_NAMES = {
    0: "Reliable Regulars",
    1: "High-Risk First-Timers",
    2: "Last-Minute Bookers",
    3: "Chronic Patients",
}


def segment_patients(df: pd.DataFrame,
                     n_clusters: int = N_CLUSTERS,
                     method: str = "kmeans") -> pd.DataFrame:
    """
    Cluster patients using K-Means or Gaussian Mixture.
    Returns df with 'cluster' and 'cluster_name' columns added.
    """
    feat_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                       n_init=10, max_iter=300)
    else:
        model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)

    labels = model.fit_predict(X_scaled)
    df = df.copy()
    df["cluster"]      = labels
    df["cluster_name"] = df["cluster"].map(
        CLUSTER_NAMES if n_clusters == 4 else {i: f"Cluster {i}" for i in range(n_clusters)}
    )
    return df, model, scaler


def plot_cluster_profiles(df: pd.DataFrame, save: bool = True):
    """Radar / bar chart showing no-show rate per cluster."""
    cluster_stats = (
        df.groupby("cluster_name")
        .agg(
            no_show_rate=("no_show", "mean"),
            avg_lead_time=("lead_time_days", "mean"),
            avg_age=("age", "mean"),
            avg_prior_noshows=("prior_no_show_rate", "mean"),
            count=("no_show", "count"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=cluster_stats, x="cluster_name", y="no_show_rate",
                palette="Set2", ax=axes[0])
    axes[0].set_title("No-Show Rate by Segment", fontsize=12)
    axes[0].set_ylabel("No-Show Rate")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=cluster_stats, x="cluster_name", y="avg_lead_time",
                palette="Set1", ax=axes[1])
    axes[1].set_title("Average Lead Time by Segment (days)", fontsize=12)
    axes[1].set_ylabel("Days")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle("Patient Behaviour Segments", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "cluster_profiles.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return cluster_stats


# ══════════════════════════════════════════════════════════════════════════════
# 3. SMS Uplift Modelling (T-Learner)
# ══════════════════════════════════════════════════════════════════════════════
def t_learner_uplift(df: pd.DataFrame, feature_cols: list,
                     treatment_col: str = "sms_received",
                     outcome_col: str = "no_show"):
    """
    T-Learner:
      - Train model_t1 on treated group  (SMS=1)  → predict P(show | SMS=1)
      - Train model_t0 on control group  (SMS=0)  → predict P(show | SMS=0)
      - Uplift = P(show | SMS=1) - P(show | SMS=0)
    Note: Outcome here is *attendance* (1 - no_show) for uplift direction.
    """
    from sklearn.ensemble import RandomForestClassifier
    # T-Learner uses two separate RF models; no cross-validation needed here.

    df = df.copy()
    df["attended"] = 1 - df[outcome_col]

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    X_cols = [c for c in feature_cols if c != treatment_col]

    clf_t1 = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                     class_weight="balanced", n_jobs=-1)
    clf_t0 = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                     class_weight="balanced", n_jobs=-1)

    clf_t1.fit(treated[X_cols], treated["attended"])
    clf_t0.fit(control[X_cols], control["attended"])

    X_all = df[X_cols]
    p_show_sms1 = clf_t1.predict_proba(X_all)[:, 1]
    p_show_sms0 = clf_t0.predict_proba(X_all)[:, 1]

    df["uplift_score"] = p_show_sms1 - p_show_sms0
    return df, clf_t1, clf_t0


def plot_uplift_distribution(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["uplift_score"], bins=50, kde=True, color="darkorange", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Distribution of SMS Uplift Scores (T-Learner)", fontsize=12)
    ax.set_xlabel("Uplift Score (P(show|SMS=1) – P(show|SMS=0))")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, "uplift_distribution.png")
        fig.savefig(path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Population Stability Index (Feature Drift)
# ══════════════════════════════════════════════════════════════════════════════
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.
    PSI < 0.1  : stable
    0.1 – 0.25 : moderate shift
    > 0.25     : significant shift
    """
    def _bucket(arr):
        counts, edges = np.histogram(arr, bins=bins)
        pcts = counts / len(arr)
        pcts = np.where(pcts == 0, 1e-6, pcts)
        return pcts, edges

    exp_pcts, edges = _bucket(expected)
    act_counts, _   = np.histogram(actual, bins=edges)
    act_pcts = act_counts / len(actual)
    act_pcts = np.where(act_pcts == 0, 1e-6, act_pcts)

    return float(np.sum((act_pcts - exp_pcts) * np.log(act_pcts / exp_pcts)))


def compute_psi_all_features(df_train: pd.DataFrame,
                              df_test:  pd.DataFrame,
                              feature_cols: list) -> pd.DataFrame:
    records = []
    for col in feature_cols:
        if df_train[col].dtype in ["float32", "float64", "int8", "int16", "int32", "int64"]:
            score = psi(df_train[col].dropna().values,
                        df_test[col].dropna().values)
            records.append({"feature": col, "PSI": round(score, 4),
                             "drift": ("Stable" if score < 0.1
                                       else "Moderate" if score < 0.25
                                       else "Significant")})
    return pd.DataFrame(records).sort_values("PSI", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Conformal Prediction (MAPIE ≥ 1.0)
# ══════════════════════════════════════════════════════════════════════════════
def conformal_predict(model, X_cal: pd.DataFrame, y_cal,
                      X_test: pd.DataFrame,
                      confidence: float = 0.90):
    """
    Wrap a pre-fitted model with conformal prediction using MAPIE ≥ 1.0.
    Uses SplitConformalClassifier with the prefit + conformalize workflow.

    Parameters
    ----------
    model      : a pre-fitted sklearn-compatible classifier
    X_cal      : calibration features (held-out, not used in model training)
    y_cal      : calibration labels
    X_test     : test features
    confidence : desired coverage level (default 0.90)

    Returns
    -------
    pred_sets   : boolean array (n_test, n_classes) – True where class is in set
    coverage    : float – empirical coverage on calibration data
    singleton_rate : float – fraction of singleton prediction sets (sharpness)
    """
    try:
        from mapie.classification import SplitConformalClassifier

        mapie = SplitConformalClassifier(
            estimator=model,
            confidence_level=confidence,
            prefit=True,
        )
        # In MAPIE ≥ 1.0 with prefit=True, use conformalize() instead of fit()
        mapie.conformalize(X_cal, y_cal)

        # predict_set returns (y_pred, prediction_sets) where
        # prediction_sets has shape (n_test, n_classes, 1)
        _, test_sets_raw = mapie.predict_set(X_test)
        # Squeeze the trailing alpha dimension → (n_test, n_classes)
        if test_sets_raw.ndim == 3:
            pred_sets = test_sets_raw[:, :, 0]
        else:
            pred_sets = test_sets_raw

        singleton_rate = float((pred_sets.sum(axis=1) == 1).mean())

        # Empirical coverage on calibration set
        _, cal_sets_raw = mapie.predict_set(X_cal)
        if cal_sets_raw.ndim == 3:
            cal_sets = cal_sets_raw[:, :, 0]
        else:
            cal_sets = cal_sets_raw
        y_cal_arr = np.asarray(y_cal)
        covered = np.array([cal_sets[i, y_cal_arr[i]] for i in range(len(y_cal_arr))])
        coverage = float(covered.mean())

        return pred_sets, coverage, singleton_rate

    except ImportError:
        print("[Conformal] mapie not installed. Run: pip install mapie")
        return None, None, None
    except Exception as e:
        print(f"[Conformal] Error: {e}")
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Fairness Audit (using Fairlearn)
# ══════════════════════════════════════════════════════════════════════════════
def fairness_audit(y_true: pd.Series,
                   y_pred: np.ndarray,
                   sensitive_features: pd.DataFrame) -> dict:
    """
    Compute demographic parity difference and equalized odds difference
    for each sensitive attribute.
    Returns a dict of {attribute: MetricFrame}.
    """
    try:
        from fairlearn.metrics import (
            MetricFrame, demographic_parity_difference,
            equalized_odds_difference, selection_rate
        )
        from sklearn.metrics import accuracy_score, recall_score

        results = {}
        for col in sensitive_features.columns:
            mf = MetricFrame(
                metrics={
                    "accuracy":       accuracy_score,
                    "recall":         lambda yt, yp: recall_score(yt, yp, zero_division=0),
                    "selection_rate": selection_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features[col],
            )
            dpd = demographic_parity_difference(y_true, y_pred,
                                                sensitive_features=sensitive_features[col])
            eod = equalized_odds_difference(y_true, y_pred,
                                            sensitive_features=sensitive_features[col])
            results[col] = {
                "metric_frame": mf,
                "demographic_parity_diff": round(dpd, 4),
                "equalized_odds_diff":     round(eod, 4),
            }
        return results
    except ImportError:
        print("[Fairness] fairlearn not installed. Run: pip install fairlearn")
        return {}
