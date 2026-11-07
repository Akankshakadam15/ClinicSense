"""
ClinicSense – Model Definitions
Logistic Regression, Random Forest, XGBoost, LightGBM and a Stacking Ensemble.
All models return calibrated probability estimates.
"""

import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE, CLASS_WEIGHT, MODELS_DIR


# ── 1. Logistic Regression (Baseline) ─────────────────────────────────────────
def build_logistic_regression() -> Pipeline:
    """Scaled Logistic Regression – interpretable baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight=CLASS_WEIGHT,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ))
    ])


# ── 2. Random Forest ──────────────────────────────────────────────────────────
def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight=CLASS_WEIGHT,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ── 3. XGBoost ────────────────────────────────────────────────────────────────
def build_xgboost(scale_pos_weight: float = 4.0) -> XGBClassifier:
    """
    scale_pos_weight ≈ n_negative / n_positive accounts for class imbalance.
    The caller should compute and pass the actual ratio.
    Compatible with XGBoost 2.x and later (no deprecated parameters used).
    """
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


# ── 4. LightGBM ───────────────────────────────────────────────────────────────
def build_lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        class_weight=CLASS_WEIGHT,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


# ── 5. Stacking Ensemble ──────────────────────────────────────────────────────
def build_stacking_ensemble(scale_pos_weight: float = 4.0) -> StackingClassifier:
    """
    Level-0 estimators: LR, RF, XGB, LGBM
    Level-1 meta-learner: Logistic Regression
    """
    estimators = [
        ("lr",  build_logistic_regression()),
        ("rf",  build_random_forest()),
        ("xgb", build_xgboost(scale_pos_weight)),
        ("lgbm",build_lightgbm()),
    ]
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=3,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )


# ── Calibration wrapper ───────────────────────────────────────────────────────
def calibrate(model, X_val, y_val, method: str = "isotonic"):
    """
    Wrap a fitted model with isotonic or Platt (sigmoid) calibration.
    Returns a fitted CalibratedClassifierCV (prefit=True).
    """
    cal = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    cal.fit(X_val, y_val)
    return cal


# ── Persistence ───────────────────────────────────────────────────────────────
def save_model(model, name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"[Models] Saved → {path}")
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)
