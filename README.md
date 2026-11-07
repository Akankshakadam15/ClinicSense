# ClinicSense: Patient No-Show Risk Engine & Dynamic Overbooking Optimizer

**Student:** Kadam Akanksha Dinkar  
**Programme:** B.Sc. Data Science Honours  
**Credits:** 6 | **Total Marks:** 150

---

## Project Overview

ClinicSense is an end-to-end **Predictive and Prescriptive Analytics** system for
outpatient clinic operations. It predicts patient no-show probability, explains
predictions with SHAP, segments patients into behaviour clusters, models SMS
reminder uplift, simulates dynamic overbooking policies with Monte Carlo methods,
audits model fairness, and includes an optional Q-Learning scheduling agent.

---

## Main Features

| Part | Description |
|------|-------------|
| Part 1 | No-Show Risk Engine – 5 ML models (LR, RF, XGBoost, LightGBM, Stacking Ensemble), calibration, cost-sensitive threshold |
| Part 2 | Dynamic Overbooking Simulator – Monte Carlo, 3 strategies, cost heatmap |
| Part 3 | Advanced Analytics – SHAP, K-Means/GMM clustering, T-Learner uplift, PSI drift, MAPIE conformal prediction, Fairlearn fairness audit |
| Part 4 | Reinforcement Learning – Tabular Q-Learning overbooking agent (optional) |

---

## Folder Structure

```
ClinicSense/
├── data/                              # Raw CSV and generated artefacts
│   ├── KaggleV2-May-2016.csv          # ← you must download this
│   ├── splits.joblib                  # written by NB02
│   ├── smote_splits.joblib            # written by NB02
│   ├── processed_data.csv             # written by NB02
│   ├── model_probs.joblib             # written by NB03
│   └── best_model_output.joblib       # written by NB04
├── notebooks/
│   ├── 01_data_collection_eda.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_predictive_modelling.ipynb
│   ├── 04_model_evaluation_optimization.ipynb
│   ├── 05_prediction_interpretation.ipynb
│   ├── 06_overbooking_simulator.ipynb
│   ├── 07_advanced_analytics.ipynb
│   └── 08_reinforcement_learning.ipynb
├── src/
│   ├── config.py                      # Central constants and paths
│   ├── data_loader.py                 # Load and validate CSV
│   ├── preprocessing.py               # Outlier capping, encoding, splits
│   ├── feature_engineering.py         # Lead time, history, age groups, etc.
│   ├── models.py                      # LR, RF, XGBoost, LightGBM, Stacking
│   ├── evaluation.py                  # Metrics, ROC/PR curves, threshold tuning
│   ├── simulator.py                   # Monte Carlo overbooking simulator
│   ├── advanced_analytics.py          # SHAP, clustering, uplift, PSI, fairness, conformal
│   ├── rl_agent.py                    # Q-Learning agent and environment
│   └── report_writer.py              # Writes Markdown + CSV reports to outputs/reports/
├── dashboard/
│   └── app.py                         # Streamlit demo dashboard
├── tests/
│   ├── conftest.py                    # Shared fixtures (synthetic data, no CSV needed)
│   ├── test_config.py
│   ├── test_feature_engineering.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_simulator.py
│   ├── test_rl_agent.py
│   ├── test_advanced_analytics.py
│   ├── test_conformal.py
│   └── test_report_writer.py
├── outputs/
│   ├── figures/                       # All saved plots (PNG)
│   ├── models/                        # Serialised model files (.joblib)
│   └── reports/                       # Markdown + CSV reports
├── .gitignore
├── pytest.ini
├── requirements.txt
├── run_all.py                         # Execute all notebooks end-to-end
└── README.md
```

---

## Supported Python Version

**Python 3.10** (tested). Python 3.11 should also work with the pinned versions
in `requirements.txt`.

---

## Installation (Windows PowerShell)

### 1. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs all required packages with pinned versions, including `tabulate`
(needed for Markdown report tables).

---

## Dataset Setup

1. Go to: <https://www.kaggle.com/datasets/joniarroba/noshowappointments>
2. Download `KaggleV2-May-2016.csv`
3. Place it in the `data/` folder:

```
ClinicSense\
└── data\
    └── KaggleV2-May-2016.csv   ← required
```

Without this file, Notebooks 01–08 will raise a `FileNotFoundError`. The test
suite does **not** require the CSV — it uses synthetic data.

---

## Notebook Execution Order and Dependencies

Run notebooks **in order**. Each notebook writes intermediate files that later
notebooks depend on.

| Notebook | Reads from `data/` | Writes to `data/` |
|----------|-------------------|--------------------|
| NB01 – EDA | `KaggleV2-May-2016.csv` | — |
| NB02 – Preprocessing | `KaggleV2-May-2016.csv` | `splits.joblib`, `smote_splits.joblib`, `processed_data.csv` |
| NB03 – Modelling | `splits.joblib`, `smote_splits.joblib` | `model_probs.joblib`, models in `outputs/models/` |
| NB04 – Evaluation | `splits.joblib`, `smote_splits.joblib`, models | `best_model_output.joblib` |
| NB05 – Interpretation | `splits.joblib`, `best_model_output.joblib`, models | figures in `outputs/figures/` |
| NB06 – Simulator | `best_model_output.joblib` | `outputs/reports/simulation_report_*.md` |
| NB07 – Advanced | `processed_data.csv`, `splits.joblib`, `best_model_output.joblib`, models | figures, `segmented_data.csv` |
| NB08 – RL (optional) | `best_model_output.joblib` | `outputs/reports/rl_report_*.md` |

### Run all notebooks automatically

```powershell
python run_all.py
```

Or run specific notebooks:

```powershell
python run_all.py --nb 01 02 03
```

This requires `nbconvert` and an installed Jupyter kernel (`ipykernel`). Install
them if missing:

```powershell
pip install nbconvert ipykernel
python -m ipykernel install --user --name python3
```

---

## Running Tests

```powershell
python -m pytest
```

Or with verbose output:

```powershell
python -m pytest -v
```

**Tests do not require the Kaggle CSV.** All fixtures use synthetic data.

Current result: **72 passed, 0 failed** (as of last verified run).

---

## Launching the Dashboard

```powershell
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. It uses a simplified rule-based
probability model for demonstration. For production use, load the trained model
from `outputs/models/best_model_calibrated.joblib`.

---

## Report Generation

Reports are written automatically during notebook execution. You can also call
the report writer directly from Python:

```python
from src.report_writer import write_model_report, write_simulation_report
from src.evaluation import comparison_table

# After running NB03/NB04 and having metrics:
report_path = write_model_report(
    comparison_df=comp_table,          # from evaluation.comparison_table()
    best_model_name="LightGBM (Tuned+Cal)",
    optimal_threshold=0.35,
)
print("Report saved to:", report_path)
```

Reports are written to `outputs/reports/` with timestamps:
- `model_report_YYYYMMDD_HHMMSS.md` + companion `.csv`
- `simulation_report_YYYYMMDD_HHMMSS.md` + companion `.csv`
- `rl_report_YYYYMMDD_HHMMSS.md` + companion `.csv`
- `fairness_report_YYYYMMDD_HHMMSS.md`

---

## Known Limitations and Troubleshooting

### Dataset
- The CSV is from a single city (Vitória, Brazil); generalisability is limited.
- Cost parameters (idle slot, waiting, overtime) are assumed; not real financial data.

### Dependencies
- `lightgbm` may print a CPU-detection warning on Windows — this is harmless.
- `mapie 1.6.0` uses `SplitConformalClassifier` with `.conformalize()` (not `.fit()`).
  The `conformal_predict()` function in `src/advanced_analytics.py` handles this correctly.
- `xgboost 3.2.0` no longer accepts `use_label_encoder` — removed from `build_xgboost()`.

### Notebook execution
- NB03 (stacking ensemble training) may take 10–20 minutes on modest hardware.
- NB04 Optuna search (50 trials) adds a few minutes.
- If a notebook fails, intermediate `.joblib` files may be incomplete; re-run
  from NB01 to ensure a clean state.

### Conformal prediction (NB07)
- Uses MAPIE 1.6.0 `SplitConformalClassifier`.
- Requires `lightgbm_tuned.joblib` from NB04 to be present in `outputs/models/`.

---

## PA Guidelines Coverage

| Phase | Notebook | Status |
|-------|----------|--------|
| 1 – Problem Definition | Synopsis + NB01 | ✅ |
| 2 – Data Collection | NB01 | ✅ |
| 3 – EDA + Statistical Analysis | NB01 | ✅ |
| 4 – Preprocessing + Feature Engineering | NB02 | ✅ |
| 5 – Predictive Modelling (≥2 models) | NB03 (5 models) | ✅ |
| 6 – Evaluation + Optimization | NB04 | ✅ |
| 7 – Prediction + Interpretation | NB05 + NB06 | ✅ |
| 8 – Conclusion + Future Scope | Summary sections in each NB | ✅ |
| Advanced – Simulation | NB06 | ✅ |
| Advanced – Clustering | NB07 | ✅ |
| Advanced – Uplift + Fairness + Drift | NB07 | ✅ |
| Optional – Reinforcement Learning | NB08 | ✅ |

---

## Key Technologies

Python 3.10, Pandas, NumPy, Scikit-learn, XGBoost 3.x, LightGBM, SHAP,
imbalanced-learn (SMOTE), Optuna, MAPIE, Fairlearn, scikit-uplift,
Streamlit, Matplotlib, Seaborn, SciPy, statsmodels, pytest

---

## Viva Preparation

See [`VIVA_PREPARATION.md`](VIVA_PREPARATION.md) for Q&A covering all project
phases: problem definition, dataset, feature engineering, class imbalance,
model selection, SHAP, simulator, advanced analytics, RL, and limitations.
