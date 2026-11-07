# ClinicSense – Viva Preparation Guide

**Student:** Kadam Akanksha Dinkar | **B.Sc. Data Science Honours**  
**Project:** ClinicSense: Patient No-Show Risk Engine & Dynamic Overbooking Optimizer

---

## 1. Problem and Motivation

**Q: What real-world problem does ClinicSense solve?**  
Outpatient clinics lose ~20% of booked appointments to no-shows. Every missed slot wastes doctor time, cuts revenue, and delays care. Clinics either ignore the problem or use a fixed overbooking rule that ignores individual patient risk.

**Q: Why is this a Predictive Analytics project, not just an ML project?**  
PA goes beyond building a model. ClinicSense covers: data collection → EDA → preprocessing → feature engineering → modelling → evaluation → **interpretation** (SHAP, risk labels) → **decision support** (simulator, policy recommendation). The prescriptive overbooking simulator is the key differentiator.

---

## 2. Dataset

**Q: What dataset is used?**  
Kaggle "Medical Appointment No Shows" – public clinics in Vitória, Brazil. ~110,000 rows, 14 columns, ~20% no-shows (imbalanced).

**Q: What data quality issues were found?**  
Negative ages, appointments scheduled after the visit date, and duplicate appointment IDs. All removed in `data_loader.py`.

**Q: What is the target variable?**  
`no_show`: 1 if patient missed the appointment, 0 if they attended.

---

## 3. Feature Engineering

**Q: What features did you engineer and why?**  

| Feature | Rationale |
|---------|-----------|
| `lead_time_days` | Longer wait → higher no-show risk (confirmed by Mann-Whitney test) |
| `prior_no_show_rate` | Past behaviour strongly predicts future (patient history) |
| `prior_appointments` | First-timers have less commitment |
| `age_group_enc` | Non-linear relationship with no-show rate |
| `appt_weekday`, `is_monday`, `is_friday` | Weekday no-show patterns exist |
| `condition_count` | Patients with chronic conditions attend more reliably |
| `same_day_booking` | Same-day bookings almost never result in no-shows |
| `neighbourhood_freq` | Frequency encoding for high-cardinality categorical |

**Q: How did you handle the neighbourhood variable?**  
Frequency encoding (replaces each neighbourhood with its proportion of total appointments). This preserves information without creating 81 dummy variables.

**Q: What is VIF and why did you compute it?**  
Variance Inflation Factor detects multicollinearity. VIF > 10 indicates a feature can be predicted from others, making model coefficients unstable. All features had VIF ≤ 10, so no removal was needed.

---

## 4. Handling Class Imbalance

**Q: Why is the dataset imbalanced and how did you handle it?**  
~80% show / ~20% no-show. If ignored, a model that always predicts "show" gets 80% accuracy but zero utility.

Approaches used:
- **SMOTE** on training data only (never on test data — data leakage prevention)
- **`class_weight='balanced'`** in LR, RF, LightGBM
- **`scale_pos_weight`** in XGBoost
- **Evaluation without accuracy** — used Recall, F1, ROC-AUC, PR-AUC, Brier Score

**Q: Why SMOTE only on training data?**  
Applying SMOTE to test data would create synthetic test samples that don't represent real patients, giving an optimistic and misleading evaluation.

---

## 5. Models and Comparison

**Q: Why five models?**  
Logistic Regression as the interpretable baseline; RF, XGBoost, LightGBM as strong tree ensembles; Stacking as the most complex ensemble. This gives a spectrum from interpretable to powerful.

**Q: Which model performed best and why?**  
LightGBM (tuned + calibrated) achieved the best balance of ROC-AUC and PR-AUC. Tree ensembles handle non-linear feature interactions (e.g., lead_time × prior_no_show_rate) that logistic regression misses.

**Q: What is probability calibration and why did you use it?**  
Raw classifier probabilities may not reflect true likelihoods (e.g., a model might predict 0.8 when the true rate is only 0.6). Isotonic calibration (via MAPIE SplitConformalClassifier or sklearn CalibratedClassifierCV) adjusts probabilities to match observed frequencies, which is critical for the overbooking simulator.

**Q: What is a cost-sensitive threshold?**  
Instead of 0.5, the threshold is chosen to minimise `cost_FN × FN + cost_FP × FP`. Missing a no-show (FN) causes an idle slot costing ₹150; a false alarm (FP) causes a minor admin overhead (~₹20). The optimal threshold is lower than 0.5, catching more no-shows at the cost of more false alarms.

---

## 6. SHAP Interpretability

**Q: What is SHAP and how did you use it?**  
SHapley Additive exPlanations assign each feature a contribution to each individual prediction. The TreeExplainer is efficient for gradient boosting models. Used to:
1. Rank global feature importance (mean |SHAP|)
2. Show dependence plots (how feature value affects prediction)
3. Explain individual high-risk predictions with waterfall charts

**Q: What were the top no-show predictors?**  
Lead time, prior no-show rate, prior appointments, SMS reminder, age group, and same-day booking flag — broadly matching domain knowledge and literature.

---

## 7. Overbooking Simulator

**Q: How does the Monte Carlo simulator work?**  
For each simulated day:
1. Sample a pool of patients with predicted attendance probabilities
2. Apply an overbooking strategy to decide how many to book
3. Simulate arrivals: each patient attends with probability `attend_prob_i`
4. Compute idle slots, overflow, and total cost
5. Repeat 5,000 times and average

**Q: What three strategies were compared?**  
1. No overbooking (baseline)
2. Fixed overbooking (+N extra patients, same every day)
3. Risk-based dynamic: extra_bookings = round(Σ(1 - attend_prob_i))

**Q: Why is risk-based dynamic better?**  
It adapts to the actual risk composition of each day's schedule. On a day with many high-risk patients, it books more extras. On a day with mostly reliable patients, it books fewer — avoiding unnecessary overcrowding.

**Q: What are the cost assumptions?**  
Idle slot: ₹150, patient waiting: ₹2/min, doctor overtime: ₹5/min, slot duration: 20 min. These are stated assumptions since the dataset has no real cost data.

---

## 8. Advanced Analytics

**Q: What is conformal prediction (MAPIE)?**  
Instead of a point probability, conformal prediction provides a prediction *set* with guaranteed coverage. At 90% confidence, at least 90% of test predictions contain the true label. Used here with MAPIE 1.6's `SplitConformalClassifier`.

**Q: What is the T-Learner uplift model?**  
Two separate models: one trained on SMS-sent patients, one on no-SMS patients. The difference in predicted attendance probability is the individual treatment effect (uplift). Patients with high positive uplift benefit most from being sent an SMS reminder.

**Q: What is PSI and what did it show?**  
Population Stability Index measures distributional drift between training and test periods. PSI < 0.1 = stable. The analysis confirmed all key features remained stable across the time split, meaning the model is robust over time.

**Q: What did the fairness audit find?**  
Demographic Parity Difference and Equalized Odds Difference were computed across gender, age group, and scholarship status using Fairlearn. Values < 0.1 indicate the model treats groups equitably.

---

## 9. Reinforcement Learning

**Q: What is the RL setup?**  
- **State:** (free_slots_bucket, expected_no-shows_bucket) — 55 discrete states
- **Action:** 0 = stop accepting bookings, 1 = accept next patient
- **Reward:** Revenue from attended appointments minus idle/wait/overtime costs
- **Algorithm:** Tabular Q-Learning with ε-greedy exploration

**Q: How does the RL agent compare to rule-based strategies?**  
After 3,000 episodes, the Q-Learning agent learns a policy competitive with risk-based dynamic overbooking. The rule-based strategy is still recommended operationally due to its simplicity and ex
plainability.

---

## 10. Limitations and Future Work

**Limitations:**
- Single-region dataset (Vitória, Brazil) — may not generalise
- Cost parameters are assumed, not real financial data
- No reason for individual no-shows is recorded
- SMOTE creates synthetic minority samples that don't represent real patients

**Future improvements:**
- Multi-clinic / multi-region data for better generalisability
- Real cost data from clinic management systems
- Online/incremental learning to adapt to drift
- DQN instead of tabular Q-Learning for larger state spaces
- A/B testing the SMS uplift recommendation in a live clinic
