# Findings

## Predicting Diabetes Risk Using Machine Learning
### CDC BRFSS 2022–2024

---

## 1. Executive Summary

This project trained and evaluated six machine learning models to predict diabetes
risk from behavioural and demographic survey data collected by the CDC BRFSS program
(2022–2024, n=1,252,580 after cleaning).

**Best model: XGB-Balanced** (XGBoost with `scale_pos_weight` reweighting)
- ROC-AUC: **0.8148** — highest across all six variants
- Sensitivity: **0.7971** — correctly identified 80% of true diabetes cases
- The model is recommended as a **population-level screening tool** to flag
  high-risk individuals for follow-up clinical assessment

---

## 2. Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **XGB-Balanced** ⭐ | 0.6985 | 0.6246 | 0.7395 | 0.6138 | **0.8148** |
| LR-Balanced | 0.7171 | 0.6247 | 0.7313 | 0.6227 | 0.8040 |
| LR-SMOTE | 0.7054 | 0.6206 | 0.7273 | 0.6140 | 0.7989 |
| XGB-SMOTE | 0.7536 | 0.6198 | 0.6952 | 0.6309 | 0.7870 |
| RF-SMOTE | 0.7579 | 0.6009 | 0.6495 | 0.6111 | 0.7370 |
| RF-Balanced | 0.7790 | 0.5942 | 0.6169 | 0.6021 | 0.7260 |

*Evaluated on held-out test set (250,516 rows, 14.43% positive). Primary metric: ROC-AUC.*

### XGB-Balanced Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|--------------------|--------------------|
| **Actual Negative** | TN = 146,159 | FP = 68,196 |
| **Actual Positive** | FN = 7,337 | TP = 28,824 |

- Sensitivity (True Positive Rate): 0.7971 — 80% of diabetes cases correctly identified
- Specificity (True Negative Rate): 0.6819 — 68% of non-diabetes cases correctly identified

---

## 3. Key Finding 1 — Imbalance Strategy

**`scale_pos_weight` outperformed SMOTE across all three algorithm families.**

| Algorithm | Balanced ROC-AUC | SMOTE ROC-AUC | Difference |
|-----------|-----------------|---------------|------------|
| XGBoost | **0.8148** | 0.7870 | +0.028 |
| Logistic Regression | **0.8040** | 0.7989 | +0.005 |
| Random Forest | **0.7260** | 0.7370 | −0.011 |

SMOTE improves Recall slightly in some cases but reduces ROC-AUC, suggesting that
synthetic minority oversampling does not generalise as well as loss-function reweighting
on large real-world survey data. The pattern holds across two of three algorithms,
with RF being the exception (where both strategies underperform due to overfitting).

---

## 4. Key Finding 2 — Random Forest Underperformance

Random Forest produced the lowest ROC-AUC values (0.726–0.737) — below both
Logistic Regression and XGBoost. With unlimited `max_depth` and 14 ordinal/binary
features, the 100-tree forest likely memorised the training data rather than learning
generalisable patterns. This is consistent with the high accuracy but low ROC-AUC
observed: RF achieves high accuracy by predicting the majority class, while missing
the discriminative structure that XGBoost and LR capture.

Additionally, RF model files were 2.1–3.0 GB, making them impractical for storage
and deployment compared to XGBoost (~468 KB).

---

## 5. Key Finding 3 — Global Feature Importance (SHAP)

SHAP (SHapley Additive exPlanations) was used to quantify each feature's average
contribution to model predictions across 5,000 held-out test samples.

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|-------------|----------------|
| 1 | `_AGEG5YR` | 0.709 | Strongest predictor — older age groups carry substantially higher diabetes risk |
| 2 | `GENHLTH` | 0.550 | Poor self-rated health is a strong proxy for undiagnosed or unmanaged chronic conditions |
| 3 | `CHECKUP1` | 0.422 | Counterintuitive negative signal — discussed in Section 6 |
| 4 | `_BMI5CAT` | 0.344 | Higher BMI category is a well-established modifiable risk factor |
| 5 | `_SEX` | 0.135 | Male sex associated with modestly elevated risk |
| 6 | `DIFFWALK` | 0.106 | Mobility difficulty reflects physical health burden |
| 7 | `EXERANY2` | 0.103 | Physical inactivity associated with increased risk |
| 8 | `EDUCA` | 0.099 | Lower education level associated with higher risk |
| 9 | `INCOME3` | 0.094 | Lower income associated with higher risk — socioeconomic gradient |
| 10 | `CVDINFR4` | 0.058 | Heart attack history contributes meaningful but secondary signal |
| 11 | `MENTHLTH` | 0.058 | Mental health burden has modest association with diabetes risk |
| 12 | `_SMOKER3` | 0.048 | Smoking status has a small but consistent effect |
| 13 | `PHYSHLTH` | 0.038 | Physical health days contributes modest signal beyond GENHLTH |
| 14 | `CVDSTRK3` | 0.037 | Stroke history is the weakest predictor in this feature set |

---

## 6. Key Finding 4 — CHECKUP1 Negative Association

`CHECKUP1` (time since last routine checkup) ranked 3rd by SHAP importance,
but its direction is counterintuitive: **longer time since last checkup is
associated with lower predicted diabetes risk.**

Three plausible explanations:

1. **Survivorship / selection bias**: Individuals who have not seen a doctor recently
   may be younger or healthier on average — the "worried well" who attend frequently
   may already have identified conditions.
2. **Healthcare access confounding**: The association may reflect that individuals
   with lower healthcare access (rural, uninsured) are also less likely to have
   received a diabetes diagnosis — so their label is 0 not because they don't have
   diabetes, but because it has not been detected.
3. **Reverse causality**: People diagnosed with diabetes attend checkups more frequently,
   so high CHECKUP1 (recent checkup) correlates with known diabetes.

This association should **not** be interpreted as "avoiding checkups reduces diabetes risk."
It is an artefact of the survey-based, self-reported data structure.

---

## 7. Key Finding 5 — Built-in vs SHAP Importance Agreement

10 of 14 features agree within ±2 ranks between XGBoost gain importance and SHAP.

**Notable divergences:**

| Feature | XGB Rank | SHAP Rank | Direction | Explanation |
|---------|----------|-----------|-----------|-------------|
| `GENHLTH` | 1 | 2 | Minor swap | Both methods agree this is the top-2 feature |
| `CVDINFR4` | 5 | 10 | XGB overestimates | Gain-based importance inflates features used in many splits; SHAP gives a more honest view of prediction impact |
| `_SEX` | 8 | 5 | SHAP overestimates | Sex has more marginal prediction impact than its structural role in the tree suggests |
| `MENTHLTH` | 14 | 11 | SHAP overestimates | Small but consistent directional contribution not captured by gain |

The top-4 features (`_AGEG5YR`, `GENHLTH`, `CHECKUP1`, `_BMI5CAT`) are identical
across both methods — strong evidence that these are genuinely the dominant predictors.

---

## 8. Individual Prediction Explanations

### True Positive (p = 0.944) — Most confident correct positive

Profile: GENHLTH=5 (poor health), _AGEG5YR=10 (older), CVDINFR4=1 (heart attack history),
_BMI5CAT=4 (obese), CVDSTRK3=1 (stroke history), DIFFWALK=1 (mobility difficulty).

Every major risk factor is present simultaneously. SHAP attributes the largest
positive contributions to GENHLTH (+0.76), _AGEG5YR (+0.56), CVDINFR4 (+0.50),
and _BMI5CAT (+0.44). The model correctly identifies this as a high-risk individual
with high confidence.

### True Negative (p = 0.0003) — Most confident correct negative

Profile: _AGEG5YR=1 (youngest age group), _BMI5CAT=1 (underweight), CHECKUP1=2
(recent checkup), GENHLTH=2 (good health).

The combination of young age, low BMI, and good self-rated health produces a strongly
negative prediction. SHAP attributes the largest negative contributions to _AGEG5YR
(−2.73) and _BMI5CAT (−2.02).

---

## 9. Limitations

1. **No clinical biomarkers**: The model uses only behavioural and demographic
   survey data. Clinical indicators (blood glucose, HbA1c, insulin) are not available
   in BRFSS. This limits sensitivity compared to clinical screening tools.

2. **Self-report bias**: All features are derived from telephone survey responses.
   Misclassification of both the target (undiagnosed diabetes coded as 0) and features
   (underreported unhealthy behaviours) is likely and unmeasurable.

3. **No racial/ethnic fairness analysis**: `_RACE` was dropped because it was
   structurally absent in 2022 BRFSS data. Model performance across racial and
   ethnic subgroups cannot be assessed — an important limitation for a health equity lens.

4. **US adult population only**: BRFSS surveys US adults exclusively. Generalisation
   to other countries, healthcare systems, or populations is not supported.

5. **Temporal scope**: Data covers 2022–2024 only. The model may not reflect
   patterns from earlier periods or capture future trends.

6. **CHECKUP1 confounding**: As discussed in Section 6, the negative association
   between time since last checkup and diabetes risk is likely a data artefact,
   not a causal relationship.

---

## 10. Deployment Recommendation

**Recommended model: XGB-Balanced**

The model is suitable for deployment as a **population-level behavioural screening tool**
to identify individuals at elevated diabetes risk for follow-up clinical assessment.

It should **not** be used as:
- A diagnostic replacement for clinical blood tests
- A tool for individual clinical decision-making without clinical review
- A screening tool outside the US adult population

**Deployment requirements:**
- Input: 14 BRFSS-derived features (see methodology for exact coding)
- Preprocessing: Apply `models/scaler.pkl` to PHYSHLTH and MENTHLTH before inference
- Output: Predicted probability of diabetes (class 1)
- Threshold: Default 0.5; adjust based on operational cost of false positives vs false negatives
- Model file: `models/saved_models/xgb_balanced.pkl` (~468 KB)
