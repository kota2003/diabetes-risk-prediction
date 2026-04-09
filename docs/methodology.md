# Methodology

## Predicting Diabetes Risk Using Machine Learning
### CDC BRFSS 2022–2024

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline to predict diabetes
risk from behavioural and demographic survey data. The goal is to identify the
most influential risk factors using a nationally representative, large-scale dataset
and to produce an interpretable model suitable for population-level screening.

---

## 2. Data Source

**CDC Behavioral Risk Factor Surveillance System (BRFSS)**
- Years: 2022, 2023, 2024
- Format: Fixed-width ASCII files with HTML codebooks
- Raw rows: 1,336,125 respondents across three survey years
- URL: https://www.cdc.gov/brfss/annual_data/annual_data.htm

BRFSS was selected over the commonly used Pima Indians dataset (UCI) for its
scale (1.25M vs 768 rows), demographic diversity (nationally representative vs
single ethnic group), recency (2022–2024 vs 1990s), and real-world data quality
challenges (special codes, structural absences, multi-year alignment).

---

## 3. Target Variable

`DIABETES` — binary classification (1 = Diabetes, 0 = No Diabetes)

Derived from BRFSS variable `DIABETE4`:
- Code 1 → 1 (diabetes)
- Code 3 → 0 (no diabetes)
- Excluded: code 2 (gestational diabetes), code 4 (pre-diabetes), codes 7/9 (don't know / refused)

Class distribution after cleaning: **14.4% positive / 85.6% negative**

---

## 4. Phase-by-Phase Methodology

### Phase 0 — Data Collection

- Downloaded ASCII fixed-width data files and HTML codebooks for 2022, 2023, 2024
- Built an automated codebook parser to extract column start positions and widths from HTML tables
- Extracted 22 target variables per year; concatenated into a single 1,336,125-row dataset
- Key engineering decision: column widths were defined explicitly per variable rather than
  inferred from codebook column gaps, which proved unreliable for one variable in one year

### Phase 1 — Data Understanding

- Profiled target distribution, missing value rates (pre- and post-recode), feature distributions
- Computed point-biserial correlations between each feature and the binary target
- Identified BRFSS special codes (88 = "none", 7/9/77/99 = "don't know / refused")
  that must be recoded before any analysis
- Ranked top predictors: GENHLTH (r=0.27), _AGEG5YR (r=0.23), DIFFWALK (r=0.22),
  _BMI5CAT (r=0.19), PHYSHLTH (r=0.17)
- Identified three variables with structural missingness justifying exclusion in Phase 2

### Phase 2 — Data Cleaning

Special code recoding:
- Days-count variables (PHYSHLTH, MENTHLTH, POORHLTH): 88 → 0 (means "none")
- All other variables: 7, 9, 77, 99 → NaN

Variables dropped:
| Variable | Reason |
|----------|--------|
| `BPHIGH6` | Present in 2023 only — 67.6% missing across combined dataset |
| `_CHOLCH3` | Present in 2023 only — 67.6% missing across combined dataset |
| `PREDIAB2` | 65.2% missing + direct target leakage |
| `SEXVAR` | Exact duplicate of `_SEX` (r=1.00) |
| `_STATE` | 50-level FIPS code — no ordinal meaning |
| `_RACE` | Structurally absent in 2022; imputing demographic identity is unsafe |

Imputation (all median except DIFFWALK):
| Variable | Method | Reason |
|----------|--------|--------|
| `INCOME3` | Median | Ordinal |
| `_AGEG5YR` | Median | 16.9% missing post-recode |
| `_BMI5CAT` | Median | Ordinal |
| `_SMOKER3` | Median | Ordinal |
| `PHYSHLTH` | Median | 4.5% missing post-recode; continuous |
| `MENTHLTH` | Median | 3.7% missing post-recode; continuous |
| `POORHLTH` | Median | Continuous |
| `DIFFWALK` | Mode | Binary (only two values) |

Output: `brfss_cleaned.csv` — 1,252,580 rows × 17 columns, 0 NaN

### Phase 3 — Feature Engineering

**Multicollinearity resolution:**
PHYSHLTH and POORHLTH had VIF of 2.18 and 2.05 respectively (r=0.70).
POORHLTH was dropped because PHYSHLTH ranked higher in Phase 1 predictor ranking (#5 vs #6).
High VIF values on binary/ordinal variables (CVDSTRK3=77, CVDINFR4=66) were identified
as artefacts of applying VIF to non-continuous inputs — no action taken.

**Temporal leakage prevention:**
The YEAR column was dropped. Although a chi-square test detected a statistically
significant prevalence shift across years (p=6.82e-20), the effect size was trivially
small (14.20% → 14.25% → 14.84%, range = 0.64pp). Including YEAR would risk
temporal leakage without meaningful predictive value.

**Binary recoding:**
BRFSS binary variables use 1/2 encoding (1=Yes, 2=No).
DIFFWALK, EXERANY2, CVDINFR4, CVDSTRK3, and _SEX were recoded to 0/1.

**Scaling:**
StandardScaler applied to continuous variables (PHYSHLTH, MENTHLTH) only.
Fit on training data; applied to both train and test. Scaler saved to `models/scaler.pkl`.

**Train/test split:**
80/20 stratified split (random_state=42). Class ratio preserved at 14.4% in both splits.

**Class imbalance:**
Two strategies were prepared and compared in Phase 4:
- SMOTE on training set → 50/50 balance (X_train_smote, y_train_smote)
- `scale_pos_weight` / `class_weight='balanced'` — model-level reweighting

### Phase 4 — Modeling

Six model variants trained across three algorithms and two imbalance strategies:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **XGB-Balanced** ⭐ | 0.6985 | 0.6246 | 0.7395 | 0.6138 | **0.8148** |
| LR-Balanced | 0.7171 | 0.6247 | 0.7313 | 0.6227 | 0.8040 |
| LR-SMOTE | 0.7054 | 0.6206 | 0.7273 | 0.6140 | 0.7989 |
| XGB-SMOTE | 0.7536 | 0.6198 | 0.6952 | 0.6309 | 0.7870 |
| RF-SMOTE | 0.7579 | 0.6009 | 0.6495 | 0.6111 | 0.7370 |
| RF-Balanced | 0.7790 | 0.5942 | 0.6169 | 0.6021 | 0.7260 |

All models evaluated on the same held-out test set (never resampled).
Primary metric: ROC-AUC (preferred over accuracy on imbalanced data).
Secondary metric: Recall/Sensitivity (minimising missed cases is critical in health screening).

**Key finding — imbalance strategy:**
`scale_pos_weight` / `class_weight='balanced'` outperformed SMOTE across all three
algorithm families. Synthetic minority samples appear not to generalise as well as
loss-function reweighting on large real-world survey data.

**Key finding — Random Forest:**
RF underperformed both LR and XGBoost on ROC-AUC (0.726–0.737 vs 0.799–0.815).
With unlimited depth and 14 ordinal/binary features, the 100-tree forest likely
memorised training data. RF model files were also 2.1–3.0 GB (excluded from Git).

### Phase 5 — Evaluation & Interpretation

**SHAP method:**
`shap.TreeExplainer` was used — it computes exact Shapley values by exploiting
XGBoost's tree structure, avoiding the approximations required by KernelSHAP.

**Sampling:**
- Background: 5,000 rows from X_train (random_state=42)
- Explanation set: 5,000 rows from X_test (random_state=42)

5,000 rows is sufficient for stable global importance estimates.

**Outputs:**
- Global importance bar chart (mean |SHAP| per feature)
- Beeswarm plot (direction and magnitude across all explanation samples)
- Waterfall plots for the most confident True Positive and True Negative
- Side-by-side comparison of XGBoost gain importance vs SHAP importance

---

## 5. Final Feature Set (14 features)

| Variable | Type | Description |
|----------|------|-------------|
| `GENHLTH` | Ordinal (1–5) | Self-rated general health (1=Excellent, 5=Poor) |
| `_AGEG5YR` | Ordinal (1–14) | Age group in 5-year intervals |
| `DIFFWALK` | Binary (0/1) | Difficulty walking or climbing stairs |
| `_BMI5CAT` | Ordinal (1–4) | BMI category (1=Underweight, 4=Obese) |
| `PHYSHLTH` | Continuous (scaled) | Days physical health not good (0–30) |
| `CVDINFR4` | Binary (0/1) | Heart attack history |
| `INCOME3` | Ordinal (1–11) | Annual household income bracket |
| `EXERANY2` | Binary (0/1) | Physical activity in past 30 days |
| `MENTHLTH` | Continuous (scaled) | Days mental health not good (0–30) |
| `CHECKUP1` | Ordinal (1–4, 8) | Time since last routine checkup |
| `CVDSTRK3` | Binary (0/1) | Stroke history |
| `EDUCA` | Ordinal (1–6) | Highest education level |
| `_SEX` | Binary (0/1) | Sex (1=Male, 0=Female) |
| `_SMOKER3` | Ordinal (1–4) | Smoking status (1=Current daily, 4=Never) |

---

## 6. Evaluation Metrics Rationale

| Metric | Role | Why used |
|--------|------|----------|
| ROC-AUC | Primary ranking metric | Threshold-independent; appropriate for imbalanced data |
| Recall (macro) | Secondary | Penalises missed cases — critical in health screening |
| Sensitivity (class 1) | Clinical metric | Direct measure of true positive rate for diabetes class |
| Specificity (class 0) | Clinical metric | True negative rate — measures false alarm burden |
| Precision | Reported for completeness | Useful if downstream cost of false positives is high |
| F1 | Reported for completeness | Harmonic mean of precision and recall |
| Accuracy | Reported for completeness | Misleading on imbalanced data; included for reference only |
