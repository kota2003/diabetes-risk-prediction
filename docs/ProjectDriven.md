# ProjectDriven.md — Living Project Log
## Predicting Diabetes Risk Using Machine Learning

> **Version note:** This document was originally drafted for the Pima Indians dataset (UCI).
> In Phase 0, the project pivoted to CDC BRFSS for superior scale, recency, and demographic diversity.
> All sections reflect the actual project state as of Phase 4 completion.

---

## Current Status

| Phase | Status | Output |
|-------|--------|--------|
| Phase 0 — Data Collection | ✅ Complete | `brfss_2022_2024_combined.csv` (1,336,125 × 23) |
| Phase 1 — Data Understanding | ✅ Complete | 11 EDA figures |
| Phase 2 — Data Cleaning | ✅ Complete | `brfss_cleaned.csv` (1,252,580 × 17) |
| Phase 3 — Feature Engineering | ✅ Complete | 6 split files + `scaler.pkl` (14 features) |
| Phase 4 — Modeling | ✅ Complete | 6 trained models + 3 figures |
| Phase 5 — Evaluation | ⏳ Pending | — |

---

## 1. Objective

Develop an interpretable machine learning model to predict diabetes risk
and identify the most influential behavioural and demographic risk factors,
using large-scale, nationally representative survey data.

---

## 2. Problem Statement

Diabetes is a widespread chronic disease, and early detection is crucial for prevention and treatment.
This project aims to:

- Build predictive models for diabetes risk from real-world survey data
- Compare multiple machine learning algorithms on a large, imbalanced dataset
- Provide interpretable insights into key behavioural and demographic risk factors
- Demonstrate an end-to-end ML pipeline with real-world data quality challenges

---

## 3. Target Users

- Healthcare analysts
- Data science recruiters
- ML engineers

---

## 4. Key Questions

- Which features are most predictive of diabetes diagnosis?
- Which model performs best on a large imbalanced dataset?
- Can we balance predictive performance and interpretability?
- What are the limitations of a self-reported, behavioural data model vs. a clinical biomarker model?

---

## 5. Data Source

### Selected: CDC Behavioral Risk Factor Surveillance System (BRFSS)

| Item | Detail |
|------|--------|
| URL | https://www.cdc.gov/brfss/annual_data/annual_data.htm |
| Years | 2022, 2023, 2024 |
| Raw rows | 1,336,125 |
| Cleaned rows | **1,252,580** |
| Features after Phase 3 | **14** |
| Target | `DIABETES` — binary (1=Yes, 0=No) |
| Class split | 14.4% positive / 85.6% negative |

### Why BRFSS Over Pima Indians (original plan)?

| | Pima Indians (UCI) | BRFSS (selected) |
|---|---|---|
| Sample size | 768 | **1,252,580** |
| Demographics | Pima Indian women only | Nationally representative, all adults |
| Variables | 7 clinical biomarkers | **14** behavioural + demographic |
| Recency | 1990s | **2022–2024** |
| Messiness | Pre-cleaned | Real-world: special codes, structural absences, multi-year |
| Generalisability | Very low | High (US adult population) |

---

## 6. Key Variables

### Target

- `DIABETES` — binary (1 = Diabetes, 0 = No Diabetes)
- Derived from BRFSS `DIABETE4`: 1→1 (diabetes), 3→0 (no diabetes)
- Excluded: code 2 (gestational), code 4 (pre-diabetes), codes 7/9 (don't know/refused)

### Final Feature Set After Phase 3 (14 features)

| Variable | Type | Description | Phase 1 rank |
|----------|------|-------------|-------------|
| `GENHLTH` | Ordinal (1–5) | Self-rated general health (1=Excellent…5=Poor) | #1 (r=0.27) |
| `_AGEG5YR` | Ordinal (1–14) | Age group in 5-year intervals | #2 (r=0.23) |
| `DIFFWALK` | Binary (0/1) | Difficulty walking or climbing stairs | #3 (r=0.22) |
| `_BMI5CAT` | Ordinal (1–4) | BMI category (1=Underweight…4=Obese) | #4 (r=0.19) |
| `PHYSHLTH` | Continuous (scaled) | Days physical health not good (0–30) | #5 (r=0.17) |
| `CVDINFR4` | Binary (0/1) | Heart attack history | #7 (r=0.15) |
| `INCOME3` | Ordinal (1–11) | Annual household income bracket | #8 (r=0.15) |
| `EXERANY2` | Binary (0/1) | Physical activity in past 30 days | #9 (r=0.15) |
| `MENTHLTH` | Continuous (scaled) | Days mental health not good (0–30) | — |
| `CHECKUP1` | Ordinal (1–4, 8) | Time since last routine checkup | — |
| `CVDSTRK3` | Binary (0/1) | Stroke history | — |
| `EDUCA` | Ordinal (1–6) | Highest education level | — |
| `_SEX` | Binary (0/1) | Sex (1=Male, 0=Female) | — |
| `_SMOKER3` | Ordinal (1–4) | Smoking status (1=Current daily…4=Never) | — |

### Variables Dropped and Why

| Variable | Dropped in | Reason |
|----------|-----------|--------|
| `BPHIGH6` | Phase 2 | Present in 2023 only — 67.6% missing |
| `_CHOLCH3` | Phase 2 | Present in 2023 only — 67.6% missing |
| `PREDIAB2` | Phase 2 | 65.2% missing + direct target leakage |
| `SEXVAR` | Phase 2 | Exact duplicate of `_SEX` (r=1.00) |
| `_STATE` | Phase 2 | 50-level FIPS code — no ordinal meaning |
| `_RACE` | Phase 2 | Structurally absent in 2022; imputing demographic identity is unsafe |
| `POORHLTH` | Phase 3 | Multicollinearity with `PHYSHLTH` (r=0.70); lower Phase 1 rank (#6 vs #5) |
| `YEAR` | Phase 3 | Temporal leakage risk; prevalence stable (<1pp shift across years) |

---

## 7. Methodology

### Phase 0 — Data Collection ✅
- Downloaded CDC BRFSS ASCII files and HTML codebooks for 2022, 2023, 2024
- Built automated codebook parser (HTML → column positions)
- Extracted 22 target variables; combined into 1,336,125-row dataset

### Phase 1 — Data Understanding ✅
- EDA: target distribution, missing value profiling, feature distributions, correlations
- Identified special codes and structural absences
- Ranked top predictors; documented cleaning plan for Phase 2

### Phase 2 — Data Cleaning ✅
- Recoded BRFSS special codes (88→0 for days-count vars; 7/9/77/99→NaN for all others)
- Binarised target; dropped 6 variables; imputed 8 variables
- Output: `brfss_cleaned.csv` — 1,252,580 rows × 17 columns, 0 NaN, 82.4 MB

### Phase 3 — Feature Engineering ✅
- Resolved `PHYSHLTH` / `POORHLTH` multicollinearity — dropped `POORHLTH` (VIF tie → Phase 1 rank)
- Dropped `YEAR` — temporal leakage; <1pp prevalence shift across years
- Recoded BRFSS binary vars (1/2 → 0/1); cast ordinal float64 → int
- StandardScaler on continuous vars (fit on train only); scaler saved to `models/scaler.pkl`
- 80/20 stratified split (random_state=42)
- Class imbalance: SMOTE on train → 50/50; `class_weight='balanced'` as model-level alternative

### Phase 4 — Modeling ✅
- Trained 6 model variants across 3 algorithms × 2 imbalance strategies
- All models evaluated on the same held-out test set (never resampled)
- Best model: **XGB-Balanced** — ROC-AUC 0.8148, Recall 0.74, Sensitivity 0.80
- All 6 models saved to `models/saved_models/`

### Phase 5 — Evaluation & Interpretation ⏳
- SHAP values for global and individual explanations (using XGB-Balanced)
- Feature importance ranking and beeswarm plot
- Model comparison discussion and limitations

---

## 8. Pipeline

```
CDC BRFSS ASC + HTML Codebooks (2022, 2023, 2024)
↓
00_data_collection.ipynb     →  data/raw/brfss_2022_2024_combined.csv   (1,336,125 × 23)  ✅
↓
01_data_understanding.ipynb  →  outputs/figures/ (11 EDA figures)                          ✅
↓
02_cleaning.ipynb            →  data/processed/brfss_cleaned.csv         (1,252,580 × 17)  ✅
↓
03_feature_engineering.ipynb →  data/processed/ (6 split files)          (14 features)     ✅
↓
04_modeling.ipynb            →  models/saved_models/ (6 models)          ROC-AUC 0.8148    ✅
↓
05_evaluation.ipynb          →  outputs/reports/, outputs/figures/                         ⏳
```

---

## 9. Outputs

### Analytical Outputs
- Model performance comparison table (Accuracy, Precision, Recall, F1, ROC-AUC) ✅ Phase 4
- Feature importance ranking (all three models) ⏳ Phase 5
- SHAP summary and waterfall plots ⏳ Phase 5

### Visual Outputs
- ROC curves (all models overlaid) ✅ Phase 4 — `04_roc_curves.png`
- Confusion matrices (per model, normalised) ✅ Phase 4 — `04_confusion_matrices.png`
- Model comparison bar chart (Recall & ROC-AUC) ✅ Phase 4 — `04_model_comparison.png`
- SHAP beeswarm / bar plots ⏳ Phase 5
- Missing value before/after imputation chart ✅ Phase 2
- Feature correlation heatmap ✅ Phase 3
- Class distribution before/after SMOTE ✅ Phase 3

### Data Outputs

| File | Phase | Shape | Size |
|------|-------|-------|------|
| `brfss_2022_2024_combined.csv` | 0 | 1,336,125 × 23 | 100.4 MB |
| `brfss_cleaned.csv` | 2 | 1,252,580 × 17 | 82.4 MB |
| `X_train.csv` | 3 | 1,002,064 × 14 | 62.2 MB |
| `X_test.csv` | 3 | 250,516 × 14 | 15.6 MB |
| `X_train_smote.csv` | 3 | 1,714,844 × 14 | 106.4 MB |
| `y_train.csv` | 3 | (1,002,064,) | 2.9 MB |
| `y_test.csv` | 3 | (250,516,) | 0.7 MB |
| `y_train_smote.csv` | 3 | (1,714,844,) | 4.9 MB |
| `models/scaler.pkl` | 3 | — | < 1 KB |
| `models/saved_models/lr_balanced.pkl` | 4 | — | ~1.3 KB |
| `models/saved_models/lr_smote.pkl` | 4 | — | ~1.3 KB |
| `models/saved_models/rf_balanced.pkl` | 4 | — | ~2.1 GB |
| `models/saved_models/rf_smote.pkl` | 4 | — | ~3.0 GB |
| `models/saved_models/xgb_balanced.pkl` | 4 | — | ~468 KB |
| `models/saved_models/xgb_smote.pkl` | 4 | — | ~432 KB |

> **Note**: RF `.pkl` files (2–3 GB) are excluded from Git via `.gitignore`.
> XGBoost and LR models are small enough to track but are also excluded
> to keep the repository lightweight. Reproduce via `04_modeling.ipynb`.

---

## 10. Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Core language |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.26 | Numerical operations |
| matplotlib | ≥ 3.7 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisation |
| scikit-learn | 1.4.2 | ML models, preprocessing, metrics |
| imbalanced-learn | 0.14.1 | SMOTE oversampling |
| statsmodels | ≥ 0.14 | VIF calculation |
| scipy | ≥ 1.11 | Chi-square test |
| xgboost | ≥ 2.0 | Gradient boosting model |
| shap | ≥ 0.44 | Model interpretability |
| joblib | ≥ 1.3 | Model and scaler serialisation |

---

## 11. Success Criteria

- Clean, reproducible pipeline from raw CDC ASC files to trained models
- Clear comparison of at least 3 models with ROC-AUC as primary metric ✅
- Interpretable results via SHAP values with actionable insights ⏳ Phase 5
- Proper handling of class imbalance and real-world data quality issues ✅
- All decisions documented in `ProjectDriven.md` ✅

---

## 12. Risks & Limitations

| Risk | Status | Mitigation |
|------|--------|-----------|
| No clinical biomarkers (glucose, HbA1c) | Accepted | Positioned as population-level behavioural screening tool |
| Self-report bias | Accepted | Documented as limitation |
| Class imbalance (14.4% positive) | ✅ Phase 3/4 | SMOTE + `scale_pos_weight` — balanced strategy outperformed SMOTE |
| `PHYSHLTH` / `POORHLTH` multicollinearity (r=0.70) | ✅ Phase 3 | Dropped `POORHLTH` (VIF tie → Phase 1 rank) |
| `_RACE` dropped — limited fairness analysis | Accepted | Documented in limitations |
| Large file (82 MB cleaned) | Managed | Full data used throughout |
| `YEAR` feature — temporal leakage risk | ✅ Phase 3 | Dropped — prevalence shift <1pp |
| RF model files too large for Git (2–3 GB) | ✅ Phase 4 | Excluded via `.gitignore`; reproducible via notebook |
| US population only | Accepted | Documented as limitation |

---

## 13. Repository Structure

```
diabetes-risk-prediction/
│
├── README.md
├── requirements.txt
├── ProjectDriven.md              # Living project log
├── .gitignore
│
├── data/
│   ├── source/                   # CDC ASC + HTML codebooks (not tracked by Git)
│   ├── raw/                      # Extracted CSVs (not tracked by Git)
│   └── processed/                # Cleaned and feature-engineered data (not tracked by Git)
│       ├── brfss_cleaned.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── X_train_smote.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── y_train_smote.csv
│
├── docs/
│   ├── ProjectScope.md
│   ├── methodology.md            # To be completed in Phase 5
│   └── findings.md               # To be completed in Phase 5
│
├── models/
│   ├── scaler.pkl                # Fitted StandardScaler (Phase 3) — tracked by Git
│   └── saved_models/             # Trained models (Phase 4) — excluded from Git
│       ├── lr_balanced.pkl
│       ├── lr_smote.pkl
│       ├── rf_balanced.pkl       # ~2.1 GB — not tracked
│       ├── rf_smote.pkl          # ~3.0 GB — not tracked
│       ├── xgb_balanced.pkl      # ~468 KB — not tracked (reproducible)
│       └── xgb_smote.pkl         # ~432 KB — not tracked (reproducible)
│
├── notebooks/
│   ├── 00_data_collection.ipynb      ✅ Done
│   ├── 01_data_understanding.ipynb   ✅ Done
│   ├── 02_cleaning.ipynb             ✅ Done
│   ├── 03_feature_engineering.ipynb  ✅ Done
│   ├── 04_modeling.ipynb             ✅ Done
│   └── 05_evaluation.ipynb           ⏳ Pending
│
└── outputs/
    ├── figures/                  # 16 figures (Phases 1–4)
    └── reports/
```

---

## 14. Phase-by-Phase Decision Log

### Phase 4 Decisions

#### Model Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **XGB-Balanced** | 0.6985 | 0.6246 | **0.7395** | 0.6138 | **0.8148** |
| LR-Balanced | 0.7171 | 0.6247 | 0.7313 | 0.6227 | 0.8040 |
| LR-SMOTE | 0.7054 | 0.6206 | 0.7273 | 0.6140 | 0.7989 |
| XGB-SMOTE | 0.7536 | 0.6198 | 0.6952 | 0.6309 | 0.7870 |
| RF-SMOTE | 0.7579 | 0.6009 | 0.6495 | 0.6111 | 0.7370 |
| RF-Balanced | 0.7790 | 0.5942 | 0.6169 | 0.6021 | 0.7260 |

#### Winner: XGB-Balanced

- ROC-AUC: **0.8148** (highest across all 6 variants)
- Recall (macro): 0.7395 | Sensitivity (class 1): **0.7971**
- Specificity (class 0): 0.6819
- Raw counts: TP=28,824 | FN=7,337 | FP=68,196 | TN=146,159
- `scale_pos_weight = 857,422 / 144,642 = 5.9294`

#### Imbalance Strategy Finding

`class_weight='balanced'` / `scale_pos_weight` **outperformed SMOTE** across all three
algorithm families. SMOTE improved Recall slightly but reduced ROC-AUC, suggesting
synthetic minority samples do not generalise as well as loss-function reweighting on real data.

#### Random Forest Performance

RF underperformed both LR and XGBoost on ROC-AUC (0.726–0.737 vs 0.799–0.815).
With unlimited `max_depth` and 14 ordinal/binary features, the 100-tree forest
likely memorised the training data rather than learning generalisable patterns.
RF model files are also extremely large (2.1–3.0 GB) vs XGBoost (~450 KB).

#### Phase 5 Input

- Model: `xgb_balanced.pkl`
- Features: 14 (all columns of `X_train`)
- Scaler: `models/scaler.pkl` (required for any new inference)

---

### Phase 3 Decisions

#### Multicollinearity — VIF Analysis

| Variable | VIF | Phase 1 Rank | Decision |
|----------|-----|--------------|----------|
| `PHYSHLTH` | 2.18 | #5 | **RETAINED** |
| `POORHLTH` | 2.05 | #6 | **DROPPED** |

- VIF delta = 0.13 — not a meaningful tie-breaker
- Tie broken by Phase 1 predictor rank
- High VIFs on binary/ordinal vars (CVDSTRK3=77, CVDINFR4=66) are artefacts of applying VIF to non-continuous inputs — no action taken

#### YEAR Column

- Prevalence by year: 14.20% → 14.25% → 14.84% (range = 0.64pp)
- Chi-square p = 6.82e-20: significant at n=1.25M, but trivially small effect size
- **Dropped** — temporal leakage risk

#### Binary Recoding (1/2 → 0/1)

| Variable | Before | After |
|----------|--------|-------|
| `DIFFWALK` | {1.0: 187,669, 2.0: 1,064,911} | {0: 1,064,911, 1: 187,669} |
| `EXERANY2` | {1.0: 958,264, 2.0: 294,316} | {0: 294,316, 1: 958,264} |
| `CVDINFR4` | {1.0: 70,106, 2.0: 1,182,474} | {0: 1,182,474, 1: 70,106} |
| `CVDSTRK3` | {1.0: 53,555, 2.0: 1,199,025} | {0: 1,199,025, 1: 53,555} |
| `_SEX` | {1.0: 594,489, 2.0: 658,091} | {0: 658,091, 1: 594,489} |

#### Split & Scaling

- 80/20 stratified split (random_state=42)
- 14.4% class ratio preserved in both train and test
- StandardScaler: PHYSHLTH mean=−0.0000, std=1.0000 | MENTHLTH mean=0.0000, std=1.0000

#### SMOTE

- Before: 857,422 (85.6%) / 144,642 (14.4%) — ratio 5.9:1
- After SMOTE: 857,422 (50.0%) / 857,422 (50.0%) — perfectly balanced

#### Environment Issue Resolved

- Base Anaconda environment had DLL conflicts after `imbalanced-learn` upgrade (`_MissingValues` import error)
- Resolution: dedicated `diabetes-ml` conda environment (Python 3.11)
- scikit-learn: 1.4.2 | imbalanced-learn: 0.14.1 | numpy: ≥1.26
- `n_jobs` parameter removed from SMOTE call (dropped in imbalanced-learn ≥ 0.12)

### Phase 2 Decisions

#### Revised Imputation Scope

Phase 1 EDA reported missing rates before recoding special codes (7/9/77/99).
After recoding, effective missing rates were substantially higher for several variables:

| Variable | Phase 1 raw missing | Post-recode missing | Action revised |
|----------|--------------------|--------------------|----------------|
| `_AGEG5YR` | 0.0% | 16.9% | Median impute (was: no action needed) |
| `_SMOKER3` | 0.0% | 6.8% | Median impute (was: no action needed) |
| `PHYSHLTH` | ~0% | 4.5% | Median impute (was: row-drop at <3%) |
| `MENTHLTH` | ~0% | 3.7% | Median impute (was: row-drop at <3%) |

Applying row-drop to these variables would have removed ~390k rows (30%) — unacceptable data loss.

#### Imputation Summary

| Variable | Method | Reason |
|----------|--------|--------|
| `INCOME3` | Median | Ordinal — median preserves rank |
| `POORHLTH` | Median | Continuous — median robust to skew |
| `_AGEG5YR` | Median | 16.9% missing post-recode |
| `_BMI5CAT` | Median | Ordinal |
| `_SMOKER3` | Median | Ordinal |
| `PHYSHLTH` | Median | 4.5% missing post-recode |
| `MENTHLTH` | Median | 3.7% missing post-recode |
| `DIFFWALK` | Mode | Binary — only two values |

---

## 15. Inputs for Phase 5

| File | Use |
|------|-----|
| `models/saved_models/xgb_balanced.pkl` | SHAP analysis — best performing model |
| `X_train.csv` + `X_test.csv` | SHAP background dataset and test samples |
| `y_test.csv` | Ground truth for evaluation plots |
| `models/scaler.pkl` | Required if running inference on new data |

### Phase 5 Planned Outputs

- SHAP beeswarm plot (global feature importance)
- SHAP waterfall plot (individual prediction explanation)
- SHAP bar chart (mean absolute SHAP values)
- Final model comparison narrative
- `docs/methodology.md` and `docs/findings.md`
