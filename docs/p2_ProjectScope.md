# Project Scope
## Predicting Diabetes Risk Using Machine Learning

> **Version note:** This document was originally drafted for the Pima Indians dataset (UCI).
> In Phase 0, the project pivoted to CDC BRFSS for superior scale, recency, and demographic diversity.
> All sections reflect the actual project state as of Phase 2 completion.

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
| Features | **16** (after Phase 2 cleaning) |
| Target | `DIABETES` — binary (1=Yes, 0=No) |
| Class split | 14.4% positive / 85.6% negative |

### Why BRFSS Over Pima Indians (original plan)?

| | Pima Indians (UCI) | BRFSS (selected) |
|---|---|---|
| Sample size | 768 | **1,252,580** |
| Demographics | Pima Indian women only | Nationally representative, all adults |
| Variables | 7 clinical biomarkers | **16** behavioural + demographic |
| Recency | 1990s | **2022–2024** |
| Messiness | Pre-cleaned | Real-world: special codes, structural absences, multi-year |
| Generalisability | Very low | High (US adult population) |

---

## 6. Key Variables

### Target

- `DIABETES` — binary (1 = Diabetes, 0 = No Diabetes)
- Derived from BRFSS `DIABETE4`: 1→1 (diabetes), 3→0 (no diabetes)
- Excluded: code 2 (gestational), code 4 (pre-diabetes), codes 7/9 (don't know/refused)

### Features Retained After Phase 2 Cleaning (16)

| Variable | Type | Description | Phase 1 rank |
|----------|------|-------------|-------------|
| `GENHLTH` | Ordinal | Self-rated general health (1=Excellent…5=Poor) | #1 (r=0.27) |
| `_AGEG5YR` | Ordinal | Age group in 5-year intervals (1=18–24…13=80+) | #2 (r=0.23) |
| `DIFFWALK` | Binary | Difficulty walking or climbing stairs | #3 (r=0.22) |
| `_BMI5CAT` | Ordinal | BMI category (1=Underweight…4=Obese) | #4 (r=0.19) |
| `PHYSHLTH` | Continuous | Days physical health not good (0–30) | #5 (r=0.17) |
| `POORHLTH` | Continuous | Days poor health limited activities (0–30) | #6 (r=0.17) |
| `CVDINFR4` | Binary | Heart attack history | #7 (r=0.15) |
| `INCOME3` | Ordinal | Annual household income bracket | #8 (r=0.15) |
| `EXERANY2` | Binary | Physical activity in past 30 days | #9 (r=0.15) |
| `MENTHLTH` | Continuous | Days mental health not good (0–30) | — |
| `CHECKUP1` | Ordinal | Time since last routine checkup | — |
| `CVDSTRK3` | Binary | Stroke history | — |
| `EDUCA` | Ordinal | Highest education level | — |
| `_SEX` | Binary | Sex (1=Male, 2=Female) | — |
| `_SMOKER3` | Ordinal | Smoking status (1=Current daily…4=Never) | — |
| `YEAR` | Categorical | Survey year (2022 / 2023 / 2024) | — |

### Variables Dropped and Why

| Variable | Dropped in | Reason |
|----------|-----------|--------|
| `BPHIGH6` | Phase 2 | Present in 2023 only — 67.6% missing |
| `_CHOLCH3` | Phase 2 | Present in 2023 only — 67.6% missing |
| `PREDIAB2` | Phase 2 | 65.2% missing + direct target leakage |
| `SEXVAR` | Phase 2 | Exact duplicate of `_SEX` (r=1.00) |
| `_STATE` | Phase 2 | 50-level FIPS code — no ordinal meaning |
| `_RACE` | Phase 2 | Structurally absent in 2022; imputing demographic identity is unsafe |

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

### Phase 3 — Feature Engineering ⏳
- Evaluate and resolve `PHYSHLTH` / `POORHLTH` multicollinearity (r=0.70)
- Assess `YEAR` column for temporal leakage risk
- Scale continuous features (StandardScaler)
- Address class imbalance: SMOTE or `class_weight='balanced'`
- Cast ordinal float64 variables to int

### Phase 4 — Modeling ⏳
- Logistic Regression (interpretable baseline)
- Random Forest (ensemble; built-in feature importance)
- XGBoost (gradient boosting; best performance on tabular data)

### Phase 5 — Evaluation & Interpretation ⏳
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix per model
- SHAP values for global and individual explanations
- Model comparison and trade-off discussion

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
03_feature_engineering.ipynb →  X_train, X_test, y_train, y_test                          ⏳
↓
04_modeling.ipynb            →  models/saved_models/                                       ⏳
↓
05_evaluation.ipynb          →  outputs/reports/, outputs/figures/                         ⏳
```

---

## 9. Outputs

### Analytical Outputs
- Model performance comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance ranking (all three models)
- SHAP summary and waterfall plots

### Visual Outputs
- ROC curves (all models overlaid)
- Confusion matrices (per model)
- SHAP beeswarm / bar plots
- Missing value before/after imputation chart ✅ (Phase 2)

---

## 10. Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Core language |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| matplotlib | ≥ 3.7 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisation |
| scikit-learn | ≥ 1.3 | ML models, preprocessing, metrics |
| xgboost | ≥ 2.0 | Gradient boosting model |
| shap | ≥ 0.44 | Model interpretability |
| jupyter | ≥ 1.0 | Notebook environment |
| joblib | ≥ 1.3 | Model serialisation |

---

## 11. Success Criteria

- Clean, reproducible pipeline from raw CDC ASC files to trained models
- Clear comparison of at least 3 models with ROC-AUC as primary metric
- Interpretable results via SHAP values with actionable insights
- Proper handling of class imbalance and real-world data quality issues
- All decisions documented in `ProjectDriven.md`

---

## 12. Risks & Limitations

| Risk | Status | Mitigation |
|------|--------|-----------|
| No clinical biomarkers (glucose, HbA1c) | Accepted | Positioned as population-level behavioural screening tool |
| Self-report bias | Accepted | Documented as limitation |
| Class imbalance (14.4% positive) | ⏳ Phase 3 | SMOTE or `class_weight='balanced'` |
| `PHYSHLTH` / `POORHLTH` multicollinearity (r=0.70) | ⏳ Phase 3 | VIF evaluation; drop one or create composite |
| `_RACE` dropped — limited fairness analysis | Accepted | Documented in limitations |
| Large file (82 MB cleaned) | Managed | Sample during development; full data for final model |
| `YEAR` feature — temporal leakage risk | ⏳ Phase 3 | Evaluate inclusion; year-stratified split if retained |
| US population only | Accepted | Documented as limitation |

---

## 13. Repository Structure

```
diabetes-risk-prediction/
│
├── README.md
├── requirements.txt
├── ProjectDriven.md          # Living project log
├── .gitignore
│
├── docs/
│   ├── ProjectScope.md       # This document
│   ├── methodology.md        # To be completed in Phase 5
│   └── findings.md           # To be completed in Phase 5
│
├── data/
│   ├── source/               # CDC ASC + HTML codebooks (not tracked by Git)
│   ├── raw/                  # Extracted CSVs (not tracked by Git)
│   └── processed/            # brfss_cleaned.csv (not tracked by Git)
│
├── notebooks/
│   ├── 00_data_collection.ipynb      ✅
│   ├── 01_data_understanding.ipynb   ✅
│   ├── 02_cleaning.ipynb             ✅
│   ├── 03_feature_engineering.ipynb  ⏳
│   ├── 04_modeling.ipynb             ⏳
│   └── 05_evaluation.ipynb           ⏳
│
├── models/
│   └── saved_models/
│
└── outputs/
    ├── figures/              # 12 figures (Phases 1–2)
    └── reports/
```
