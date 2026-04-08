# Predicting Diabetes Risk Using Machine Learning

An interpretable machine learning project to predict diabetes risk using
real-world survey data from the CDC Behavioral Risk Factor Surveillance System (BRFSS).

---

## Project Structure

```
diabetes-risk-prediction/
│
├── README.md
├── requirements.txt
├── ProjectDriven.md              # Living project log — updated each phase
├── .gitignore
│
├── data/
│   ├── source/                   # CDC ASC + HTML codebook files (not tracked by Git)
│   ├── raw/                      # Extracted year CSVs + combined (not tracked by Git)
│   │   ├── brfss_2022_diabetes.csv
│   │   ├── brfss_2023_diabetes.csv
│   │   ├── brfss_2024_diabetes.csv
│   │   └── brfss_2022_2024_combined.csv
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
│   ├── ProjectScope.md           # Project specification
│   ├── methodology.md            # To be completed in Phase 5
│   └── findings.md               # To be completed in Phase 5
│
├── models/
│   ├── scaler.pkl                # Fitted StandardScaler (Phase 3) — tracked by Git
│   └── saved_models/             # Trained models (Phase 4) — excluded from Git
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
    ├── figures/                  # 16 figures generated (Phases 1–4)
    └── reports/
```

---

## 1. Background

Diabetes is a widespread chronic disease affecting hundreds of millions of people
worldwide. Early identification of high-risk individuals is critical for timely
intervention and prevention. This project builds and compares multiple machine
learning models using large-scale, nationally representative survey data to support that goal.

---

## 2. Dataset

- **Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- **Years**: 2022, 2023, 2024
- **Raw samples**: 1,336,125 survey respondents
- **Cleaned samples**: 1,252,580
- **Target**: `DIABETES` — binary (1 = Diabetes, 0 = No Diabetes)
- **Class split**: 14.4% positive / 85.6% negative

### Why BRFSS Over the Pima Indians Benchmark?

| | Pima Indians (UCI benchmark) | BRFSS (this project) |
|---|---|---|
| Sample size | 768 | **1,252,580** |
| Demographics | Pima Indian women only | Nationally representative, all adults |
| Feature count | 7 | **14** (after Phase 3) |
| Feature type | Clinical biomarkers only | Behavioural + demographic + comorbidity |
| Recency | 1990s | **2022–2024** |
| Real-world messiness | Pre-cleaned | Special codes, structural absences, multi-year |

### Final Feature Set After Phase 3 (14 features + 1 target)

| Variable | Type | Description | Predictive rank |
|----------|------|-------------|----------------|
| `GENHLTH` | Ordinal (1–5) | Self-rated general health (1=Excellent…5=Poor) | #1 (r=0.27) |
| `_AGEG5YR` | Ordinal (1–14) | Age group in 5-year intervals | #2 (r=0.23) |
| `DIFFWALK` | Binary (0/1) | Difficulty walking or climbing stairs | #3 (r=0.22) |
| `_BMI5CAT` | Ordinal (1–4) | BMI category (1=Underweight…4=Obese) | #4 (r=0.19) |
| `PHYSHLTH` | Continuous (scaled) | Days physical health not good (past 30 days) | #5 (r=0.17) |
| `CVDINFR4` | Binary (0/1) | Heart attack history | #7 (r=0.15) |
| `INCOME3` | Ordinal (1–11) | Annual household income bracket | #8 (r=0.15) |
| `EXERANY2` | Binary (0/1) | Physical activity in past 30 days | #9 (r=0.15) |
| `MENTHLTH` | Continuous (scaled) | Days mental health not good (past 30 days) | — |
| `CHECKUP1` | Ordinal (1–4, 8) | Time since last routine checkup | — |
| `CVDSTRK3` | Binary (0/1) | Stroke history | — |
| `EDUCA` | Ordinal (1–6) | Highest education level | — |
| `_SEX` | Binary (0/1) | Sex (1=Male, 0=Female) | — |
| `_SMOKER3` | Ordinal (1–4) | Smoking status (1=Current daily…4=Never) | — |
| `DIABETES` | **Binary target** | Diabetes diagnosis (1=Yes, 0=No) | — |

> **Variables dropped in Phase 2:**
> `BPHIGH6`, `_CHOLCH3` — present in 2023 only (67.6% missing);
> `PREDIAB2` — 65.2% missing + target leakage;
> `SEXVAR` — exact duplicate of `_SEX` (r=1.00);
> `_STATE` — 50-level FIPS code, no ordinal meaning;
> `_RACE` — structurally absent in 2022, imputing demographic identity is ethically unsafe.
>
> **Variables dropped in Phase 3:**
> `POORHLTH` — multicollinearity with `PHYSHLTH` (r=0.70), lower Phase 1 rank;
> `YEAR` — temporal leakage risk, prevalence shift <1pp across years.

---

## 3. Pipeline

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

## 4. Model Results (Phase 4)

Six model variants were trained across three algorithms and two class imbalance strategies,
all evaluated on the same held-out test set (250,516 respondents, never resampled).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **XGB-Balanced** ⭐ | 0.699 | 0.625 | **0.740** | 0.614 | **0.815** |
| LR-Balanced | 0.717 | 0.625 | 0.731 | 0.623 | 0.804 |
| LR-SMOTE | 0.705 | 0.621 | 0.727 | 0.614 | 0.799 |
| XGB-SMOTE | 0.754 | 0.620 | 0.695 | 0.631 | 0.787 |
| RF-SMOTE | 0.758 | 0.601 | 0.650 | 0.611 | 0.737 |
| RF-Balanced | 0.779 | 0.594 | 0.617 | 0.602 | 0.726 |

**Winner: XGB-Balanced** — highest ROC-AUC and Recall across all six variants.

### Key Finding: Imbalance Strategy

`scale_pos_weight` (loss-function reweighting on real data) consistently outperformed
SMOTE (synthetic oversampling) across all three algorithm families. SMOTE marginally
improves Recall but reduces ROC-AUC, suggesting synthetic minority samples do not
generalise as well as reweighting the loss on real observations.

### Best Model: XGB-Balanced — Confusion Matrix

|  | Predicted: No Diabetes | Predicted: Diabetes |
|--|----------------------|-------------------|
| **Actual: No Diabetes** | 146,159 (TN) | 68,196 (FP) |
| **Actual: Diabetes** | 7,337 (FN) | 28,824 (TP) |

- **Sensitivity** (Recall, class 1): **0.797** — 4 in 5 diabetic respondents correctly flagged
- **Specificity** (Recall, class 0): 0.682

---

## 5. Evaluation Metrics

| Metric | Role |
|--------|------|
| **ROC-AUC** | Primary — robust to class imbalance |
| Precision | Secondary |
| Recall | Secondary — critical for medical screening (minimise false negatives) |
| F1-score | Secondary |
| Confusion Matrix | Per model |

Model interpretability via **SHAP values** (global + individual explanations) — Phase 5.

---

## 6. Feature Engineering Summary (Phase 3)

| Decision | Detail |
|----------|--------|
| Dropped | `POORHLTH` — multicollinearity with `PHYSHLTH` (r=0.70); lower Phase 1 rank |
| Dropped | `YEAR` — temporal leakage; prevalence stable (<1pp shift across years) |
| Binary recoded | `DIFFWALK`, `EXERANY2`, `CVDINFR4`, `CVDSTRK3`, `_SEX` — BRFSS 1/2 → 0/1 |
| Ordinal cast | `GENHLTH`, `_AGEG5YR`, `_BMI5CAT`, `INCOME3`, `CHECKUP1`, `EDUCA`, `_SMOKER3` — float64 → int |
| Scaled | `PHYSHLTH`, `MENTHLTH` — StandardScaler, fit on train only |
| Split | 80/20 stratified, random_state=42 |
| Imbalance | SMOTE on train only → 857,422 per class (50/50); `scale_pos_weight` as model-level alternative |

---

## 7. Limitations

- No clinical biomarkers (glucose, HbA1c) — positioned as behavioural/demographic screening tool
- Self-report bias inherent in survey data
- `_RACE` dropped due to structural absence in 2022 — limited fairness analysis
- US adult population only — generalisability outside the US is limited
- Random Forest models (2–3 GB) are excluded from Git; reproduce via `04_modeling.ipynb`

---

## 8. Tech Stack

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
| jupyter | ≥ 1.0 | Notebook environment |

---

## 9. Reproducibility

All notebooks are self-contained and designed to run in sequence (00 → 05).
Random seeds are set where applicable (`random_state=42`).
All cleaning and modelling decisions are documented in `ProjectDriven.md`.

**Environment:** Use the `diabetes-ml` conda environment (Python 3.11).
See `requirements.txt` for pinned library versions.

To reproduce from scratch:
1. Download BRFSS ASCII files and HTML codebooks for 2022–2024 from https://www.cdc.gov/brfss/annual_data/annual_data.htm
2. Place in `data/source/`
3. Run notebooks 00 → 04 in sequence

> **Note on model files**: Trained models in `models/saved_models/` are excluded from Git
> (Random Forest files are 2–3 GB). Re-run `04_modeling.ipynb` to regenerate all models.
> `models/scaler.pkl` (< 1 KB) is tracked and included.

---

## License

This project is for educational and portfolio purposes only.
BRFSS data is publicly available from the CDC under open data policy.
