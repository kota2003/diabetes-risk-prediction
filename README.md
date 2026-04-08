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
│   └── processed/                # Cleaned, analysis-ready data (not tracked by Git)
│       └── brfss_cleaned.csv
│
├── docs/
│   ├── ProjectScope.md           # Project specification
│   ├── methodology.md            # To be completed in Phase 5
│   └── findings.md               # To be completed in Phase 5
│
├── notebooks/
│   ├── 00_data_collection.ipynb      ✅ Done
│   ├── 01_data_understanding.ipynb   ✅ Done
│   ├── 02_cleaning.ipynb             ✅ Done
│   ├── 03_feature_engineering.ipynb  ⏳ Pending
│   ├── 04_modeling.ipynb             ⏳ Pending
│   └── 05_evaluation.ipynb           ⏳ Pending
│
├── models/
│   └── saved_models/
│
└── outputs/
    ├── figures/                  # 12 figures generated (Phases 1–2)
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
| Feature count | 7 | **16** |
| Feature type | Clinical biomarkers only | Behavioural + demographic + comorbidity |
| Recency | 1990s | **2022–2024** |
| Real-world messiness | Pre-cleaned | Special codes, structural absences, multi-year |

### Cleaned Feature Set (16 features + 1 target)

| Variable | Type | Description | Predictive rank |
|----------|------|-------------|----------------|
| `GENHLTH` | Ordinal | Self-rated general health (1=Excellent…5=Poor) | #1 (r=0.27) |
| `_AGEG5YR` | Ordinal | Age group in 5-year intervals (1=18–24…13=80+) | #2 (r=0.23) |
| `DIFFWALK` | Binary | Difficulty walking or climbing stairs | #3 (r=0.22) |
| `_BMI5CAT` | Ordinal | BMI category (1=Underweight…4=Obese) | #4 (r=0.19) |
| `PHYSHLTH` | Continuous | Days physical health not good (past 30 days) | #5 (r=0.17) |
| `POORHLTH` | Continuous | Days poor health limited activities (past 30 days) | #6 (r=0.17) |
| `CVDINFR4` | Binary | Heart attack history | #7 (r=0.15) |
| `INCOME3` | Ordinal | Annual household income bracket | #8 (r=0.15) |
| `EXERANY2` | Binary | Physical activity in past 30 days | #9 (r=0.15) |
| `MENTHLTH` | Continuous | Days mental health not good (past 30 days) | — |
| `CHECKUP1` | Ordinal | Time since last routine checkup | — |
| `CVDSTRK3` | Binary | Stroke history | — |
| `EDUCA` | Ordinal | Highest education level | — |
| `_SEX` | Binary | Sex (1=Male, 2=Female) | — |
| `_SMOKER3` | Ordinal | Smoking status (1=Current daily…4=Never) | — |
| `YEAR` | Categorical | Survey year (2022 / 2023 / 2024) | — |
| `DIABETES` | **Binary target** | Diabetes diagnosis (1=Yes, 0=No) | — |

> **Variables dropped in Phase 2:**
> `BPHIGH6`, `_CHOLCH3` — present in 2023 only (67.6% missing)
> `PREDIAB2` — 65.2% missing + target leakage
> `SEXVAR` — exact duplicate of `_SEX` (r=1.00)
> `_STATE` — 50-level FIPS code; no ordinal meaning
> `_RACE` — structurally absent in 2022; imputing demographic identity is ethically unsafe

---

## 3. Methodology

### Pipeline

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

### Models

- **Logistic Regression** — interpretable linear baseline
- **Random Forest** — ensemble; built-in feature importance
- **XGBoost** — gradient boosting; typically best performance on tabular data

### Evaluation Metrics

- Accuracy · Precision · Recall · F1-score · **ROC-AUC** (primary metric for imbalanced classification)

---

## 4. Results

| Model | Accuracy | ROC-AUC | F1-score |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

*Results will be updated upon completion of Phase 5.*

---

## 5. Interpretation

- Feature importance from Random Forest and XGBoost
- SHAP values for global and individual prediction explanations
- Key predictors identified in Phase 1 EDA:
  **General health status, age group, walking difficulty, BMI, physical health days, income**

---

## 6. Limitations

- **Self-reported data** — subject to recall bias and social desirability effects
- **Cross-sectional design** — causal direction cannot be established from survey responses
- **No clinical biomarkers** — glucose, HbA1c, and insulin are not collected by BRFSS;
  this model is a behavioural/demographic screening tool, not a clinical diagnostic instrument
- **US population only** — limited generalisability to other countries or health systems
- **Class imbalance** — 14.4% positive rate; addressed via SMOTE or class weights in Phase 3
- **`_RACE` dropped** — structurally absent in 2022; fairness analysis across demographic groups is limited

---

## 7. Setup

```bash
# Clone the repository
git clone https://github.com/kota2003/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook
```

### Data Access

Raw BRFSS data must be downloaded separately from the CDC:

```
https://www.cdc.gov/brfss/annual_data/annual_data.htm
```

For each year (2022, 2023, 2024), download:
- **ASCII Data File** (.zip containing .ASC)
- **HTML Codebook** (USCODE\*\*.HTML)

Place both files in `data/source/`, then run the notebooks in sequence:

```
00_data_collection.ipynb   → generates data/raw/
02_cleaning.ipynb          → generates data/processed/brfss_cleaned.csv
```

---

## 8. Tech Stack

| Library | Purpose |
|---------|---------|
| Python ≥ 3.10 | Core language |
| pandas ≥ 2.0 | Data loading and manipulation |
| numpy ≥ 1.24 | Numerical operations |
| matplotlib ≥ 3.7 | Plotting |
| seaborn ≥ 0.12 | Statistical visualisation |
| scikit-learn ≥ 1.3 | ML models, preprocessing, metrics |
| xgboost ≥ 2.0 | Gradient boosting model |
| shap ≥ 0.44 | Model interpretability |
| jupyter ≥ 1.0 | Notebook environment |

---

## 9. Reproducibility

All notebooks are self-contained and designed to run in sequence (00 → 05).
Random seeds are set where applicable.
All cleaning and modelling decisions are documented in `ProjectDriven.md`.

---

## License

This project is for educational and portfolio purposes only.
BRFSS data is publicly available from the CDC under open data policy.
