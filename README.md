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
├── .gitignore
│
├── data/
│   ├── source/               # Original ASC + HTML codebook files (not tracked by Git)
│   ├── raw/                  # Extracted CSVs (not tracked by Git)
│   │   ├── brfss_2022_diabetes.csv
│   │   ├── brfss_2023_diabetes.csv
│   │   ├── brfss_2024_diabetes.csv
│   │   └── brfss_2022_2024_combined.csv
│   └── processed/            # Cleaned, analysis-ready data (not tracked by Git)
│       └── brfss_cleaned.csv
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
    ├── figures/
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
- **Cleaned samples**: 1,252,580 (after exclusions and imputation)
- **Target**: `DIABETES` — diabetes diagnosis (binary: 1 = Diabetes, 0 = No Diabetes)
- **Class split**: 14.4% positive (diabetes) / 85.6% negative (no diabetes)
- **Features**: 16 (demographics, health behaviours, chronic conditions)

### Why BRFSS Over the Pima Indians Benchmark?

| | Pima Indians (UCI benchmark) | BRFSS (this project) |
|---|---|---|
| Sample size | 768 | **1,252,580** |
| Demographics | Pima Indian women only | Nationally representative, all adults |
| Feature count | 7 | **16** |
| Feature type | Clinical biomarkers only | Behavioural + demographic + comorbidity |
| Recency | 1990s | **2022–2024** |
| Real-world messiness | Pre-cleaned | Special codes, structural absences, multi-year |

### Cleaned Feature Set

| Variable | Type | Description |
|----------|------|-------------|
| `GENHLTH` | Ordinal | Self-rated general health — **top predictor (r=0.27)** |
| `_AGEG5YR` | Ordinal | Age group in 5-year intervals — **2nd predictor (r=0.23)** |
| `DIFFWALK` | Binary | Difficulty walking or climbing stairs (r=0.22) |
| `_BMI5CAT` | Ordinal | BMI category (r=0.19) |
| `PHYSHLTH` | Continuous | Days physical health not good (past 30 days) |
| `POORHLTH` | Continuous | Days poor health limited activities (past 30 days) |
| `MENTHLTH` | Continuous | Days mental health not good (past 30 days) |
| `CVDINFR4` | Binary | Heart attack history |
| `CVDSTRK3` | Binary | Stroke history |
| `INCOME3` | Ordinal | Household income bracket |
| `EDUCA` | Ordinal | Education level |
| `_SEX` | Binary | Sex |
| `_SMOKER3` | Ordinal | Smoking status |
| `CHECKUP1` | Ordinal | Time since last routine checkup |
| `EXERANY2` | Binary | Physical activity in past 30 days |
| `YEAR` | Categorical | Survey year (2022 / 2023 / 2024) |

> **Note on dropped variables:**  
> `BPHIGH6` and `_CHOLCH3` were extracted but present in 2023 only (67.6% missing) — dropped in Phase 2.  
> `_RACE` was absent in 2022 (structural, not random) and was dropped to avoid demographic imputation bias.  
> `PREDIAB2` was dropped due to target leakage (65.2% missing and directly encodes borderline status).

---

## 3. Methodology

### Pipeline

```
Raw ASC Data (CDC BRFSS 2022–2024)
↓
00 — Data Collection      (ASC + HTML codebook → CSV; 1,336,125 rows × 23 cols)
↓
01 — Data Understanding   (EDA, distributions, missing values, correlations; 11 figures)
↓
02 — Data Cleaning        (recode special codes, binarise target, drop/impute;
                           1,252,580 rows × 17 cols, 0 NaN)
↓
03 — Feature Engineering  (encoding, scaling, class balancing, VIF check)
↓
04 — Modeling             (Logistic Regression, Random Forest, XGBoost)
↓
05 — Evaluation           (metrics, SHAP values, model comparison)
```

### Models

- Logistic Regression (interpretable baseline)
- Random Forest (ensemble; built-in feature importance)
- XGBoost (gradient boosting; state-of-the-art performance)

### Evaluation Metrics

- Accuracy · Precision · Recall · F1-score · ROC-AUC

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
- SHAP values for individual prediction explanations
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
- **`_RACE` dropped** — race/ethnicity was structurally absent in 2022 (one-third of data);
  fairness analysis across demographic groups is limited without this variable

---

## 7. Setup

```bash
# Clone the repository
git clone https://github.com/your-username/diabetes-risk-prediction.git
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

Download the **ASCII (.ASC)** file and **HTML codebook** for each year
(2022, 2023, 2024) and place them in `data/source/`.
Then run `00_data_collection.ipynb` to generate the raw CSVs,
followed by `02_cleaning.ipynb` to produce `data/processed/brfss_cleaned.csv`.

---

## 8. Tech Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · Jupyter

---

## 9. Reproducibility

All notebooks are self-contained and run in sequence.
Random seeds are set where applicable.
All cleaning and modelling decisions are documented in `ProjectDriven.md`.

---

## License

This project is for educational and portfolio purposes only.
BRFSS data is publicly available from the CDC under open data policy.
