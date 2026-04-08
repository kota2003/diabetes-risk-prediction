# Predicting Diabetes Risk Using Machine Learning

An interpretable machine learning project to predict diabetes risk using
real-world survey data from the CDC Behavioral Risk Factor Surveillance System (BRFSS).

---

## Project Structure

```
project_diabetes_ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── source/               # Original ASC + HTML codebook files (not tracked by Git)
│   └── raw/                  # Extracted CSVs (not tracked by Git)
│       ├── brfss_2022_diabetes.csv
│       ├── brfss_2023_diabetes.csv
│       ├── brfss_2024_diabetes.csv
│       └── brfss_2022_2024_combined.csv
│
├── notebooks/
│   ├── 00_data_collection.ipynb      ✅ Done
│   ├── 01_data_understanding.ipynb   ✅ Done
│   ├── 02_cleaning.ipynb             ⏳ Pending
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
learning models using large-scale, real-world survey data to support that goal.

---

## 2. Dataset

- **Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- **Years**: 2022, 2023, 2024
- **Total samples**: ~1,336,000 survey respondents
- **Target**: `DIABETE4` — diabetes diagnosis (binary: 1 = diabetes, 0 = no diabetes)
- **Class split**: ~14.5% positive (diabetes) / ~85.5% negative

### Why BRFSS?

| | Pima Indians (common benchmark) | BRFSS (this project) |
|---|---|---|
| Sample size | 768 | ~1,336,000 |
| Demographics | Pima Indian women only | Nationally representative |
| Variables | 8 | 22 |
| Recency | 1990s | 2022–2024 |

### Key Variables

| Variable | Description |
|---|---|
| `DIABETE4` | Diabetes diagnosis — **target variable** |
| `GENHLTH` | General health status — top predictor (r = 0.27) |
| `_AGEG5YR` | Age group — top predictor (r = 0.23) |
| `DIFFWALK` | Difficulty walking or climbing stairs (r = 0.22) |
| `_BMI5CAT` | BMI category (r = 0.19) |
| `PHYSHLTH` | Days physical health not good (past 30 days) |
| `EXERANY2` | Physical activity in past 30 days |
| `CVDINFR4` | Heart attack history |
| `INCOME3` | Household income |
| `EDUCA` | Education level |
| `_RACE` | Race/ethnicity (calculated) |
| `_SEX` | Sex (calculated) |

> **Note**: `BPHIGH6` (high blood pressure) and `_CHOLCH3` (cholesterol check) were extracted
> but are present in 2023 only — they will be dropped in Phase 2 data cleaning.

---

## 3. Methodology

### Pipeline

```
Raw ASC Data (CDC BRFSS)
↓
00 — Data Collection      (ASC + codebook → CSV)
↓
01 — Data Understanding   (EDA, distributions, missing values, correlations)
↓
02 — Data Cleaning        (recode special codes, binarise target, drop/impute)
↓
03 — Feature Engineering  (encoding, scaling, class balancing)
↓
04 — Modeling             (Logistic Regression, Random Forest, XGBoost)
↓
05 — Evaluation           (metrics, SHAP, model comparison)
```

### Models

- Logistic Regression
- Random Forest
- XGBoost

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
  **General health status**, **age**, **walking difficulty**, **BMI**, **physical health days**, **income**

---

## 6. Limitations

- Self-reported survey data — subject to recall and response bias
- Cross-sectional design — causality cannot be established
- US population only — limited international generalisability
- Class imbalance (~14.5% positive rate) — handled via resampling or class weights

---

## 7. Setup

```bash
# Clone the repository
git clone https://github.com/your-username/project_diabetes_ml.git
cd project_diabetes_ml

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
Then run `00_data_collection.ipynb` to generate the analysis-ready CSVs.

---

## 8. Tech Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · Jupyter

---

## License

This project is for educational and portfolio purposes only.
BRFSS data is publicly available from the CDC under open data policy.
