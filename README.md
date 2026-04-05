# Predicting Diabetes Risk Using Machine Learning

An interpretable machine learning project to predict diabetes risk using real-world survey data from the CDC Behavioral Risk Factor Surveillance System (BRFSS).

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
│   ├── raw/
│   │   ├── source/               # Original ASC + HTML codebook files (not tracked by git)
│   │   └── prepre/               # Extracted CSV files (not tracked by git)
│   └── processed/                # Cleaned and engineered features
│
├── notebooks/
│   ├── 00_data_collection.ipynb      ✅ Done
│   ├── 01_data_understanding.ipynb   🔄 In Progress
│   ├── 02_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
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

Diabetes is a widespread chronic disease affecting hundreds of millions of people worldwide. Early identification of high-risk individuals is critical for timely intervention and prevention. This project builds and compares multiple machine learning models using large-scale, real-world survey data to support that goal.

---

## 2. Dataset

- **Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- **Years**: 2022, 2023, 2024
- **Total samples**: ~1,300,000 survey respondents
- **Target**: `DIABETE4` — diabetes diagnosis (1 = Yes, 3 = No, 4 = Pre-diabetes)

### Key Variables

| Variable | Description |
|---|---|
| `DIABETE4` | Diabetes diagnosis — **target variable** |
| `_BMI5CAT` | BMI category |
| `_AGEG5YR` | Age group |
| `BPHIGH6` | High blood pressure |
| `_CHOLCH3` | High cholesterol |
| `EXERANY2` | Physical activity |
| `_SMOKER3` | Smoking status |
| `GENHLTH` | General health status |
| `INCOME3` | Household income |
| `EDUCA` | Education level |
| `_RACE` | Race/ethnicity |
| `_SEX` | Sex |

---

## 3. Methodology

### Pipeline

```
Raw ASC Data (CDC BRFSS)
↓
00 — Data Collection      (ASC + codebook → CSV)
↓
01 — Data Understanding   (EDA, distributions, missing values)
↓
02 — Data Cleaning        (handle missing values, encode target)
↓
03 — Feature Engineering  (scaling, encoding, class balancing)
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
- Expected key predictors: BMI, age, blood pressure, general health status, income

---

## 6. Limitations

- Self-reported survey data — subject to recall and response bias
- Cross-sectional design — causality cannot be established
- US population only — limited international generalisability
- Class imbalance (~11% positive rate) — handled via resampling or class weights

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

Download the **ASCII (.ASC)** file and **HTML codebook** for each year (2022, 2023, 2024).  
Place them in `data/raw/source/` and run `00_data_collection.ipynb`.

---

## 8. Tech Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · Jupyter

---

## License

This project is for educational and portfolio purposes only.  
BRFSS data is publicly available from the CDC under open data policy.
