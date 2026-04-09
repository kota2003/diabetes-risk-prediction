# 🩺 Predicting Diabetes Risk Using Machine Learning

> End-to-end machine learning pipeline identifying individuals at high risk of diabetes
> using nationally representative behavioural survey data — built for healthcare analysts,
> data science practitioners, and ML engineers.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?logo=xgboost&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-f7931e?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-8e44ad)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.815-2e7d32)
![Sensitivity](https://img.shields.io/badge/Sensitivity-0.797-1565c0)
![Samples](https://img.shields.io/badge/Samples-1.25M-6a1b9a)
![Years](https://img.shields.io/badge/Years-2022--2024-00838f)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why BRFSS?](#-why-brfss)
3. [Pipeline](#️-pipeline)
4. [Results](#-results)
5. [SHAP Interpretation](#-shap-interpretation)
6. [Feature Set](#-feature-set)
7. [Key Findings](#-key-findings)
8. [Limitations](#️-limitations)
9. [Reproducibility](#-reproducibility)
10. [Tech Stack](#️-tech-stack)
11. [Docs](#-docs)

---

## 📌 Project Overview

Diabetes affects hundreds of millions of people worldwide. Early identification of
high-risk individuals is critical for timely intervention and prevention. This project
builds and compares **six machine learning models** — Logistic Regression, Random Forest,
and XGBoost, each under two class-imbalance strategies — using survey-based behavioural
and demographic features to predict diabetes diagnosis.

> **Goal:** Demonstrate a production-quality end-to-end ML pipeline on a real, messy,
> large-scale dataset — with interpretable, clinically grounded explanations via SHAP.

### Target Users
- 🏥 **Healthcare analysts** — population-level risk identification
- 📊 **Data science recruiters** — end-to-end pipeline with documented decision rationale
- ⚙️ **ML engineers** — reproducible pipeline with class imbalance strategy comparison

### Dataset at a Glance

| | |
|---|---|
| **Source** | CDC Behavioral Risk Factor Surveillance System (BRFSS) |
| **Years** | 2022, 2023, 2024 |
| **Raw samples** | 1,336,125 |
| **Cleaned samples** | 1,252,580 |
| **Features** | 14 (behavioural + demographic) |
| **Target** | `DIABETES` — binary (1 = Diabetes, 0 = No Diabetes) |
| **Class split** | 14.4% positive / 85.6% negative |
| **Best model ROC-AUC** | **0.815** |

---

## 🌏 Why BRFSS?

Most public diabetes ML projects use the Pima Indians dataset (UCI, 768 rows, 1990s).
This project deliberately uses a harder, richer alternative:

| | Pima Indians (UCI) | CDC BRFSS (this project) |
|---|---|---|
| Sample size | 768 | **1,252,580** |
| Demographics | Pima Indian women only | Nationally representative US adults |
| Features | 7 clinical biomarkers | **14** behavioural + demographic |
| Recency | 1990s | **2022–2024** |
| Data quality | Pre-cleaned | Special codes, structural absences, multi-year alignment |
| Generalisability | Very low | High |

Using BRFSS introduces real data engineering challenges: fixed-width ASCII parsing,
automated HTML codebook parsing, BRFSS-specific special codes (88 = "none", 7/9/77/99 = refused),
multi-year variable alignment, and structural missingness that varies by survey year.

---

## ⚙️ Pipeline

```
CDC BRFSS ASCII files + HTML codebooks (2022, 2023, 2024)
│
├── 00_data_collection.ipynb
│       Automated codebook parser → fixed-width extraction → 1,336,125 × 23 rows
│
├── 01_data_understanding.ipynb
│       EDA: distributions, correlations, missing profiles, special code audit → 11 figures
│
├── 02_cleaning.ipynb
│       Recode specials, drop 6 vars, impute 8 vars → brfss_cleaned.csv (1,252,580 × 17)
│
├── 03_feature_engineering.ipynb
│       VIF analysis, binary recoding, scaling, 80/20 split, SMOTE → 14 features, 6 data files
│
├── 04_modeling.ipynb
│       6 model variants (LR / RF / XGB × Balanced / SMOTE) → XGB-Balanced ROC-AUC 0.815
│
└── 05_evaluation.ipynb
        SHAP global + individual explanations → 5 figures + methodology.md + findings.md
```

All notebooks run in sequence (00 → 05). Every cleaning and modelling decision is
logged with rationale in [`ProjectDriven.md`](ProjectDriven.md).

---

## 🤖 Results

Six model variants were trained and evaluated on the same held-out test set
(250,516 respondents, 14.4% positive — **never resampled**).
Primary metric: **ROC-AUC** (robust to class imbalance).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| **XGB-Balanced** ⭐ | 0.699 | 0.625 | **0.740** | 0.614 | **0.815** |
| LR-Balanced | 0.717 | 0.625 | 0.731 | 0.623 | 0.804 |
| LR-SMOTE | 0.705 | 0.621 | 0.727 | 0.614 | 0.799 |
| XGB-SMOTE | 0.754 | 0.620 | 0.695 | 0.631 | 0.787 |
| RF-SMOTE | 0.758 | 0.601 | 0.650 | 0.611 | 0.737 |
| RF-Balanced | 0.779 | 0.594 | 0.617 | 0.602 | 0.726 |

### Best Model: XGB-Balanced — Confusion Matrix (test set, n=250,516)

|  | Predicted: No Diabetes | Predicted: Diabetes |
|--|:----------------------:|:-------------------:|
| **Actual: No Diabetes** | 146,159 (TN) | 68,196 (FP) |
| **Actual: Diabetes** | 7,337 (FN) | 28,824 (TP) |

- **Sensitivity** (True Positive Rate): **0.797** — 4 in 5 diabetic respondents correctly identified
- **Specificity** (True Negative Rate): 0.682

### Imbalance Strategy: Balanced vs SMOTE

![Model Comparison](https://raw.githubusercontent.com/kota2003/diabetes-risk-prediction/main/outputs/figures/04_model_comparison.png)

`scale_pos_weight` (loss-function reweighting) **outperformed SMOTE** across all three
algorithm families on ROC-AUC. At 1.25M scale, synthetic minority samples do not add
generalisable signal beyond what reweighting achieves on real data.

---

## 🔍 SHAP Interpretation

[SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/) was used to explain
the XGB-Balanced model. `shap.TreeExplainer` computes **exact** Shapley values by exploiting
XGBoost's tree structure — no approximations required.

**Explanation set:** 5,000 randomly sampled held-out test respondents (`random_state=42`).

### Global Feature Importance — Mean |SHAP Value|

![SHAP Bar Chart](https://raw.githubusercontent.com/kota2003/diabetes-risk-prediction/main/outputs/figures/05_shap_bar.png)

*Age and self-rated health dominate. The top 4 features account for the majority of
predictive signal — and are identical across both SHAP and XGBoost built-in rankings.*

### Direction & Spread — Beeswarm Plot

![SHAP Beeswarm](https://raw.githubusercontent.com/kota2003/diabetes-risk-prediction/main/outputs/figures/05_shap_beeswarm.png)

*Each dot is one respondent. Colour = feature value (red = high, blue = low).
Older age (high `_AGEG5YR`, red) and poor health (high `GENHLTH`, red) push strongly
right — toward the positive (diabetes) class.*

### Individual Predictions — Waterfall Plots

| True Positive (p = 0.944) | True Negative (p = 0.0003) |
|:---:|:---:|
| ![Waterfall TP](https://raw.githubusercontent.com/kota2003/diabetes-risk-prediction/main/outputs/figures/05_shap_waterfall_tp.png) | ![Waterfall TN](https://raw.githubusercontent.com/kota2003/diabetes-risk-prediction/main/outputs/figures/05_shap_waterfall_tn.png) |

**True Positive profile:** GENHLTH=5 (poor health), age group 10 (older adult), heart attack
history, obese BMI, stroke history — every major risk factor stacked simultaneously.

**True Negative profile:** Age group 1 (youngest), underweight BMI, good health, recent
checkup — the model pushes strongly negative with near-certainty.

### Built-in Gain vs SHAP Comparison

![Importance Comparison](https://raw.githubusercontent.com/kota2003/diabetes-risk-prediction/main/outputs/figures/05_importance_comparison.png)

*Top-4 features are **identical** across both methods — robust signal.
`CVDINFR4` (heart attack history) is inflated by XGBoost gain (rank 5) vs SHAP (rank 10),
because gain measures split frequency, not prediction impact.*

---

## 📊 Feature Set

14 features retained after Phase 3 feature engineering (SHAP-ranked):

| SHAP Rank | Variable | Type | Description |
|:---------:|----------|------|-------------|
| **#1** | `_AGEG5YR` | Ordinal (1–14) | Age group in 5-year intervals |
| **#2** | `GENHLTH` | Ordinal (1–5) | Self-rated general health (1=Excellent, 5=Poor) |
| **#3** | `CHECKUP1` | Ordinal (1–4) | Time since last routine checkup |
| **#4** | `_BMI5CAT` | Ordinal (1–4) | BMI category (1=Underweight, 4=Obese) |
| #5 | `_SEX` | Binary (0/1) | Sex (1=Male, 0=Female) |
| #6 | `DIFFWALK` | Binary (0/1) | Difficulty walking or climbing stairs |
| #7 | `EXERANY2` | Binary (0/1) | Physical activity in past 30 days |
| #8 | `EDUCA` | Ordinal (1–6) | Highest education level |
| #9 | `INCOME3` | Ordinal (1–11) | Annual household income bracket |
| #10 | `CVDINFR4` | Binary (0/1) | Heart attack history |
| #11 | `MENTHLTH` | Continuous (scaled) | Days mental health not good (0–30) |
| #12 | `_SMOKER3` | Ordinal (1–4) | Smoking status (1=Current daily, 4=Never) |
| #13 | `PHYSHLTH` | Continuous (scaled) | Days physical health not good (0–30) |
| #14 | `CVDSTRK3` | Binary (0/1) | Stroke history |

<details>
<summary>📋 Variables dropped (click to expand)</summary>

| Variable | Phase | Reason |
|----------|-------|--------|
| `BPHIGH6`, `_CHOLCH3` | Phase 2 | Present in 2023 only — 67.6% structurally missing |
| `PREDIAB2` | Phase 2 | 65.2% missing + direct target leakage |
| `SEXVAR` | Phase 2 | Exact duplicate of `_SEX` (r=1.00) |
| `_STATE` | Phase 2 | 50-level FIPS code — no ordinal meaning |
| `_RACE` | Phase 2 | Structurally absent in 2022; imputing demographic identity is unsafe |
| `POORHLTH` | Phase 3 | Multicollinearity with `PHYSHLTH` (r=0.70); lower Phase 1 rank |
| `YEAR` | Phase 3 | Temporal leakage risk; prevalence shift <1pp across years |

</details>

---

## 💡 Key Findings

### Finding 1 — Age Is the Dominant Predictor
**Mean |SHAP| = 0.709** — 29% higher than the second-ranked feature.

Older age groups push predictions strongly toward the positive class. `_AGEG5YR` alone
accounts for more predictive signal than any other variable by a clear margin.

---

### Finding 2 — Self-Rated Health Captures Broad Chronic Disease Burden
**Mean |SHAP| = 0.550**

`GENHLTH` ranks 2nd. Poor self-rated health is a powerful proxy for undiagnosed or poorly
managed chronic conditions — including diabetes that has not yet been formally diagnosed.

---

### Finding 3 — `CHECKUP1` Shows a Counterintuitive Negative Association
**Mean |SHAP| = 0.422 — direction is negative**

Longer time since last routine checkup is associated with *lower* predicted diabetes risk.
This is **not** a causal finding — it likely reflects survivorship bias (healthier individuals
attend checkups less) or healthcare access confounding (uninsured individuals are both less
likely to attend checkups *and* less likely to have a diabetes diagnosis on record).

---

### Finding 4 — Loss-Function Reweighting Outperforms SMOTE at Scale
`scale_pos_weight` / `class_weight='balanced'` outperformed SMOTE on ROC-AUC across
**all three algorithm families**. At 1.25M rows, synthetic minority oversampling does
not add generalisable signal beyond what reweighting achieves on real data.

---

### Finding 5 — Built-in Gain Inflates `CVDINFR4`
XGBoost ranks heart attack history 5th by gain but SHAP ranks it 10th. Gain measures
split frequency — not prediction impact. The top-4 are identical across both methods,
confirming robust signal at the top of the ranking.

---

## ⚠️ Limitations

| Limitation | Impact |
|---|---|
| No clinical biomarkers (glucose, HbA1c) | Model uses behavioural/demographic proxies only; lower sensitivity than a biomarker-based tool |
| Self-report bias | Underreporting of unhealthy behaviours and undiagnosed diabetes (coded 0) are unmeasurable |
| `_RACE` excluded | Structurally absent in 2022 BRFSS; fairness analysis across racial/ethnic groups is not possible |
| US adult population only | Generalisation to other countries or healthcare systems is not supported |
| Temporal scope 2022–2024 | Pre-pandemic patterns and future trends are not captured |
| `CHECKUP1` confounding | Negative association is a data artefact — not a causal protective effect |

> ✅ **Intended use:** Population-level behavioural screening tool to flag high-risk individuals
> for follow-up clinical assessment — **not a diagnostic replacement.**

---

## 🚀 Reproducibility

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/kota2003/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# Create conda environment
conda create -n diabetes-ml python=3.11
conda activate diabetes-ml

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download BRFSS ASCII files (`.ASC`) and HTML codebooks for 2022, 2023, 2024 from
   [CDC BRFSS Annual Survey Data](https://www.cdc.gov/brfss/annual_data/annual_data.htm)
2. Place all files in `data/source/`
3. Run notebooks `00` → `05` in sequence

> **Note on model files:** Trained models in `models/saved_models/` are excluded from Git.
> Random Forest files are 2–3 GB; XGBoost files are ~450 KB.
> All models are fully reproducible by running `04_modeling.ipynb`.
> `models/scaler.pkl` (< 1 KB) is tracked and included.

All random seeds are fixed at `random_state=42`. Every decision is documented in
[`ProjectDriven.md`](ProjectDriven.md).

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|:-------:|---------|
| Python | 3.11 | Core language |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.26 | Numerical operations |
| matplotlib | ≥ 3.7 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisation |
| scikit-learn | 1.4.2 | Models, preprocessing, metrics |
| imbalanced-learn | 0.14.1 | SMOTE oversampling |
| statsmodels | ≥ 0.14 | VIF calculation |
| scipy | ≥ 1.11 | Chi-square test |
| xgboost | ≥ 2.0 | Gradient boosting |
| shap | ≥ 0.44 | Model interpretability |
| joblib | ≥ 1.3 | Model serialisation |

---

## 📄 Docs

| Document | Description |
|----------|-------------|
| [`ProjectDriven.md`](ProjectDriven.md) | Living project log — all phase decisions and rationale |
| [`docs/methodology.md`](docs/methodology.md) | Full methodology — data, cleaning, modelling, evaluation |
| [`docs/findings.md`](docs/findings.md) | Key findings — results, SHAP insights, limitations |
| [`docs/p2_ProjectScope.md`](docs/p2_ProjectScope.md) | Original project specification |

---

## 📁 Repository Structure

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
│   ├── raw/                      # Extracted year CSVs (not tracked by Git)
│   └── processed/                # Cleaned and split data (not tracked by Git)
│
├── docs/
│   ├── p2_ProjectScope.md
│   ├── methodology.md            ✅ Complete
│   └── findings.md               ✅ Complete
│
├── models/
│   ├── scaler.pkl                # Fitted StandardScaler — tracked by Git
│   └── saved_models/             # Trained models — excluded from Git (reproducible)
│
├── notebooks/
│   ├── 00_data_collection.ipynb      ✅
│   ├── 01_data_understanding.ipynb   ✅
│   ├── 02_cleaning.ipynb             ✅
│   ├── 03_feature_engineering.ipynb  ✅
│   ├── 04_modeling.ipynb             ✅
│   └── 05_evaluation.ipynb           ✅
│
└── outputs/
    └── figures/                  # 21 figures (Phases 1–5)
```

---

*Data: CDC BRFSS 2022–2024 · Python 3.11 · 2024–2025*  
*Figures saved to `outputs/figures/` · Full methodology in `docs/methodology.md`*
