# 🏦 Loan Default Prediction System

![ML Pipeline](https://github.com/Bhavyasoni01/loan-default-prediction/actions/workflows/ml_pipeline.yml/badge.svg)

## 📌 Overview

Production-style MLOps system for loan default prediction with SHAP-based
explainability, MLflow experiment tracking, Dockerized deployment, and
dual FastAPI endpoints for prediction + risk reasoning.

Trained on 2.26M real Lending Club loan records. Designed to resemble production-style lending risk APIs used in regulated financial systems where explainability is important.

**Deployment:** Containerized with Docker, deployed on GCP Compute Engine (e2-small, asia-south2-c)

### What makes this different from a basic ML project:
Unlike typical ML projects that return only a classification result,
this system generates prediction level explanations using SHAP values,
surfacing the financial features that most influenced each risk decision.

This mirrors explainability requirements commonly seen in regulated
lending and fintech systems.

## 📊 Dataset

**Source:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — Kaggle

| Property | Detail |
|----------|--------|
| Raw rows | 2,260,701 |
| Features | 151 → reduced to ~73 after cleaning |
| Class balance | ~81% non-default, ~19% default |

### Data Challenges Solved:
- Removed 14 post-loan leakage columns (payment history, recovery amounts)
- Parsed messy string columns (`term`, `emp_length`, `revol_util`)
- Dropped columns with >40% missing values
- Engineered 8 new features from raw financial data

## 🧠 Model & Performance

**Algorithm:** XGBoost Classifier with Optuna hyperparameter tuning

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.71+ |
| Threshold | 0.45 (tuned for business tradeoff) |
| Tuning | Optuna (30 trials, maximizing ROC-AUC) |

### Tuned Hyperparameters via Optuna:
- `n_estimators`, `max_depth`, `learning_rate`
- `subsample`, `colsample_bytree`
- `gamma`, `min_child_weight`

### MLflow Experiment Tracking:
All training runs logged with parameters, metrics, and model artifacts.
Best model registered in MLflow Model Registry as `Loan_Defaulter_Detection`.

## 📂 Repository Structure

```
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml
├── notebooks/
├── .dockerignore
├── .gitignore
├── app.py
├── create_test_dataset.py
├── dockerfile.api
├── feature_names.pkl
├── README.md
├── requirements.txt
├── sample_dataset.csv
├── saved_model.pkl
├── shap_explainer.pkl
└── validate_model.py
```

## 📡 API Endpoints 

### `GET /health`
Returns service health status.

---

### `POST /predict`
Returns default probability and decision for a loan applicant.

**Input**

```json
{
  "loan_amnt": 10000,
  "funded_amnt": 10000,
  "funded_amnt_inv": 10000,
  "term": 36,
  "int_rate": 7.5,
  "installment": 311.06,
  "grade": "A",
  "sub_grade": "A2",
  "emp_length": 10,
  "home_ownership": "MORTGAGE",
  "annual_inc": 120000,
  "verification_status": "Source Verified",
  "issue_d": "Jan-2020",
  "purpose": "credit_card",
  "addr_state": "CA",
  "dti": 10.5,
  "delinq_2yrs": 0,
  "earliest_cr_line": "Aug-2010",
  "inq_last_6mths": 0,
  "open_acc": 10,
  "pub_rec": 0,
  "revol_bal": 5000,
  "revol_util": 15.0,
  "total_acc": 20,
  "initial_list_status": "w",
  "collections_12_mths_ex_med": 0,
  "acc_now_delinq": 0,
  "tot_coll_amt": 0,
  "tot_cur_bal": 250000,
  "total_rev_hi_lim": 50000,
  "acc_open_past_24mths": 2,
  "avg_cur_bal": 25000,
  "bc_open_to_buy": 25000,
  "bc_util": 15.0,
  "chargeoff_within_12_mths": 0,
  "delinq_amnt": 0,
  "mo_sin_old_il_acct": 120,
  "mo_sin_old_rev_tl_op": 144,
  "mo_sin_rcnt_rev_tl_op": 12,
  "mo_sin_rcnt_tl": 10,
  "mort_acc": 2,
  "mths_since_recent_bc": 14,
  "mths_since_recent_inq": 12,
  "num_accts_ever_120_pd": 0,
  "num_actv_bc_tl": 2,
  "num_actv_rev_tl": 3,
  "num_bc_sats": 4,
  "num_bc_tl": 6,
  "num_il_tl": 4,
  "num_op_rev_tl": 8,
  "num_rev_accts": 12,
  "num_rev_tl_bal_gt_0": 3,
  "num_sats": 10,
  "num_tl_120dpd_2m": 0,
  "num_tl_30dpd": 0,
  "num_tl_90g_dpd_24m": 0,
  "num_tl_op_past_12m": 1,
  "pct_tl_nvr_dlq": 100.0,
  "percent_bc_gt_75": 0.0,
  "pub_rec_bankruptcies": 0,
  "tax_liens": 0,
  "tot_hi_cred_lim": 300000,
  "total_bal_ex_mort": 25000,
  "total_bc_limit": 35000,
  "total_il_high_credit_limit": 30000,
  "debt_to_installment": 15.5,
  "fico_avg": 760.0,
  "funded_ratio": 1.0,
  "revol_util_bucket": "0-20%",
  "payment_to_income": 0.03,
  "delinq_ratio": 0.0,
  "credit_intensity": 0.5,
  "open_acc_ratio": 0.5
}
```

**Response:**
```json
{
  "prediction": "default",
  "default_probability_percent": 78.43,
  "timestamp": "2026-05-30 15:30:00"
}
```

---

### `POST /explain`
Returns default prediction **with SHAP-powered explanation** — the top 
factors driving this specific applicant's risk score.

This endpoint mirrors what production lending systems use to satisfy 
regulatory explainability requirements.

**Response:**
```json
{
  "prediction": "no_default",
  "default_probability_percent": 8.49,
  "top_reasons": [
    {
      "feature": "int_rate",
      "impact": -0.6394,
      "direction": "decreases default risk"
    },
    {
      "feature": "fico_avg",
      "impact": -0.2144,
      "direction": "decreases default risk"
    },
    {
      "feature": "dti",
      "impact": -0.199,
      "direction": "decreases default risk"
    },
    {
      "feature": "acc_open_past_24mths",
      "impact": -0.1694,
      "direction": "decreases default risk"
    },
    {
      "feature": "payment_to_income",
      "impact": -0.1673,
      "direction": "decreases default risk"
    }
  ],
  "timestamp": "2026-06-03 17:48:21"
}
```

**How to read this:**
- `impact` = how much this feature shifted the default probability (SHAP value)
- Positive impact = pushes toward default
- Negative impact = pushes away from default
- Features sorted by absolute impact (most influential first) 


## 🚀 Quickstart

Run the complete inference API locally using Docker.

```bash
# Clone the repo
git clone https://github.com/Bhavyasoni01/loan-default-prediction.git
cd loan-default-prediction

# Build and run the container
docker build -f dockerfile.api -t loan-prediction-api .
docker run -p 8080:8080 loan-prediction-api

```

## 🛠️ Technology Stack

| Layer | Tools |
|-------|-------|
| **ML** | XGBoost, scikit-learn, pandas, numpy |
| **Tuning** | Optuna |
| **Explainability** | SHAP |
| **Experiment Tracking** | MLflow |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud** | GCP Compute Engine |

--- 

## 🏗️ Architecture Flow
``` 
Raw Data
   ↓
Preprocessing Pipeline
   ↓
Feature Engineering
   ↓
XGBoost + Optuna
   ↓
MLflow Registry
   ↓
FastAPI Service
   ↓
Docker
   ↓
GCP Deployment

```

## 🔄 CI/CD Pipeline

GitHub Actions pipeline automatically:
- Runs API tests
- Builds Docker image
- Validates dependencies
- Verifies inference endpoint health

## 🎯 Key Engineering Decisions

**Why XGBoost over simpler models?**
Handles mixed feature types, missing values, and class imbalance 
via `scale_pos_weight`. Optuna tuning across 7 hyperparameters 
gave measurable AUC improvement over defaults.

**Why SHAP for explainability?**
Tree-based SHAP (TreeExplainer) is computationally efficient and 
provides exact Shapley values for XGBoost. Returns per-prediction 
feature attribution — not just global importance — which is what 
production lending APIs need.

**Why threshold 0.45 instead of 0.50?**
In loan default prediction, missing a defaulter (false negative) 
costs more than incorrectly flagging a good borrower (false positive). 
Lowering the threshold increases recall at an acceptable precision cost.

