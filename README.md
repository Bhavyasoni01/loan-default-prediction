# 🏦 Loan Default Prediction System

## 📌 Overview

This repository contains an end-to-end Machine Learning pipeline designed to predict loan default probabilities. The model is trained on historical loan datasets and deployed as a scalable RESTful API.

Built with modern MLOps best practices, the project includes:

* Automated model training
* Experiment tracking with MLflow
* SHAP explainability visualizations
* Dockerized deployment
* CI/CD automation using GitHub Actions
* Model validation workflows

---

# 🛠️ Technology Stack

## Machine Learning

* `scikit-learn`
* `pandas`
* `xgboost`
* `numpy`
* `optuna`
* `SHAP`

## MLOps & Tracking

* `MLflow`

## Backend API

* `FastAPI`
* `Uvicorn`
* `Pydantic`

## Infrastructure & DevOps

* `Docker`
* `GitHub Actions`

---

# 📂 Project Structure

```bash
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml
├── notebooks/
│   ├── loanDefaulter.ipynb
├── .dockerignore
├── .gitignore
├── app.py
├── create_test_dataset.py
├── dockerfile.api
├── README.md
├── requirements.txt
├── sample_dataset.csv
├── saved_model.pkl
└── validate_model.py
```

---

# 📊 Dataset Information

The project uses historical loan-related data containing borrower financial details and loan repayment status.

### Features Include:

* Income
* Credit history
* Loan amount
* Debt ratio
* Employment details
* Loan repayment outcome

> **Note:** Large datasets are excluded from version control. Place dataset files locally before training the model.

---

# 🚀 Local Development Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Bhavyasoni01/loan-default-prediction.git
cd loan-default-prediction
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Linux / macOS

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Model Training

Run the notebook or training scripts to build the model.

```bash
python validate_model.py
```

or open:

```bash
notebooks/loanDefaulter.ipynb
```

---

# 📈 MLflow Experiment Tracking

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

Visit:

```text
http://localhost:5000
```

to monitor experiments, metrics, and artifacts.

---

# ⚡ Running the API

Start the FastAPI server locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 🐳 Docker Deployment

## Build Docker Image

```bash
docker build -t loan-default-api -f dockerfile.api .
```

---

## Run Docker Container

```bash
docker run -d -p 8080:8080 --name loan-api loan-default-api
```

API will run at:

```text
http://localhost:8080
```

---

# 🔄 CI/CD Pipeline

GitHub Actions automatically performs:

* Environment setup
* Dependency installation
* Model validation
* API testing
* Docker build workflows

Pipeline configuration:

```text
.github/workflows/ml_pipeline.yml
```

---

# 📡 API Endpoints

| Method | Endpoint   | Description                      |
| ------ | ---------- | -------------------------------- |
| GET    | `/health`        | Health check                     |
| POST   | `/predict` | Predict loan default probability |
| GET    | `/docs`    | Swagger UI documentation         |

---

# 📉 Explainability

The project includes SHAP visualizations for model interpretability:

* `shap_plot.png`
* `shap_summary_plot.png`

These plots help explain feature importance and prediction behavior.

---

# ✅ Features

* End-to-end ML workflow
* FastAPI inference API
* Dockerized deployment
* MLflow tracking
* SHAP explainability
* GitHub Actions CI/CD
* Model validation pipeline
* Production-ready structure

---

