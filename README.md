# 🏥 Healthcare Fraud Analytics Platform

An end-to-end healthcare fraud analytics platform that uses machine learning to identify healthcare provider patterns associated with potential fraud and supports investigation workflows.

> **Important:** The model produces a risk signal for investigation. A high-risk prediction does not establish that fraud has actually occurred.

---
## Live Demo

https://healthcare-fraud-detection-duybv2cwv92n34bezyivk4.streamlit.app/

## GitHub Repository

https://github.com/anupgouda/Healthcare-Fraud-Detection

---

## 🚀 Project Overview

This project analyzes inpatient, outpatient, and beneficiary healthcare claims data at the provider level.

The platform combines:

- Data preprocessing
- Provider-level feature engineering
- Machine learning prediction
- Risk classification
- FastAPI prediction services
- PostgreSQL prediction storage
- Prediction run history
- Investigation management
- SHAP-based explainability
- Interactive Streamlit dashboard
- Automated testing

---
## 📊 Dataset

This project uses the publicly available **Healthcare Provider Fraud Detection Analysis** dataset, which contains Medicare healthcare claims and beneficiary information for provider-level fraud-risk modeling.

### Dataset Source

**Primary source:** [Kaggle — Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis?utm_source=chatgpt.com)

**Alternative verified download source:** [Zenodo — Medicare Fraud Detection Dataset](https://zenodo.org/records/18138102?utm_source=chatgpt.com)

The dataset contains separate provider, beneficiary, inpatient-claim, and outpatient-claim files. The project uses the labeled **training files** for feature engineering and prediction.

### Files Used by FraudLens AI

| File                                      | Description                                                                                                | Approx. Size |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -----------: |
| `Train-1542865627584.csv`                 | Provider-level labels containing `Provider` and `PotentialFraud`                                           |        87 KB |
| `Train_Beneficiarydata-1542865627584.csv` | Beneficiary demographics, coverage, chronic conditions, reimbursement and deductible information           |      11.4 MB |
| `Train_Inpatientdata-1542865627584.csv`   | Inpatient healthcare claims, diagnoses, procedures, admission/discharge information and reimbursement data |       8.6 MB |
| `Train_Outpatientdata-1542865627584.csv`  | Outpatient healthcare claims, diagnoses, procedures and reimbursement information                          |      77.4 MB |

The four training files together form the input required by the **Run a New Healthcare Analysis** workflow.

### Dataset Scale

The training dataset contains approximately:

* **5,410 providers**
* **138,556 beneficiary records**
* **40,474 inpatient claims**
* **517,737 outpatient claims**

The provider label file contains the `PotentialFraud` target used for supervised learning.

### How the Dataset Is Used

FraudLens AI processes the datasets in the following order:

```text
Training CSV Files
       │
       ▼
Date Conversion & Cleaning
       │
       ▼
Claims + Beneficiary Join
       │
       ▼
Provider-Level Feature Engineering
       │
       ▼
Provider Aggregation
       │
       ▼
16 Model Features
       │
       ▼
Random Forest Classifier
       │
       ▼
Fraud-Risk Probability
       │
       ├── Low
       ├── Medium
       └── High
       │
       ▼
FastAPI
       │
       ▼
PostgreSQL
```

### Downloading the Dataset

To run a new analysis on the live application, download the dataset from the Kaggle source above and extract the training files.

Upload these three source datasets in FraudLens AI:

1. `Train_Beneficiarydata-1542865627584.csv`
2. `Train_Inpatientdata-1542865627584.csv`
3. `Train_Outpatientdata-1542865627584.csv`

The provider label file (`Train-1542865627584.csv`) is used during the model-development/training workflow and is not required by the deployed prediction upload interface.

### Important Note

This is a publicly available research dataset. It is used for machine-learning experimentation and demonstration of provider-level risk analysis.

The model produces a **risk signal for investigation**. A high-risk prediction does not establish that a provider committed fraud or constitute a legal finding.

----

## 🏗️ System Architecture

```text
Healthcare Claims Data
        │
        ▼
Data Preprocessing
        │
        ▼
Provider-Level Feature Engineering
        │
        ▼
Random Forest Model
        │
        ▼
Fraud Probability
        │
        ├── Low Risk
        ├── Medium Risk
        └── High Risk
        │
        ▼
FastAPI
        │
        ├── Prediction API
        ├── Prediction Runs
        └── Investigation API
        │
        ▼
PostgreSQL
        │
        ├── Predictions
        ├── Prediction Runs
        └── Investigations
        │
        ▼
Streamlit Analytics Dashboard
        │
        └── SHAP Explainability
🛠️ Technology Stack
Category	Technology
Language	Python 3.11
Data Analysis	Pandas, NumPy
Machine Learning	Scikit-learn
Model Persistence	Joblib
API	FastAPI
API Server	Uvicorn
Validation	Pydantic
Database	PostgreSQL
Database Driver	psycopg2
Explainability	SHAP
Visualization	Plotly
Dashboard	Streamlit
Testing	Pytest
📁 Project Structure
HEALTHCARE-FRAUD-DETECTION/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── api/
│   ├── main.py
│   ├── prediction.py
│   ├── schemas.py
│   └── investigation.py
│
├── data/
│   ├── Train-1542865627584.csv
│   ├── Train_Beneficiarydata-1542865627584.csv
│   ├── Train_Inpatientdata-1542865627584.csv
│   └── Train_Outpatientdata-1542865627584.csv
│
├── models/
│   └── fraud_pipeline.pkl
│
├── notebooks/
│   ├── Sagility1.ipynb
│   └── Sagility1_engineering_v2.ipynb
│
├── src/
│   ├── api_client.py
│   ├── model_input.py
│   ├── preprocessing.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── prediction_repository.py
│   │
│   └── explainability/
│       ├── __init__.py
│       └── shap_explainer.py
│
├── tests/
│   ├── __init__.py
│   ├── test_integration.py
│   ├── test_investigations.py
│   ├── test_model.py
│   ├── test_prediction_api.py
│   └── test_prediction_repository.py
│
├── test_explainability.py
│
└── images/
    ├── dashboard.png
    ├── analysis.png
    └── providers.png
⚙️ Installation
1. Clone the repository
git clone <repository_url>
cd Healthcare-Fraud-Detection
2. Create the Python environment

Using Conda:

conda create -n fraud python=3.11
conda activate fraud
3. Install dependencies
pip install -r requirements.txt
4. Configure PostgreSQL

Create the database:

CREATE DATABASE healthcare_fraud;

Configure the PostgreSQL connection using the environment variables expected by the application.

▶️ Run the Application
Start FastAPI
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

API:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs
Start Streamlit

In another terminal:

streamlit run app.py
🤖 Machine Learning Model

The production model is stored at:

models/fraud_pipeline.pkl
Model Configuration
Property	Value
Model	RandomForestClassifier
Version	2.1
Features	16
Threshold	0.45
Random State	42
Evaluation Metrics
Metric	Value
Accuracy	0.9473
Precision	0.7391
Recall	0.6733
F1 Score	0.7047
ROC-AUC	0.9673
PR-AUC	0.7864
🔍 Prediction Workflow
Provider Data
     │
     ▼
Feature Validation
     │
     ▼
Model Prediction
     │
     ▼
Fraud Probability
     │
     ▼
Risk Classification
     │
     ├── Low
     ├── Medium
     └── High
     │
     ▼
Database Storage

The binary prediction threshold is:

0.45

Risk levels are intended as an investigation-prioritization signal.

🗄️ PostgreSQL

The platform stores prediction and investigation information in PostgreSQL.

Core entities:

prediction_runs
      │
      ▼
predictions
      │
      ▼
investigations
Prediction Runs

Stores:

Model version
Provider count
High-risk count
Medium-risk count
Low-risk count
Run status
Start time
Completion time
Predictions

Stores:

Provider ID
Fraud probability
Prediction
Risk level
Model version
Threshold
Prediction run
Investigations

Stores:

Provider ID
Prediction ID
Investigation status
Priority
Assigned investigator/team
Notes
Created time
Updated time
🔎 Investigation Workflow

Investigators can create an investigation for a provider prediction.

Statuses
Open
Under Review
Resolved
Priorities
Low
Normal
High
Critical

The investigation API validates that the provider prediction exists before creating an investigation.

🧠 Explainability

The platform includes SHAP-based explainability to help users understand which model features contributed to a prediction.

Implementation:

src/explainability/shap_explainer.py
📊 Dashboard

The Streamlit dashboard provides:

Provider risk analysis
Prediction results
High-risk provider identification
Investigation management
Model explainability
Analytics
Prediction run history
Data exploration
Dashboard Preview
Dashboard

Fraud Analysis

High-Risk Providers

🧪 Testing

The project includes automated tests for:

Machine learning model
Model artifact
Prediction API
Prediction repository
PostgreSQL-backed prediction retrieval
Investigation API
Investigation validation

Run the complete test suite:

python -m pytest -v

Current verified result:

30 passed

There are currently three dependency-level SHAP deprecation warnings. These do not represent failing application tests.

📌 API Endpoints
GET    /
GET    /health

GET    /prediction-runs
GET    /predictions/run/{run_id}

GET    /investigations
GET    /investigations/{investigation_id}
POST   /investigations
PATCH  /investigations/{investigation_id}

FastAPI documentation:

http://127.0.0.1:8000/docs
🔐 Project Safety

This platform is designed as an analytics and investigation-support system.

A model prediction should not be interpreted as proof of fraud. Predictions identify historical patterns associated with potential risk and can be used to prioritize further human review.

🔮 Future Improvements
Model monitoring
Data drift detection
Automated retraining pipelines
Advanced anomaly detection
Cloud deployment
Role-based authentication
Audit logging
Advanced investigation workflows
Additional explainability methods
Production monitoring and observability
👨‍💻 Author

Appaji Gouda

Artificial Intelligence and Machine Learning Engineer
