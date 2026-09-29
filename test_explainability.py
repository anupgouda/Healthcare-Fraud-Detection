import joblib
import pandas as pd

from src.preprocessing import (
    convert_date_columns,
    create_features
)

from src.model_input import (
    provider_aggregation
)

from src.explainability.shap_explainer import (
    FraudExplainer
)


# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

artifact = joblib.load(
    "models/fraud_pipeline.pkl"
)

model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]


# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

beneficiary = pd.read_csv(
    "data/Train_Beneficiarydata-1542865627584.csv"
)

inpatient = pd.read_csv(
    "data/Train_Inpatientdata-1542865627584.csv"
)

outpatient = pd.read_csv(
    "data/Train_Outpatientdata-1542865627584.csv"
)


# ----------------------------------------------------
# DATE CONVERSION
# ----------------------------------------------------

beneficiary = convert_date_columns(
    beneficiary,
    ["DOB", "DOD"]
)

inpatient = convert_date_columns(
    inpatient,
    [
        "ClaimStartDt",
        "ClaimEndDt",
        "AdmissionDt",
        "DischargeDt"
    ]
)

outpatient = convert_date_columns(
    outpatient,
    [
        "ClaimStartDt",
        "ClaimEndDt"
    ]
)


# ----------------------------------------------------
# COMBINE CLAIMS
# ----------------------------------------------------

claims = pd.concat(
    [
        inpatient,
        outpatient
    ],
    ignore_index=True
)


# ----------------------------------------------------
# MERGE BENEFICIARY
# ----------------------------------------------------

claims = claims.merge(
    beneficiary,
    on="BeneID",
    how="left"
)


# ----------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------

claims = create_features(
    claims
)


# ----------------------------------------------------
# PROVIDER AGGREGATION
# ----------------------------------------------------

aggregated = provider_aggregation(
    claims
)

aggregated = aggregated.fillna(0)


# ----------------------------------------------------
# MODEL INPUT
# ----------------------------------------------------

X = aggregated[
    features
]


# ----------------------------------------------------
# MODEL PREDICTION
# ----------------------------------------------------

probabilities = model.predict_proba(
    X
)[:, 1]


predictions = (
    probabilities >= threshold
).astype(int)


# ----------------------------------------------------
# ADD RESULTS
# ----------------------------------------------------

aggregated["FraudProbability"] = probabilities

aggregated["Prediction"] = predictions


# ----------------------------------------------------
# FIND HIGHEST-RISK PROVIDER
# ----------------------------------------------------

highest_risk_index = (
    aggregated[
        "FraudProbability"
    ]
    .idxmax()
)


provider_row = X.loc[
    highest_risk_index
]


provider_id = aggregated.loc[
    highest_risk_index,
    "Provider"
]


probability = aggregated.loc[
    highest_risk_index,
    "FraudProbability"
]


prediction = aggregated.loc[
    highest_risk_index,
    "Prediction"
]


# ----------------------------------------------------
# DISPLAY PROVIDER
# ----------------------------------------------------

print("\n")
print("=" * 60)

print(
    f"Provider: {provider_id}"
)

print(
    f"Fraud Probability: {probability:.4f}"
)

print(
    f"Prediction: {prediction}"
)

print("=" * 60)


# ----------------------------------------------------
# SHAP EXPLAINER
# ----------------------------------------------------

explainer = FraudExplainer(
    model,
    features
)


explanation = explainer.explain(
    provider_row
)


# ----------------------------------------------------
# DISPLAY EXPLANATION
# ----------------------------------------------------

print("\nTop contributing features:\n")

print(
    explanation.head(10).to_string(
        index=False
    )
)