from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.pkl"


artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]
model_version = artifact.get("model_version", "unknown")


def predict_provider(data: dict):

    df = pd.DataFrame([data])

    # Ensure the exact feature order used during training
    df = df[features]

    probability = float(model.predict_proba(df)[0][1])

    prediction = int(probability >= threshold)

    if probability < 0.30:
        risk_level = "Low"
    elif probability < 0.70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 4),
        "risk_level": risk_level,
        "model_version": model_version,
        "threshold": threshold,
    }