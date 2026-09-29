from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.pkl"


def test_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"


def test_model_artifact_structure():
    artifact = joblib.load(MODEL_PATH)

    assert isinstance(artifact, dict)

    assert "model" in artifact
    assert "features" in artifact
    assert "threshold" in artifact
    assert "model_version" in artifact


def test_model_features():
    artifact = joblib.load(MODEL_PATH)

    features = artifact["features"]

    assert isinstance(features, list)
    assert len(features) == 16


def test_model_threshold():
    artifact = joblib.load(MODEL_PATH)

    threshold = artifact["threshold"]

    assert 0 < threshold < 1
    assert threshold == pytest.approx(0.45)


def test_model_version():
    artifact = joblib.load(MODEL_PATH)

    assert artifact["model_version"] == "2.1"


def test_model_prediction():
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    features = artifact["features"]

    sample = {
        feature: 0
        for feature in features
    }

    df = pd.DataFrame([sample])

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= artifact["threshold"])

    assert 0 <= probability <= 1
    assert prediction in [0, 1]


def test_model_metrics_exist():
    artifact = joblib.load(MODEL_PATH)

    assert "metrics" in artifact

    metrics = artifact["metrics"]

    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    for metric in required_metrics:
        assert metric in metrics
        assert 0 <= metrics[metric] <= 1