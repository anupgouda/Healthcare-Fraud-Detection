import requests


BASE_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_version"] == "2.1"
    assert data["model_type"] == "RandomForestClassifier"

    assert 0 < data["threshold"] < 1


def test_root():
    response = requests.get(
        f"{BASE_URL}/",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


def test_prediction_runs():
    response = requests.get(
        f"{BASE_URL}/prediction-runs",
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    # Actual API response structure:
    # {
    #     "success": True,
    #     "count": 5,
    #     "runs": [...]
    # }

    assert data["success"] is True
    assert data["count"] >= 1
    assert isinstance(data["runs"], list)
    assert len(data["runs"]) >= 1

    run = data["runs"][0]

    assert "id" in run
    assert "model_version" in run
    assert "provider_count" in run
    assert "high_risk_count" in run
    assert "medium_risk_count" in run
    assert "low_risk_count" in run
    assert "status" in run


def test_existing_run():
    response = requests.get(
        f"{BASE_URL}/predictions/run/2",
        timeout=60,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["run_id"] == 2
    assert data["count"] == 5410
    assert len(data["predictions"]) == 5410


def test_known_provider_prediction():
    response = requests.get(
        f"{BASE_URL}/predictions/run/2",
        timeout=60,
    )

    assert response.status_code == 200

    data = response.json()

    provider = next(
        (
            p
            for p in data["predictions"]
            if p["provider_id"] == "PRV51390"
        ),
        None,
    )

    assert provider is not None

    assert provider["run_id"] == 2
    assert provider["provider_id"] == "PRV51390"
    assert provider["prediction"] == 1
    assert provider["risk_level"] == "High"
    assert provider["model_version"] == "2.1"

    assert 0 <= provider["fraud_probability"] <= 1
    assert 0 < provider["threshold"] < 1