import requests


BASE_URL = "http://127.0.0.1:8000"

RUN_ID = 5
PROVIDER_ID = "PRV51390"


def test_list_investigations():
    response = requests.get(
        f"{BASE_URL}/investigations/",
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["count"] >= 1
    assert isinstance(data["investigations"], list)


def test_create_investigation():
    payload = {
        "provider_id": PROVIDER_ID,
        "run_id": RUN_ID,
        "status": "Open",
        "priority": "High",
        "assigned_to": "Fraud Investigation Test",
        "notes": "Automated test investigation.",
    }

    response = requests.post(
        f"{BASE_URL}/investigations/",
        json=payload,
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["message"] == "Investigation created successfully"

    assert data["provider_id"] == PROVIDER_ID
    assert data["run_id"] == RUN_ID

    assert data["prediction_id"] is not None

    assert 0 <= data["fraud_probability"] <= 1

    assert data["risk_level"] == "High"
    assert data["model_version"] == "2.1"

    assert data["status"] == "Open"
    assert data["priority"] == "High"

    # Store the generated investigation ID
    global investigation_id
    investigation_id = data["investigation_id"]


def test_get_created_investigation():
    assert "investigation_id" in globals()

    response = requests.get(
        f"{BASE_URL}/investigations/{investigation_id}",
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    investigation = data["investigation"]

    assert investigation["id"] == investigation_id
    assert investigation["provider_id"] == PROVIDER_ID
    assert investigation["run_id"] == RUN_ID
    assert investigation["status"] == "Open"
    assert investigation["priority"] == "High"
    assert investigation["assigned_to"] == "Fraud Investigation Test"


def test_update_investigation():
    assert "investigation_id" in globals()

    payload = {
        "status": "Under Review",
        "priority": "Critical",
        "assigned_to": "Senior Fraud Investigator",
        "notes": "Automated test update.",
    }

    response = requests.patch(
        f"{BASE_URL}/investigations/{investigation_id}",
        json=payload,
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    investigation = data["investigation"]

    assert investigation["id"] == investigation_id
    assert investigation["status"] == "Under Review"
    assert investigation["priority"] == "Critical"
    assert investigation["assigned_to"] == "Senior Fraud Investigator"
    assert investigation["notes"] == "Automated test update."


def test_invalid_status():
    payload = {
        "provider_id": PROVIDER_ID,
        "run_id": RUN_ID,
        "status": "Invalid Status",
        "priority": "High",
        "assigned_to": "Test",
        "notes": "Should fail.",
    }

    response = requests.post(
        f"{BASE_URL}/investigations/",
        json=payload,
        timeout=30,
    )

    assert response.status_code == 400

    data = response.json()

    assert "Invalid status" in data["detail"]


def test_invalid_priority():
    payload = {
        "provider_id": PROVIDER_ID,
        "run_id": RUN_ID,
        "status": "Open",
        "priority": "Invalid Priority",
        "assigned_to": "Test",
        "notes": "Should fail.",
    }

    response = requests.post(
        f"{BASE_URL}/investigations/",
        json=payload,
        timeout=30,
    )

    assert response.status_code == 400

    data = response.json()

    assert "Invalid priority" in data["detail"]


def test_unknown_provider():
    payload = {
        "provider_id": "PROVIDER_DOES_NOT_EXIST",
        "run_id": RUN_ID,
        "status": "Open",
        "priority": "Normal",
        "assigned_to": "Test",
        "notes": "Should fail.",
    }

    response = requests.post(
        f"{BASE_URL}/investigations/",
        json=payload,
        timeout=30,
    )

    assert response.status_code == 404

    data = response.json()

    assert "No prediction found" in data["detail"]


def test_unknown_investigation():
    response = requests.get(
        f"{BASE_URL}/investigations/999999999",
        timeout=30,
    )

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Investigation not found"