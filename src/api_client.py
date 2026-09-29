import requests


# ============================================================
# API CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# PREDICTION
# ============================================================

def predict_providers(provider_features):

    response = requests.post(
        f"{API_BASE_URL}/predict/batch",
        json=provider_features,
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# PREDICTION RUN HISTORY
# ============================================================

def get_prediction_runs():

    response = requests.get(
        f"{API_BASE_URL}/prediction-runs",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# GET PREDICTIONS FOR A RUN
# ============================================================

def get_predictions_by_run(run_id):

    response = requests.get(
        f"{API_BASE_URL}/predictions/run/{run_id}",
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# INVESTIGATIONS
# ============================================================

def get_investigations():

    response = requests.get(
        f"{API_BASE_URL}/investigations/",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_investigation(investigation_id):

    response = requests.get(
        f"{API_BASE_URL}/investigations/{investigation_id}",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def create_investigation(
    provider_id,
    run_id,
    status="Open",
    priority="Normal",
    assigned_to=None,
    notes=None,
):

    payload = {
        "provider_id": provider_id,
        "run_id": run_id,
        "status": status,
        "priority": priority,
        "assigned_to": assigned_to,
        "notes": notes,
    }

    response = requests.post(
        f"{API_BASE_URL}/investigations/",
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def update_investigation(
    investigation_id,
    status=None,
    priority=None,
    assigned_to=None,
    notes=None,
):

    payload = {
        "status": status,
        "priority": priority,
        "assigned_to": assigned_to,
        "notes": notes,
    }

    response = requests.patch(
        f"{API_BASE_URL}/investigations/{investigation_id}",
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()