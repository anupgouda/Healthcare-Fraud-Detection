import pytest

from src.database.prediction_repository import (
    get_prediction_by_run_and_provider,
)


RUN_ID = 2
PROVIDER_ID = "PRV51390"


def test_prediction_repository_returns_prediction():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    assert result is not None


def test_prediction_repository_prediction_id():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    assert result["id"] == 11128


def test_prediction_repository_run_id():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    assert result["run_id"] == RUN_ID


def test_prediction_repository_provider_id():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    assert result["provider_id"] == PROVIDER_ID


def test_prediction_repository_prediction():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    assert result["prediction"] == 1


def test_prediction_repository_risk():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    assert result["risk_level"] == "High"


def test_prediction_repository_probability():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    probability = float(result["fraud_probability"])

    assert 0 <= probability <= 1
    assert probability == pytest.approx(0.9992)


def test_prediction_repository_model_version():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    assert result["model_version"] == "2.1"


def test_prediction_repository_threshold():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        PROVIDER_ID,
    )

    threshold = float(result["threshold"])

    assert threshold == pytest.approx(0.45)


def test_prediction_repository_unknown_provider():
    result = get_prediction_by_run_and_provider(
        RUN_ID,
        "PROVIDER_DOES_NOT_EXIST",
    )

    assert result is None