from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd

from api.schemas import (
    ProviderPredictionRequest,
    PredictionResponse,
)

from api.prediction import (
    predict_provider,
    model,
    features,
    threshold,
    model_version,
)

from api.investigation import router as investigation_router

from src.database.prediction_repository import (
    create_prediction_run,
    update_prediction_run,
    save_predictions,
    get_prediction_runs,
    get_predictions_by_run,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Healthcare Fraud Detection API",
    description="Provider-level healthcare fraud risk prediction service",
    version="2.1.0",
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(investigation_router)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Healthcare Fraud Detection API",
        "version": "2.1.0",
        "status": "running",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": model_version,
        "model_type": "RandomForestClassifier",
        "threshold": float(threshold),
    }


# ============================================================
# SINGLE PROVIDER PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: ProviderPredictionRequest):

    try:

        result = predict_provider(
            request.features.model_dump()
        )

        result["provider_id"] = request.provider_id

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/predict/batch")
def predict_batch(
    requests: List[ProviderPredictionRequest]
):

    # --------------------------------------------------------
    # Empty request
    # --------------------------------------------------------

    if not requests:

        return {
            "success": True,
            "run_id": None,
            "count": 0,
            "predictions": [],
            "database": {
                "success": True,
                "saved_count": 0,
            },
        }

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    try:

        data = [
            request.features.model_dump()
            for request in requests
        ]

        df = pd.DataFrame(data)

        X = df[features]

        # ----------------------------------------------------
        # Vectorized prediction
        # ----------------------------------------------------

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        # ----------------------------------------------------
        # Risk classification
        # ----------------------------------------------------

        risk_levels = []

        for probability in probabilities:

            probability = float(probability)

            if probability < 0.30:

                risk_levels.append("Low")

            elif probability < 0.70:

                risk_levels.append("Medium")

            else:

                risk_levels.append("High")

        # ----------------------------------------------------
        # Build prediction results
        # ----------------------------------------------------

        results = []

        for (
            request,
            prediction,
            probability,
            risk_level,
        ) in zip(
            requests,
            predictions,
            probabilities,
            risk_levels,
        ):

            results.append(
                {
                    "provider_id": request.provider_id,
                    "prediction": int(prediction),
                    "fraud_probability": round(
                        float(probability),
                        4,
                    ),
                    "risk_level": risk_level,
                    "model_version": model_version,
                    "threshold": float(threshold),
                }
            )

        # ----------------------------------------------------
        # Risk statistics
        # ----------------------------------------------------

        high_risk_count = sum(
            1
            for item in results
            if item["risk_level"] == "High"
        )

        medium_risk_count = sum(
            1
            for item in results
            if item["risk_level"] == "Medium"
        )

        low_risk_count = sum(
            1
            for item in results
            if item["risk_level"] == "Low"
        )

        provider_count = len(results)

        # ----------------------------------------------------
        # Create prediction run
        # ----------------------------------------------------

        run_result = create_prediction_run(
            model_version=model_version,
            provider_count=provider_count,
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            status="Running",
        )

        if not run_result["success"]:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to create prediction run: "
                    + run_result["error"]
                ),
            )

        run_id = run_result["run_id"]

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        database_result = save_predictions(
            results,
            run_id,
        )

        if not database_result["success"]:

            update_prediction_run(
                run_id=run_id,
                provider_count=provider_count,
                high_risk_count=high_risk_count,
                medium_risk_count=medium_risk_count,
                low_risk_count=low_risk_count,
                status="Failed",
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to save predictions: "
                    + database_result["error"]
                ),
            )

        # ----------------------------------------------------
        # Complete prediction run
        # ----------------------------------------------------

        update_result = update_prediction_run(
            run_id=run_id,
            provider_count=provider_count,
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count,
            status="Completed",
        )

        if not update_result["success"]:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Predictions saved, but failed "
                    "to update prediction run: "
                    + update_result["error"]
                ),
            )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "success": True,

            "run_id": run_id,

            "count": provider_count,

            "risk_summary": {
                "high": high_risk_count,
                "medium": medium_risk_count,
                "low": low_risk_count,
            },

            "model": {
                "version": model_version,
                "type": "RandomForestClassifier",
                "threshold": float(threshold),
            },

            "predictions": results,

            "database": {
                "success": True,
                "saved_count": database_result["count"],
                "run_id": run_id,
            },
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


# ============================================================
# PREDICTION RUN HISTORY
# ============================================================

@app.get("/prediction-runs")
def prediction_runs():

    try:

        runs = get_prediction_runs()

        return {
            "success": True,
            "count": len(runs),
            "runs": runs,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve prediction runs: "
                + str(e)
            ),
        )


# ============================================================
# GET PREDICTIONS FOR A SPECIFIC RUN
# ============================================================

@app.get("/predictions/run/{run_id}")
def predictions_by_run(run_id: int):

    if run_id < 1:

        raise HTTPException(
            status_code=400,
            detail="run_id must be greater than or equal to 1",
        )

    try:

        predictions = get_predictions_by_run(
            run_id=run_id
        )

        if not predictions:

            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for run {run_id}",
            )

        return {
            "success": True,
            "run_id": run_id,
            "count": len(predictions),
            "predictions": predictions,
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to retrieve predictions "
                f"for run {run_id}: {str(e)}"
            ),
        )