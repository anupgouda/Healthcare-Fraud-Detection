from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.database.connection import get_connection
from src.database.prediction_repository import (
    get_prediction_by_run_and_provider,
)


router = APIRouter(
    prefix="/investigations",
    tags=["Investigations"],
)


# ============================================================
# VALID VALUES
# ============================================================

VALID_STATUSES = {
    "Open",
    "Under Review",
    "Resolved",
}

VALID_PRIORITIES = {
    "Low",
    "Normal",
    "High",
    "Critical",
}


# ============================================================
# SCHEMAS
# ============================================================

class InvestigationCreate(BaseModel):
    provider_id: str = Field(
        min_length=1,
        max_length=50,
    )

    # New run-aware field
    run_id: int = Field(
        ge=1,
    )

    status: str = "Open"

    priority: str = "Normal"

    assigned_to: Optional[str] = None

    notes: Optional[str] = None


class InvestigationUpdate(BaseModel):
    status: Optional[str] = None

    priority: Optional[str] = None

    assigned_to: Optional[str] = None

    notes: Optional[str] = None


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_all_investigations():
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                i.id,
                i.provider_id,
                i.prediction_id,
                p.run_id,
                p.fraud_probability,
                p.prediction,
                p.risk_level,
                p.model_version,
                p.threshold,
                i.status,
                i.priority,
                i.assigned_to,
                i.notes,
                i.created_at,
                i.updated_at
            FROM investigations i
            LEFT JOIN predictions p
                ON i.prediction_id = p.id
            ORDER BY i.created_at DESC
            """
        )

        rows = cursor.fetchall()

        columns = [
            "id",
            "provider_id",
            "prediction_id",
            "run_id",
            "fraud_probability",
            "prediction",
            "risk_level",
            "model_version",
            "threshold",
            "status",
            "priority",
            "assigned_to",
            "notes",
            "created_at",
            "updated_at",
        ]

        return [
            dict(zip(columns, row))
            for row in rows
        ]

    finally:
        if cursor:
            cursor.close()

        if connection:
            connection.close()


def get_investigation_by_id(investigation_id):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                i.id,
                i.provider_id,
                i.prediction_id,
                p.run_id,
                p.fraud_probability,
                p.prediction,
                p.risk_level,
                p.model_version,
                p.threshold,
                i.status,
                i.priority,
                i.assigned_to,
                i.notes,
                i.created_at,
                i.updated_at
            FROM investigations i
            LEFT JOIN predictions p
                ON i.prediction_id = p.id
            WHERE i.id = %s
            """,
            (investigation_id,),
        )

        row = cursor.fetchone()

        if not row:
            return None

        columns = [
            "id",
            "provider_id",
            "prediction_id",
            "run_id",
            "fraud_probability",
            "prediction",
            "risk_level",
            "model_version",
            "threshold",
            "status",
            "priority",
            "assigned_to",
            "notes",
            "created_at",
            "updated_at",
        ]

        return dict(zip(columns, row))

    finally:
        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# CREATE INVESTIGATION
# ============================================================

@router.post("/")
def create_investigation(data: InvestigationCreate):

    # --------------------------------------------------------
    # Validate status
    # --------------------------------------------------------

    if data.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid status '{data.status}'. "
                f"Allowed values: {sorted(VALID_STATUSES)}"
            ),
        )

    # --------------------------------------------------------
    # Validate priority
    # --------------------------------------------------------

    if data.priority not in VALID_PRIORITIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid priority '{data.priority}'. "
                f"Allowed values: {sorted(VALID_PRIORITIES)}"
            ),
        )

    # --------------------------------------------------------
    # Find prediction using RUN + PROVIDER
    # --------------------------------------------------------

    prediction = get_prediction_by_run_and_provider(
        data.run_id,
        data.provider_id,
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No prediction found for provider "
                f"'{data.provider_id}' in run {data.run_id}"
            ),
        )

    prediction_id = prediction["id"]

    # --------------------------------------------------------
    # Insert investigation
    # --------------------------------------------------------

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO investigations (
                provider_id,
                prediction_id,
                status,
                priority,
                assigned_to,
                notes
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                data.provider_id,
                prediction_id,
                data.status,
                data.priority,
                data.assigned_to,
                data.notes,
            ),
        )

        investigation_id = cursor.fetchone()[0]

        connection.commit()

        return {
            "success": True,
            "message": "Investigation created successfully",
            "investigation_id": investigation_id,
            "provider_id": data.provider_id,
            "run_id": prediction["run_id"],
            "prediction_id": prediction_id,
            "fraud_probability": float(
                prediction["fraud_probability"]
            ),
            "risk_level": prediction["risk_level"],
            "model_version": prediction["model_version"],
            "status": data.status,
            "priority": data.priority,
        }

    except Exception as e:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# GET ALL INVESTIGATIONS
# ============================================================

@router.get("/")
def list_investigations():

    try:

        investigations = get_all_investigations()

        return {
            "success": True,
            "count": len(investigations),
            "investigations": investigations,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# GET SINGLE INVESTIGATION
# ============================================================

@router.get("/{investigation_id}")
def get_investigation(investigation_id: int):

    investigation = get_investigation_by_id(
        investigation_id
    )

    if not investigation:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    return {
        "success": True,
        "investigation": investigation,
    }


# ============================================================
# UPDATE INVESTIGATION
# ============================================================

@router.patch("/{investigation_id}")
def update_investigation(
    investigation_id: int,
    data: InvestigationUpdate,
):

    # --------------------------------------------------------
    # Validate status
    # --------------------------------------------------------

    if (
        data.status is not None
        and data.status not in VALID_STATUSES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid status '{data.status}'. "
                f"Allowed values: {sorted(VALID_STATUSES)}"
            ),
        )

    # --------------------------------------------------------
    # Validate priority
    # --------------------------------------------------------

    if (
        data.priority is not None
        and data.priority not in VALID_PRIORITIES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid priority '{data.priority}'. "
                f"Allowed values: {sorted(VALID_PRIORITIES)}"
            ),
        )

    # --------------------------------------------------------
    # Check investigation exists
    # --------------------------------------------------------

    existing = get_investigation_by_id(
        investigation_id
    )

    if not existing:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    # --------------------------------------------------------
    # Update only provided fields
    # --------------------------------------------------------

    fields = []
    values = []

    if data.status is not None:
        fields.append("status = %s")
        values.append(data.status)

    if data.priority is not None:
        fields.append("priority = %s")
        values.append(data.priority)

    if data.assigned_to is not None:
        fields.append("assigned_to = %s")
        values.append(data.assigned_to)

    if data.notes is not None:
        fields.append("notes = %s")
        values.append(data.notes)

    # Nothing to update
    if not fields:
        return {
            "success": True,
            "message": "No changes requested",
            "investigation": existing,
        }

    fields.append(
        "updated_at = CURRENT_TIMESTAMP"
    )

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        query = f"""
            UPDATE investigations
            SET
                {", ".join(fields)}
            WHERE id = %s
        """

        values.append(investigation_id)

        cursor.execute(
            query,
            tuple(values),
        )

        connection.commit()

        updated = get_investigation_by_id(
            investigation_id
        )

        return {
            "success": True,
            "message": "Investigation updated successfully",
            "investigation": updated,
        }

    except Exception as e:

        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()