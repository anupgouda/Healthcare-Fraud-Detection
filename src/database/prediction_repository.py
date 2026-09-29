from src.database.connection import get_connection


# ============================================================
# CREATE PREDICTION RUN
# ============================================================

def create_prediction_run(
    model_version,
    provider_count,
    high_risk_count=0,
    medium_risk_count=0,
    low_risk_count=0,
    status="Running",
):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO prediction_runs (
                model_version,
                provider_count,
                high_risk_count,
                medium_risk_count,
                low_risk_count,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                model_version,
                provider_count,
                high_risk_count,
                medium_risk_count,
                low_risk_count,
                status,
            ),
        )

        run_id = cursor.fetchone()[0]

        connection.commit()

        return {
            "success": True,
            "run_id": run_id,
        }

    except Exception as e:

        if connection:
            connection.rollback()

        return {
            "success": False,
            "error": str(e),
        }

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# UPDATE PREDICTION RUN
# ============================================================

def update_prediction_run(
    run_id,
    provider_count,
    high_risk_count,
    medium_risk_count,
    low_risk_count,
    status="Completed",
):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE prediction_runs
            SET
                provider_count = %s,
                high_risk_count = %s,
                medium_risk_count = %s,
                low_risk_count = %s,
                status = %s,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (
                provider_count,
                high_risk_count,
                medium_risk_count,
                low_risk_count,
                status,
                run_id,
            ),
        )

        connection.commit()

        return {
            "success": True,
            "run_id": run_id,
        }

    except Exception as e:

        if connection:
            connection.rollback()

        return {
            "success": False,
            "error": str(e),
        }

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(predictions, run_id):

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO predictions (
                run_id,
                provider_id,
                fraud_probability,
                prediction,
                risk_level,
                model_version,
                threshold
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        for item in predictions:

            cursor.execute(
                query,
                (
                    run_id,
                    item["provider_id"],
                    item["fraud_probability"],
                    item["prediction"],
                    item["risk_level"],
                    item["model_version"],
                    item["threshold"],
                ),
            )

        connection.commit()

        return {
            "success": True,
            "count": len(predictions),
            "run_id": run_id,
        }

    except Exception as e:

        if connection:
            connection.rollback()

        return {
            "success": False,
            "error": str(e),
        }

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# GET RECENT PREDICTIONS
# ============================================================

def get_predictions(limit=100):

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                run_id,
                provider_id,
                fraud_probability,
                prediction,
                risk_level,
                model_version,
                threshold,
                created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        columns = [
            "id",
            "run_id",
            "provider_id",
            "fraud_probability",
            "prediction",
            "risk_level",
            "model_version",
            "threshold",
            "created_at",
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


# ============================================================
# GET PREDICTIONS FOR A SPECIFIC RUN
# ============================================================

def get_predictions_by_run(
    run_id,
    limit=10000,
):
    """
    Return predictions belonging to a specific
    prediction run.

    Example:

        Run 2
          ↓
        5,410 predictions
    """

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                run_id,
                provider_id,
                fraud_probability,
                prediction,
                risk_level,
                model_version,
                threshold,
                created_at
            FROM predictions
            WHERE run_id = %s
            ORDER BY fraud_probability DESC
            LIMIT %s
            """,
            (
                run_id,
                limit,
            ),
        )

        rows = cursor.fetchall()

        columns = [
            "id",
            "run_id",
            "provider_id",
            "fraud_probability",
            "prediction",
            "risk_level",
            "model_version",
            "threshold",
            "created_at",
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


# ============================================================
# GET PREDICTION RUNS
# ============================================================

def get_prediction_runs(limit=50):

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                model_version,
                provider_count,
                high_risk_count,
                medium_risk_count,
                low_risk_count,
                status,
                started_at,
                completed_at
            FROM prediction_runs
            ORDER BY started_at DESC
            LIMIT %s
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        columns = [
            "id",
            "model_version",
            "provider_count",
            "high_risk_count",
            "medium_risk_count",
            "low_risk_count",
            "status",
            "started_at",
            "completed_at",
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


# ============================================================
# GET PREDICTION BY RUN + PROVIDER
# ============================================================

def get_prediction_by_run_and_provider(
    run_id,
    provider_id,
):

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                run_id,
                provider_id,
                fraud_probability,
                prediction,
                risk_level,
                model_version,
                threshold,
                created_at
            FROM predictions
            WHERE run_id = %s
              AND provider_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                run_id,
                provider_id,
            ),
        )

        row = cursor.fetchone()

        if not row:
            return None

        columns = [
            "id",
            "run_id",
            "provider_id",
            "fraud_probability",
            "prediction",
            "risk_level",
            "model_version",
            "threshold",
            "created_at",
        ]

        return dict(
            zip(columns, row)
        )

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()