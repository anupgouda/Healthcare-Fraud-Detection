import psycopg2


# ----------------------------------------------------
# DATABASE CONFIGURATION
# ----------------------------------------------------

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "healthcare_fraud"
DB_USER = "anupgouda"


# ----------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------

def get_connection():

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER
    )

    return connection


# ----------------------------------------------------
# TEST CONNECTION
# ----------------------------------------------------

def test_connection():

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            "SELECT current_database();"
        )

        database_name = cursor.fetchone()[0]

        cursor.close()

        return {
            "success": True,
            "database": database_name
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        if connection:
            connection.close()