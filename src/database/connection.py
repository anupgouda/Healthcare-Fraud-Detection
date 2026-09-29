import os

import psycopg2


# ----------------------------------------------------
# DATABASE CONFIGURATION
# ----------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "healthcare_fraud")
DB_USER = os.getenv("DB_USER", "anupgouda")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ----------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------

def get_connection():

    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


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