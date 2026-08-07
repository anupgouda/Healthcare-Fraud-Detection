import pandas as pd


def convert_date_columns(df, columns):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    return df


def create_features(claims):

    # -------------------------------------------------
    # AGE
    # -------------------------------------------------

    if (
        "ClaimStartDt" in claims.columns
        and "DOB" in claims.columns
    ):

        claims["Age"] = (
            claims["ClaimStartDt"].dt.year
            - claims["DOB"].dt.year
        )

    # -------------------------------------------------
    # CLAIM DURATION
    # -------------------------------------------------

    if (
        "ClaimStartDt" in claims.columns
        and "ClaimEndDt" in claims.columns
    ):

        claims["ClaimDuration"] = (
            claims["ClaimEndDt"]
            - claims["ClaimStartDt"]
        ).dt.days

    # -------------------------------------------------
    # HOSPITAL STAY
    # -------------------------------------------------

    if (
        "AdmissionDt" in claims.columns
        and "DischargeDt" in claims.columns
    ):

        claims["HospitalStay"] = (
            claims["DischargeDt"]
            - claims["AdmissionDt"]
        ).dt.days

    # -------------------------------------------------
    # DECEASED PATIENTS
    # -------------------------------------------------

    if "DOD" in claims.columns:

        claims["Dead"] = (
            claims["DOD"]
            .notnull()
            .astype(int)
        )

    # -------------------------------------------------
    # CHRONIC CONDITIONS
    # -------------------------------------------------

    chronic_columns = [

        "ChronicCond_Alzheimer",
        "ChronicCond_Heartfailure",
        "ChronicCond_KidneyDisease",
        "ChronicCond_Cancer",
        "ChronicCond_ObstrPulmonary",
        "ChronicCond_Depression",
        "ChronicCond_Diabetes",
        "ChronicCond_IschemicHeart",
        "ChronicCond_Osteoporasis",
        "ChronicCond_rheumatoidarthritis",
        "ChronicCond_stroke"
    ]

    existing_columns = [

        column
        for column in chronic_columns
        if column in claims.columns
    ]

    if len(existing_columns) > 0:

        claims["ChronicCount"] = (
            claims[existing_columns] == 1
        ).sum(axis=1)

    return claims