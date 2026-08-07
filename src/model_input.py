import pandas as pd


def provider_aggregation(claims):

    aggregated = claims.groupby(
        "Provider"
    ).agg({

        "InscClaimAmtReimbursed": [
            "sum",
            "mean",
            "max",
            "min"
        ],

        "DeductibleAmtPaid": [
            "sum",
            "mean"
        ],

        "ClaimID": "count",

        "Age": [
            "mean",
            "max"
        ],

        "HospitalStay": [
            "mean",
            "max"
        ],

        "ClaimDuration": [
            "mean",
            "max"
        ],

        "ChronicCount": [
            "mean",
            "sum"
        ],

        "Dead": "sum"
    })

    aggregated.columns = [

        "_".join(column)
        for column in aggregated.columns
    ]

    aggregated.reset_index(
        inplace=True
    )

    return aggregated