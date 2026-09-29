from pydantic import BaseModel, Field, model_validator


# ====================================================
# PROVIDER FEATURES
# ====================================================

class ProviderFeatures(BaseModel):

    InscClaimAmtReimbursed_sum: float = Field(ge=0)
    InscClaimAmtReimbursed_mean: float = Field(ge=0)
    InscClaimAmtReimbursed_max: float = Field(ge=0)
    InscClaimAmtReimbursed_min: float = Field(ge=0)

    DeductibleAmtPaid_sum: float = Field(ge=0)
    DeductibleAmtPaid_mean: float = Field(ge=0)

    ClaimID_count: int = Field(ge=0)

    Age_mean: float = Field(ge=0, le=120)
    Age_max: float = Field(ge=0, le=120)

    HospitalStay_mean: float = Field(ge=0)
    HospitalStay_max: float = Field(ge=0)

    ClaimDuration_mean: float = Field(ge=0)
    ClaimDuration_max: float = Field(ge=0)

    ChronicCount_mean: float = Field(ge=0)
    ChronicCount_sum: float = Field(ge=0)

    Dead_sum: int = Field(ge=0)

    # ------------------------------------------------
    # CROSS-FIELD VALIDATION
    # ------------------------------------------------

    @model_validator(mode="after")
    def validate_ranges(self):

        if (
            self.InscClaimAmtReimbursed_min
            > self.InscClaimAmtReimbursed_max
        ):
            raise ValueError(
                "Minimum reimbursement cannot exceed maximum reimbursement"
            )

        if (
            self.InscClaimAmtReimbursed_mean
            > self.InscClaimAmtReimbursed_max
        ):
            raise ValueError(
                "Mean reimbursement cannot exceed maximum reimbursement"
            )

        if (
            self.DeductibleAmtPaid_mean
            > self.DeductibleAmtPaid_sum
        ):
            raise ValueError(
                "Mean deductible cannot exceed total deductible"
            )

        if self.Age_mean > self.Age_max:
            raise ValueError(
                "Mean age cannot exceed maximum age"
            )

        if (
            self.HospitalStay_mean
            > self.HospitalStay_max
        ):
            raise ValueError(
                "Mean hospital stay cannot exceed maximum hospital stay"
            )

        if (
            self.ClaimDuration_mean
            > self.ClaimDuration_max
        ):
            raise ValueError(
                "Mean claim duration cannot exceed maximum claim duration"
            )

        return self


# ====================================================
# PROVIDER PREDICTION REQUEST
# ====================================================

class ProviderPredictionRequest(BaseModel):

    provider_id: str = Field(
        min_length=1,
        max_length=50
    )

    features: ProviderFeatures


# ====================================================
# PREDICTION RESPONSE
# ====================================================

class PredictionResponse(BaseModel):

    provider_id: str

    prediction: int

    fraud_probability: float

    risk_level: str

    model_version: str

    threshold: float