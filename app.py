import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from src.preprocessing import (
    convert_date_columns,
    create_features
)

from src.model_input import (
    provider_aggregation
)

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="Healthcare Fraud Analytics Platform",
    page_icon="🏥",
    layout="wide"
)
with st.sidebar:
    st.success("Application deployed successfully")
    st.markdown(
        "[GitHub Repository]"
        "(https://github.com/anupgouda/Healthcare-Fraud-Detection)"
    )

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #0f4c81;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

try:
    pipeline = joblib.load("models/fraud_pipeline.pkl")

    model = pipeline["model"]
    features = pipeline["features"]

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.title("🏥 Healthcare Fraud Analytics Platform")

st.markdown(
    """
### AI-Powered Provider Risk Assessment Dashboard

This platform uses machine learning to identify fraudulent
providers, analyze claim patterns, estimate provider risk,
and support faster investigations.

---
"""
)

# ----------------------------------------------------
# DASHBOARD
# ----------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Features", len(features))
col2.metric("Model", "Random Forest")
col3.metric("Status", "Active")
col4.metric("Version", "1.0")
col5.metric("Algorithm", "RF")

# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------

st.divider()

st.subheader("📁 Upload Healthcare Datasets")

beneficiary_file = st.file_uploader(
    "Beneficiary File",
    type=["csv"]
)

inpatient_file = st.file_uploader(
    "Inpatient File",
    type=["csv"]
)

outpatient_file = st.file_uploader(
    "Outpatient File",
    type=["csv"]
)

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------

if beneficiary_file and inpatient_file and outpatient_file:

    beneficiary = pd.read_csv(beneficiary_file)
    inpatient = pd.read_csv(inpatient_file)
    outpatient = pd.read_csv(outpatient_file)

    beneficiary = convert_date_columns(
        beneficiary,
        ["DOB", "DOD"]
    )

    inpatient = convert_date_columns(
        inpatient,
        [
            "ClaimStartDt",
            "ClaimEndDt",
            "AdmissionDt",
            "DischargeDt"
        ]
    )

    outpatient = convert_date_columns(
        outpatient,
        [
            "ClaimStartDt",
            "ClaimEndDt"
        ]
    )

    claims = pd.concat(
        [inpatient, outpatient],
        ignore_index=True
    )

    claims = claims.merge(
        beneficiary,
        on="BeneID",
        how="left"
    )

    claims = create_features(claims)

    aggregated = provider_aggregation(claims)

    aggregated = aggregated.fillna(0)

    X = aggregated[features]

    with st.spinner("Analyzing provider claims..."):

        prediction = model.predict(X)
        probability = model.predict_proba(X)

    aggregated["Prediction"] = prediction
    aggregated["Fraud Probability"] = probability[:, 1]

    # ------------------------------------------------
    # RISK LEVEL
    # ------------------------------------------------

    def risk_level(score):
        if score < 0.30:
            return "Low"
        elif score < 0.70:
            return "Medium"
        else:
            return "High"

    aggregated["Risk Level"] = aggregated[
        "Fraud Probability"
    ].apply(risk_level)

    st.success("Prediction completed successfully.")

    # ------------------------------------------------
    # METRICS
    # ------------------------------------------------

    high_risk = (aggregated["Risk Level"] == "High").sum()
    medium_risk = (aggregated["Risk Level"] == "Medium").sum()
    low_risk = (aggregated["Risk Level"] == "Low").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("High Risk", high_risk)
    col2.metric("Medium Risk", medium_risk)
    col3.metric("Low Risk", low_risk)

    # ------------------------------------------------
    # TOP PROVIDERS
    # ------------------------------------------------

    st.subheader("Top High-Risk Providers")

    top_providers = aggregated.sort_values(
        by="Fraud Probability",
        ascending=False
    )

    st.dataframe(
        top_providers.head(10),
        use_container_width=True
    )

    # ------------------------------------------------
    # BAR CHART
    # ------------------------------------------------

    st.subheader("⚠️ Top Provider Analysis")

    fig = px.bar(
        top_providers.head(10),
        x="Provider",
        y="Fraud Probability",
        color="Fraud Probability",
        title="Highest-Risk Providers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------
    # HISTOGRAM AND PIE CHART
    # ------------------------------------------------

    st.subheader("📊 Fraud Analysis") 
    left_column, right_column = st.columns(2)

    with left_column:

        fig = px.histogram(
            aggregated,
            x="Fraud Probability",
            nbins=20,
            title="Fraud Probability Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right_column:

        risk_counts = (
            aggregated["Risk Level"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Count"
        ]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            title="Provider Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ------------------------------------------------
    # COMPLETE DATASET
    # ------------------------------------------------

    st.subheader("📋 Prediction Results")

    st.dataframe(
        aggregated.head(20),
        use_container_width=True
    )

    # ------------------------------------------------
    # DOWNLOAD RESULTS
    # ------------------------------------------------

    csv = aggregated.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

# ----------------------------------------------------
# INSIGHTS
# ----------------------------------------------------

st.divider()

st.subheader("💡 Key Insights")

st.info(
    """
    • Monitor abnormal claim patterns.

    • Detect duplicate claims.

    • Identify high-risk providers.

    • Improve payment accuracy.

    • Reduce operational costs.

    • Increase fraud detection efficiency.

    • Implement real-time monitoring.
    """
)

st.markdown("---")

st.caption(
    """
    Developed by Appaji Gouda

    Artificial Intelligence and Machine Learning Engineer
    """
)