import os
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠"
)

# Title
st.title("🏠 Sydney Housing Price Predictor")

st.write(
    "Enter broad property characteristics to estimate the sale price "
    "using the SIT720 8.1D Random Forest model."
)

# Model file
MODEL_PATH = "sydney_housing_rf_pipeline.joblib"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Put "
        "sydney_housing_rf_pipeline.joblib in the same folder as this app."
    )
    st.stop()

# Load trained model
model = joblib.load(MODEL_PATH)

# User inputs
town = st.selectbox(
    "Suburb",
    ["Blacktown", "Castle Hill", "Coogee"]
)

property_type = st.selectbox(
    "Property Type",
    ["House", "Apartment", "Unit", "Townhouse"]
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0,
    max_value=10,
    value=2
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=10,
    value=2
)

land_area_m2 = st.number_input(
    "Land / Area (m²)",
    min_value=20.0,
    max_value=10000.0,
    value=500.0,
    step=10.0
)

sold_year = st.number_input(
    "Sale Year",
    min_value=2000,
    max_value=2035,
    value=2025
)

sold_month = st.number_input(
    "Sale Month",
    min_value=1,
    max_value=12,
    value=6
)

# Prediction
if st.button("Predict Sale Price"):

    input_data = pd.DataFrame([{
        "town": town,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "land_area_m2": land_area_m2,
        "sold_year": sold_year,
        "sold_month": sold_month
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated sale price: ${prediction:,.0f} AUD"
    )

    st.caption(
        "Educational ML estimate only; not a professional property valuation."
    )