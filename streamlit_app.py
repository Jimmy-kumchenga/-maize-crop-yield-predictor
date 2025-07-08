import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Malawi Maize Yield Predictor", layout="centered")

st.title("🌽 Malawi Maize Yield Predictor")
st.markdown("Provide basic farm details to estimate **yield (kg/ha)**.")


MODEL_PATH = Path("improved_rf_yield_model.pkl")
if not MODEL_PATH.exists():
    st.error("❌ Model file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)


col1, col2 = st.columns(2)

with col1:
    year            = st.selectbox("Season Year", [2023, 2024, 2025], index=1)
    region          = st.selectbox("Region", ["North", "Central", "South"])
    maize_type      = st.selectbox("Maize Type", ["Local", "Hybrid", "OPV"])
    soil_quality    = st.selectbox("Soil Quality", ["Poor", "Average", "Good", "Excellent"])
    fertilizer_type = st.selectbox("Fertilizer Type", ["Organic", "Inorganic", "Mixed"])
    irrigated       = st.checkbox("Irrigated Field")

with col2:
    crop_rotation   = st.checkbox("Crop Rotation Practised")
    farmer_exp      = st.slider("Farmer Experience (years)", 1, 30, 5)
    area_ha         = st.number_input("Area Cultivated (ha)", 0.5, 50.0, 2.0, step=0.1)

    # Rainfall and Temp Level Options
    rain_level = st.selectbox("Rainfall Level", ["Low", "Moderate", "High"], index=1)
    temp_level = st.selectbox("Temperature Level", ["Cool", "Moderate", "Warm"], index=1)

    # Convert to numeric for model input
    rainfall_mm = {"Low": 700, "Moderate": 1100, "High": 1450}[rain_level]
    temp_c = {"Cool": 23.0, "Moderate": 25.0, "Warm": 27.0}[temp_level]

    fertilizer_kg = st.number_input("Fertilizer Used (kg/ha)", 20.0, 300.0, 100.0, step=1.0)


if st.button("Predict Yield"):
    input_data = {
        "Year"             : [year],
        "Maize_Type"       : [maize_type],
        "Region"           : [region],
        "Soil_Quality"     : [soil_quality],
        "Fertilizer_Type"  : [fertilizer_type],
        "Irrigated"        : [int(irrigated)],
        "Crop_Rotation"    : [int(crop_rotation)],
        "Farmer_Experience": [farmer_exp],
        "Area_ha"          : [area_ha],
        "Rainfall_mm"      : [rainfall_mm],
        "Avg_Temp_C"       : [temp_c],
        "Fertilizer_kg_ha" : [fertilizer_kg],
    }
    df_input = pd.DataFrame(input_data)

    # Predict
    prediction = model.predict(df_input)[0]
    st.success(f"🌾 Estimated Maize Yield: **{prediction:,.0f} kg/ha**")

    with st.expander("View Input Summary"):
        st.write(df_input)
