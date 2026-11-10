
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="AI Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)

with open("machine_failure_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("⚙️ AI-Based Industrial Machine Failure Prediction")
st.write("Machine Health Monitoring using Machine Learning")

st.sidebar.header("Machine Sensor Inputs")

temperature = st.sidebar.slider(
    "Temperature (°C)", 40.0, 120.0, 70.0
)

vibration = st.sidebar.slider(
    "Vibration (g)", 0.1, 5.0, 1.0
)

rpm = st.sidebar.slider(
    "Motor Speed (RPM)", 500, 3000, 1500
)

pressure = st.sidebar.slider(
    "Pressure (bar)", 1.0, 10.0, 5.0
)

if st.button("Predict Machine Condition"):
    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Vibration": vibration,
        "RPM": rpm,
        "Pressure": pressure
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ MACHINE FAILURE RISK DETECTED")
    else:
        st.success("✅ MACHINE CONDITION: NORMAL")

    st.metric(
        "Predicted Failure Probability",
        f"{probability * 100:.2f}%"
    )

    st.subheader("Sensor Readings")
    st.dataframe(input_data, use_container_width=True)

st.caption("AI prediction based on a synthetic training dataset.")
