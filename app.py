import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a CSV file or enter transaction details manually.")

# File uploader
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Uploaded Data:")
    st.write(df.head())

    if st.button("Predict Fraud"):
        predictions = model.predict(df)
        df["Prediction"] = predictions
        df["Prediction"] = df["Prediction"].map({0: "Not Fraud", 1: "Fraud"})
        st.write("Prediction Results:")
        st.write(df[["Prediction"]])
else:
    st.subheader("Or Enter Transaction Details Manually")
    v_features = {}
    for i in range(1, 29):  # V1 to V28
        v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
    amount = st.number_input("Amount", value=0.0)
    time = st.number_input("Time", value=0.0)

    input_data = pd.DataFrame([{**v_features, "Amount": amount, "Time": time}])

    if st.button("Predict"):
        result = model.predict(input_data)[0]
        st.success("✅ Not Fraud" if result == 0 else "⚠️ Fraud Detected!")
