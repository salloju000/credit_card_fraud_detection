import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.markdown("<h1 style='text-align: center;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Detect fraudulent credit card transactions using machine learning.</p>", unsafe_allow_html=True)

st.header("📁 Upload CSV File")
file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🚀 Predict Fraud on Uploaded Data"):
        with st.spinner("Running predictions..."):
            predictions = model.predict(df)
            df["Prediction"] = predictions
            df["Prediction"] = df["Prediction"].map({0: "🟢 Not Fraud", 1: "🔴 Fraud"})
        st.success("✅ Prediction complete!")
        st.toast("✅ Predictions done on uploaded data!", icon="✅")
        st.write("### Prediction Results:")
        st.dataframe(df[["Prediction"]], use_container_width=True)

st.markdown("---")
st.markdown("<h2 style='text-align: center;'>✍️ Manual Transaction Entry</h2>", unsafe_allow_html=True)

if st.button("🎲 Generate Random Values for Manual Entry"):
    for i in range(1, 29):
        st.session_state[f"V{i}"] = np.round(np.random.normal(), 2)
    st.session_state["Amount"] = np.round(np.random.uniform(0, 2500), 2)
    st.session_state["Time"] = np.round(np.random.uniform(0, 172800), 2)
    st.toast(" Random values generated!", icon="🎲")

selected_feature = st.selectbox("Select a feature to enter value", [f"V{i}" for i in range(1, 29)])

default_val = st.session_state.get(selected_feature, 0.0)
val = st.number_input(f"Enter value for {selected_feature}", value=default_val, step=0.01, key=f"input_{selected_feature}")
st.session_state[selected_feature] = val

amount = st.number_input("Amount", value=st.session_state.get("Amount", 0.0), step=0.01, key="Amount")
time = st.number_input("Time", value=st.session_state.get("Time", 0.0), step=0.01, key="Time")

v_features = {}
for i in range(1, 29):
    key = f"V{i}"
    v_features[key] = st.session_state.get(key, 0.0)

input_data = pd.DataFrame([{**v_features, "Amount": amount, "Time": time}])

if st.button("🔍 Predict This Transaction"):
    result = model.predict(input_data)[0]
    if result == 0:
        st.balloons()
        st.toast(" This transaction is Not Fraudulent", icon="✅")
        st.success(" This transaction is **Not Fraudulent**.")
    else:
        st.toast("⚠️ Fraudulent transaction detected!", icon="⚠️")
        st.error("⚠️ This transaction is **Fraudulent**.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>| Powered by $$R |</p>", unsafe_allow_html=True)
