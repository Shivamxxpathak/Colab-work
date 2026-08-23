"""Streamlit UI for customer churn risk prediction."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_churn_model.joblib"
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Estimate churn risk from customer profile information.")
if not MODEL_PATH.exists():
    st.warning("Train the model first with: python scripts/train.py")
    st.stop()
model = joblib.load(MODEL_PATH)
with st.form("prediction_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=840.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    submitted = st.form_submit_button("Predict Churn Risk")
if submitted:
    row = {
        "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": internet,
        "OnlineSecurity": "No internet service" if internet == "No" else "No",
        "OnlineBackup": "No internet service" if internet == "No" else "No",
        "DeviceProtection": "No internet service" if internet == "No" else "No",
        "TechSupport": "No internet service" if internet == "No" else "No",
        "StreamingTV": "No internet service" if internet == "No" else "No",
        "StreamingMovies": "No internet service" if internet == "No" else "No",
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    probability = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    if probability >= 0.70:
        risk, action = "HIGH", "Prioritize this customer for proactive retention outreach."
    elif probability >= 0.40:
        risk, action = "MEDIUM", "Consider a targeted retention offer and service check-in."
    else:
        risk, action = "LOW", "Continue standard engagement and monitor behavior."
    st.metric("Churn Probability", f"{probability:.1%}")
    st.subheader(f"Risk Level: {risk}")
    st.info(action)
