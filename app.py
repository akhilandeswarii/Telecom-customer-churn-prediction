import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Analysis", layout="wide")
st.title("📡 Telecom Customer Churn Prediction & Retention Analysis")

# Load clean data and trained models
df = pd.read_csv("telco_churn_clean.csv")
data = joblib.load("models.pkl")
log_model, dt_model = data["log_model"], data["dt_model"]
model_columns = data["columns"]

# --- STEP 6: DISPLAY GRAPHS AND PLOTS ---
st.header("📊 1. Customer Churn Visual Analysis & Model Comparison")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Count by Contract Type")
    contract_churn = pd.crosstab(df["Contract"], df["Churn"])
    st.bar_chart(contract_churn)

with col2:
    st.subheader("Model Accuracy Comparison")
    acc_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree"],
        "Accuracy (%)": [data["log_acc"] * 100, data["dt_acc"] * 100]
    }).set_index("Model")
    st.bar_chart(acc_df)

st.divider()

# --- STEP 7: PREDICT CUSTOMER CHURN FOR NEW INPUT DATA ---
st.header("🔮 2. Predict Churn for a New Customer")

selected_model_name = st.radio(
    "Choose Model for Prediction:",
    ["Logistic Regression", "Decision Tree"],
    horizontal=True
)

c1, c2, c3 = st.columns(3)
with c1:
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 75.0)
with c2:
    total_charges = st.number_input("Total Charges ($)", 18.0, 9000.0, float(tenure * monthly_charges))
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with c3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

if st.button("Predict Customer Churn"):
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment
    }])
    input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)
    
    model = log_model if selected_model_name == "Logistic Regression" else dt_model
    prob = model.predict_proba(input_encoded)[0][1] * 100
    
    if prob > 50:
        st.error(f"⚠️ High Churn Risk ({prob:.1f}%) — Customer is likely to LEAVE.")
        st.info("💡 Retention Strategy: Offer a 15% discount on a 1-Year Contract upgrade.")
    else:
        st.success(f"✅ Low Churn Risk ({prob:.1f}%) — Customer is likely to STAY.")