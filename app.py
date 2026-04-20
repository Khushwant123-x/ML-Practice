import streamlit as st
import pandas as pd
import joblib
import os

# ---------- LOAD MODEL ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# ---------- UI ----------
st.set_page_config(page_title="Loan Predictor", layout="centered")

st.title("💳 Loan Approval Predictor")
st.markdown("### 📌 Please enter values based on realistic financial ranges")

# ---------- GUIDELINES ----------
with st.expander("📊 Data Guidelines (IMPORTANT)"):
    st.markdown("""
    **Use values close to dataset range:**

    - 👤 Applicant Income: 2000 – 20000  
    - 👥 Coapplicant Income: 0 – 10000  
    - 🎂 Age: 21 – 59  
    - 👨‍👩‍👧 Dependents: 0 – 3  
    - 📈 Credit Score: 550 – 799  
    - 💳 Existing Loans: 0 – 4  
    - 📉 DTI Ratio: 0.1 – 0.6  
    - 💰 Savings: 0 – 20000  
    - 🏠 Collateral Value: 1000 – 50000  
    - 💵 Loan Amount: 1000 – 40000  
    - ⏳ Loan Term: 12 – 84 months  

    ⚠️ Out-of-range values may give incorrect predictions.
    """)

st.markdown("---")

# ---------- INPUTS ----------
Applicant_Income = st.number_input(
    "Applicant Income",
    min_value=2000, max_value=20000, value=10000,
    help="Monthly income of applicant"
)

Coapplicant_Income = st.number_input(
    "Coapplicant Income",
    min_value=0, max_value=10000, value=2000,
    help="Income of co-applicant (if any)"
)

Age = st.number_input(
    "Age",
    min_value=21, max_value=59, value=30
)

Dependents = st.number_input(
    "Dependents",
    min_value=0, max_value=3, value=1
)

Credit_Score = st.number_input(
    "Credit Score",
    min_value=550, max_value=799, value=650,
    help="Higher score = better chances"
)

Existing_Loans = st.number_input(
    "Existing Loans",
    min_value=0, max_value=4, value=1
)

DTI_Ratio = st.number_input(
    "DTI Ratio",
    min_value=0.1, max_value=0.6, value=0.3
)

Savings = st.number_input(
    "Savings",
    min_value=0, max_value=20000, value=5000
)

Collateral_Value = st.number_input(
    "Collateral Value",
    min_value=1000, max_value=50000, value=20000
)

Loan_Amount = st.number_input(
    "Loan Amount",
    min_value=1000, max_value=40000, value=15000
)

Loan_Term = st.number_input(
    "Loan Term (months)",
    min_value=12, max_value=84, value=36
)

# ---------- PREDICT ----------
if st.button("🚀 Predict Loan Status"):

    if Applicant_Income == 0:
        st.error("❌ Income cannot be zero")
    else:

        data = {
            "Applicant_Income": Applicant_Income,
            "Coapplicant_Income": Coapplicant_Income,
            "Age": Age,
            "Dependents": Dependents,
            "Credit_Score": Credit_Score,
            "Existing_Loans": Existing_Loans,
            "DTI_Ratio": DTI_Ratio,
            "Savings": Savings,
            "Collateral_Value": Collateral_Value,
            "Loan_Amount": Loan_Amount,
            "Loan_Term": Loan_Term
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        df = scaler.transform(df)

        pred = model.predict(df)[0]

        st.markdown("---")

        if pred == 1:
            st.success("✅ Loan Approved")
            st.info("Based on financial profile, applicant is eligible")
        else:
            st.error("❌ Loan Rejected")
            st.warning("Try improving credit score or reducing loan amount")