import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("loan_approval.pkl")

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval Prediction")
st.write("Enter the applicant details to predict loan approval.")

st.divider()

# -----------------------------
# Input features
# -----------------------------
credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=720
)

loan_int_rate = st.number_input(
    "Loan Interest Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=11.5,
    step=0.1
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.01
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length",
    min_value=0.0,
    max_value=50.0,
    value=4.0,
    step=0.1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Approval"):

    # Same 4 features used in your Colab prediction code
    new_applicant = pd.DataFrame({
        "credit_score": [credit_score],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length]
    })

    # Match the columns used during model training
    new_applicant = new_applicant.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Prediction
    predicted_class = model.predict(new_applicant)[0]

    # Probability
    predicted_probability = model.predict_proba(new_applicant)[0]

    # Find probability of class 1
    class_1_index = list(model.classes_).index(1)
    approval_probability = predicted_probability[class_1_index] * 100

    st.divider()

    if predicted_class == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(
        f"**Approval Probability: {approval_probability:.2f}%**"
    )
