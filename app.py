import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("loan_approval.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Model file 'loan_approval.pkl' could not be loaded.")
    st.write(e)
    st.stop()


# Title
st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval.")

st.divider()

# --------------------------------------------------
# INPUT FEATURES
# --------------------------------------------------

st.subheader("📋 Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    person_gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    person_education = st.selectbox(
        "Education",
        [
            "High School",
            "Associate",
            "Bachelor",
            "Master",
            "Doctorate"
        ]
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

with col2:

    person_emp_exp = st.number_input(
        "Employment Experience (Years)",
        min_value=0,
        max_value=60,
        value=5
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER"
        ]
    )

    person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

with col3:

    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Default",
        [
            "No",
            "Yes"
        ]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1
    )

loan_percent_income = st.number_input(
    "Loan Percent of Income",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Loan Approval", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_gender": [person_gender],
        "person_education": [person_education],
        "person_income": [person_income],
        "person_emp_exp": [person_emp_exp],
        "person_home_ownership": [person_home_ownership],
        "person_cred_hist_length": [person_cred_hist_length],
        "credit_score": [credit_score],
        "previous_loan_defaults_on_file": [
            previous_loan_defaults_on_file
        ],
        "loan_amnt": [loan_amnt],
        "loan_intent": [loan_intent],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income]
    })

    # Convert categorical variables into dummy variables
    input_data = pd.get_dummies(
        input_data,
        drop

