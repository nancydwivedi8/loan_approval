```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Your notebook saved the model as loan_approval.pkl
    # If your GitHub file has another name, change it here.
    return joblib.load("loan_approval.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Model file could not be loaded.")
    st.error("Make sure 'loan_approval.pkl' is uploaded in the same GitHub repository.")
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🏦 Loan Approval Prediction")

st.markdown(
    """
    ### Machine Learning Project
    **Loan Approval Classification using Logistic Regression**

    Enter the applicant's details below and the model will predict
    whether the loan is likely to be **Approved** or **Not Approved**.
    """
)

st.divider()


# =========================================================
# INPUT FORM
# =========================================================

st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:

    person_age = st.number_input(
        "👤 Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1
    )

    person_gender = st.selectbox(
        "⚥ Gender",
        ["male", "female"]
    )

    person_education = st.selectbox(
        "🎓 Education",
        [
            "High School",
            "Associate",
            "Bachelor",
            "Master",
            "Doctorate"
        ]
    )

    person_income = st.number_input(
        "💰 Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    person_emp_exp = st.number_input(
        "💼 Employment Experience (Years)",
        min_value=0,
        max_value=60,
        value=5,
        step=1
    )


with col2:

    person_home_ownership = st.selectbox(
        "🏠 Home Ownership",
        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER"
        ]
    )

    person_cred_hist_length = st.number_input(
        "📅 Credit History Length (Years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    credit_score = st.number_input(
        "📊 Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )

    previous_loan_defaults_on_file = st.selectbox(
        "⚠️ Previous Loan Default",
        [
            "No",
            "Yes"
        ]
    )


with col3:

    loan_amnt = st.number_input(
        "💵 Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    loan_intent = st.selectbox(
        "🎯 Loan Intent",
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
        "📈 Loan Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1
    )

    loan_percent_income = st.number_input(
        "📌 Loan Percent of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):

    # -----------------------------------------------------
    # Create raw input DataFrame
    # -----------------------------------------------------

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
            "Yes" if previous_loan_defaults_on_file == "Yes" else "No"
        ],
        "loan_amnt": [loan_amnt],
        "loan_intent": [loan_intent],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income]
    })


    # -----------------------------------------------------
    # Convert categorical variables into dummy variables
    #
    # This matches your notebook:
    #
    # x = pd.get_dummies(x, drop_first=True)
    # -----------------------------------------------------

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # -----------------------------------------------------
    # Get the feature names used during model training
    # -----------------------------------------------------

    try:

        expected_features = model.named_steps[
            "scaler"
        ].feature_names_in_

        # Make sure the Streamlit input has exactly
        # the same columns as training data.

        input_encoded = input_encoded.reindex(
            columns=expected_features,
            fill_value=0
        )

    except Exception:

        st.warning(
            "⚠️ Feature names could not be automatically detected. "
            "Please check the model and training code."
        )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(input_encoded)[0]


    # -----------------------------------------------------
    # Prediction Probability
    # -----------------------------------------------------

    try:

        probability = model.predict_proba(
            input_encoded
        )[0]

        approved_probability = probability[1] * 100
        rejected_probability = probability[0] * 100

    except Exception:

        approved_probability = None
        rejected_probability = None


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.subheader("📢 Prediction Result")

    if prediction == 1:

        st.success("### ✅ Loan Approved")

        if approved_probability is not None:

            st.metric(
                "Approval Probability",
                f"{approved_probability:.2f}%"
            )

    else:

        st.error("### ❌ Loan Not Approved")

        if rejected_probability is not None:

            st.metric(
                "Rejection Probability",
                f"{rejected_probability:.2f}%"
            )


    # -----------------------------------------------------
    # Show both probabilities
    # -----------------------------------------------------

    if approved_probability is not None:

        st.subheader("📊 Prediction Probability")

        probability_df = pd.DataFrame({
            "Result": [
                "Approved",
                "Not Approved"
            ],
            "Probability": [
                approved_probability,
                rejected_probability
            ]
        })

        st.bar_chart(
            probability_df.set_index("Result")
        )


    # -----------------------------------------------------
    # Show input summary
    # -----------------------------------------------------

    st.subheader("📝 Applicant Details")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Loan Approval Classification | Logistic Regression | "
    "Educational Machine Learning Project"
)

st.caption(
    "⚠️ This application is for educational demonstration "
    "and should not be used as a real banking decision system."
)
```
   





