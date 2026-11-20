import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    # Your PPT says the saved model is:
    # loan_approval_model.pkl

    try:
        return joblib.load("loan_approval_model.pkl")
    except:
        return joblib.load("loan_approval.pkl")


try:
    model = load_model()

except Exception as e:
    st.error("❌ Model file could not be loaded.")
    st.write("Make sure the .pkl model file is uploaded in GitHub.")
    st.write(e)
    st.stop()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏦 Loan Approval Prediction")

st.write(
    "Enter the applicant details below to predict "
    "whether the loan is likely to be approved or not."
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("📋 Applicant Information")

col1, col2, col3 = st.columns(3)


# --------------------------------------------------
# COLUMN 1
# --------------------------------------------------

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


# --------------------------------------------------
# COLUMN 2
# --------------------------------------------------

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


# --------------------------------------------------
# COLUMN 3
# --------------------------------------------------

with col3:

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

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

    person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0,
        max_value=50,
        value=5
    )


# --------------------------------------------------
# PREVIOUS DEFAULT
# --------------------------------------------------

previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Default",
    ["No", "Yes"]
)


st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🔮 Predict Loan Approval",
    type="primary",
    use_container_width=True
):

    # Create input dataframe
    input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_gender": [person_gender],
        "person_education": [person_education],
        "person_income": [person_income],
        "person_emp_exp": [person_emp_exp],
        "person_home_ownership": [person_home_ownership],
        "loan_amnt": [loan_amnt],
        "loan_intent": [loan_intent],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "credit_score": [credit_score],
        "person_cred_hist_length": [
            person_cred_hist_length
        ],
        "previous_loan_defaults_on_file": [
            previous_loan_defaults_on_file
        ]
    })


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        # Probability
        try:
            probabilities = model.predict_proba(
                input_data
            )[0]

            approval_probability = probabilities[1] * 100
            rejection_probability = probabilities[0] * 100

        except:
            approval_probability = None
            rejection_probability = None


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.subheader("📢 Prediction Result")

        if prediction == 1:

            st.success("## ✅ Loan Approved")

            if approval_probability is not None:

                st.metric(
                    "Approval Probability",
                    f"{approval_probability:.2f}%"
                )

        else:

            st.error("## ❌ Loan Not Approved")

            if rejection_probability is not None:

                st.metric(
                    "Rejection Probability",
                    f"{rejection_probability:.2f}%"
                )


        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        if approval_probability is not None:

            st.subheader("📊 Prediction Probability")

            probability_data = pd.DataFrame({
                "Result": [
                    "Approved",
                    "Not Approved"
                ],
                "Probability (%)": [
                    approval_probability,
                    rejection_probability
                ]
            })

            st.bar_chart(
                probability_data.set_index("Result")
            )


        # --------------------------------------------------
        # SHOW INPUT
        # --------------------------------------------------

        st.subheader("📝 Entered Applicant Details")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.write(
            "The model and input features may not match."
        )

        st.write("Error details:", e)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Loan Approval Classification | Logistic Regression"
)

st.caption(
    "⚠️ This is an educational Machine Learning project "
    "and not a real banking decision system."
)
 

