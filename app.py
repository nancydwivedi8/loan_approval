import streamlit as st
import pandas as pd
import joblib

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ==============================
# CUSTOM STYLE
# ==============================

st.markdown("""
<style>

.main {
    padding-top: 20px;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():

    try:
        return joblib.load("loan_approval_model.pkl")
    except:
        return joblib.load("loan_approval.pkl")


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model file not found.")

    st.write(
        "Please check that your .pkl file is uploaded "
        "in the GitHub repository."
    )

    st.stop()


# ==============================
# HEADER
# ==============================

st.markdown(
    '<div class="title">🏦 Loan Approval Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based Loan Approval Classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==============================
# INPUT SECTION
# ==============================

st.subheader("📋 Applicant Information")

st.caption(
    "Enter the important applicant and loan details."
)

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "👤 Age",
        min_value=18,
        max_value=100,
        value=25
    )

    income = st.number_input(
        "💰 Annual Income",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )

    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )


with col2:

    loan_amount = st.number_input(
        "🏦 Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=1000.0
    )

    interest_rate = st.number_input(
        "📈 Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1
    )

    previous_default = st.selectbox(
        "⚠️ Previous Loan Default",
        ["No", "Yes"]
    )


st.divider()


# ==============================
# PREDICTION
# ==============================

if st.button(
    "🔮 Predict Loan Approval",
    type="primary",
    use_container_width=True
):

    # Calculate loan-to-income ratio
    if income > 0:
        loan_percent_income = loan_amount / income
    else:
        loan_percent_income = 0


    # ==================================================
    # CREATE ALL 13 FEATURES EXPECTED BY THE MODEL
    # ==================================================

    input_data = pd.DataFrame({

        "person_age": [age],

        "person_gender": ["male"],

        "person_education": ["Bachelor"],

        "person_income": [income],

        "person_emp_exp": [5],

        "person_home_ownership": ["RENT"],

        "loan_amnt": [loan_amount],

        "loan_intent": ["PERSONAL"],

        "loan_int_rate": [interest_rate],

        "loan_percent_income": [
            loan_percent_income
        ],

        "credit_score": [credit_score],

        "person_cred_hist_length": [5],

        "previous_loan_defaults_on_file": [
            previous_default
        ]
    })


    # ==================================================
    # PREDICT
    # ==================================================

    try:

        prediction = model.predict(input_data)[0]


        # Probability

        try:

            probabilities = model.predict_proba(
                input_data
            )[0]

            approval_probability = (
                probabilities[1] * 100
            )

        except:

            approval_probability = None


        st.divider()

        st.subheader("📊 Prediction Result")


        # ==============================
        # APPROVED
        # ==============================

        if prediction == 1:

            st.success(
                "## ✅ Loan Approved"
            )

            if approval_probability is not None:

                st.metric(
                    "Approval Probability",
                    f"{approval_probability:.2f}%"
                )


        # ==============================
        # NOT APPROVED
        # ==============================

        else:

            st.error(
                "## ❌ Loan Not Approved"
            )

            if approval_probability is not None:

                st.metric(
                    "Approval Probability",
                    f"{approval_probability:.2f}%"
                )


        # ==============================
        # APPLICATION SUMMARY
        # ==============================

        st.subheader("📝 Application Summary")

        summary = pd.DataFrame({

            "Feature": [
                "Age",
                "Annual Income",
                "Credit Score",
                "Loan Amount",
                "Interest Rate",
                "Previous Default"
            ],

            "Value": [
                age,
                f"₹ {income:,.0f}",
                credit_score,
                f"₹ {loan_amount:,.0f}",
                f"{interest_rate:.2f}%",
                previous_default
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.write(e)


# ==============================
# FOOTER
# ==============================

st.divider()

st.caption(
    "Loan Approval Classification | Logistic Regression"
)

st.caption(
    "Educational ML project – not a real banking decision."
)
