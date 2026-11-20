import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .app-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        background: #161b22;
        border: 1px solid #30363d;
        margin-bottom: 15px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background: #161b22;
        border: 1px solid #30363d;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    try:
        return joblib.load("loan_approval_model.pkl")
    except:
        return joblib.load("loan_approval.pkl")


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model file could not be loaded.")

    st.info(
        "Please make sure your .pkl model file is uploaded "
        "in the GitHub repository."
    )

    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="app-title">🏦 Loan Approval Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered loan approval prediction using Logistic Regression'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# IMPORTANT FEATURES
# =====================================================

st.markdown("### 📋 Applicant Information")

st.caption(
    "Enter the key applicant and loan details to get a prediction."
)

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "👤 Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1
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
        value=650,
        step=1
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


# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button(
    "🔮  Predict Loan Approval",
    type="primary",
    use_container_width=True
):

    # -------------------------------------------------
    # DERIVED VALUE
    # -------------------------------------------------

    loan_percent_income = (
        loan_amount / income
        if income > 0
        else 0
    )


    # -------------------------------------------------
    # DEFAULT VALUES FOR OTHER MODEL FEATURES
    # -------------------------------------------------
    #
    # The trained model expects 13 features.
    # We are showing only the most important ones
    # in the UI and using reasonable default values
    # for the remaining features.
    #

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


    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]


        # Probability

        try:

            probabilities = model.predict_proba(
                input_data
            )[0]

            rejection_probability = (
                probabilities[0] * 100
            )

            approval_probability = (
                probabilities[1] * 100
            )

        except:

            approval_probability = None
            rejection_probability = None


        st.divider()

        st.markdown("### 📊 Prediction Result")


        # -------------------------------------------------
        # APPROVED
        # -------------------------------------------------

        if prediction == 1:

            st.success(
                "## ✅ Loan Approved"
            )

            if approval_probability is not None:

                st.metric(
                    "Approval Probability",
                    f"{approval_probability:.2f}%"
                )


        # -------------------------------------------------
        # NOT APPROVED
        # -------------------------------------------------

        else:

            st.error(
                "## ❌ Loan Not Approved"
            )

            if rejection_probability is not None:

                st.metric(
                    "Rejection Probability",
                    f"{rejection_probability:.2f}%"
                )


        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        if approval_probability is not None:

            st.markdown("#### Prediction Confidence")

            probability_df = pd.DataFrame(
                {
                    "Probability (%)": [
                        approval_probability,
                        rejection_probability
                    ]
                },
                index=[
                    "Approved",
                    "Not Approved"
                ]
            )

            st.bar_chart(
                probability_df
            )


        # -------------------------------------------------
        # SHOW IMPORTANT INPUTS
        # -------------------------------------------------

        st.markdown("#### 📝 Application Summary")

        summary = pd.DataFrame(
            {
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
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "❌ Prediction could not be completed."
        )

        st.write(
            "Model and input features are not matching."
        )

        st.write(e)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Loan Approval Classification • Logistic Regression"
)

st.caption(
    "⚠️ Educational demonstration only. "
    "Not a real banking decision system."
)    
       
