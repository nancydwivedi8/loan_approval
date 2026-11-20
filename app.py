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

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 25px;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


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

    st.write(e)

    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title">🏦 Loan Approval Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Loan Approval Classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# INPUTS
# =====================================================

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


# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button(
    "🔮 Predict Loan Approval",
    type="primary"
):

    # -------------------------------------------------
    # LOAN TO INCOME RATIO
    # -------------------------------------------------

    if income > 0:
        loan_percent_income = loan_amount / income
    else:
        loan_percent_income = 0


    # -------------------------------------------------
    # CREATE RAW INPUT
    # -------------------------------------------------

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

        "cb_person_cred_hist_length": [5],

        "credit_score": [credit_score],

        "cb_person_default_on_file": [
            "Y" if previous_default == "Yes" else "N"
        ]
    })


    # =================================================
    # CONVERT CATEGORICAL FEATURES TO DUMMY VARIABLES
    # =================================================

    categorical_columns = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file"
    ]


    input_data = pd.get_dummies(
        input_data,
        columns=categorical_columns
    )


    # =================================================
    # MATCH EXACT TRAINING FEATURES
    # =================================================

    try:

        expected_features = list(
            model.feature_names_in_
        )

        input_data = input_data.reindex(
            columns=expected_features,
            fill_value=0
        )

    except AttributeError:

        pass


    # =================================================
    # PREDICTION
    # =================================================

    try:

        prediction = model.predict(
            input_data
        )[0]


        # ------------------------------------------------
        # PROBABILITY
        # ------------------------------------------------

        try:

            probabilities = model.predict_proba(
                input_data
            )[0]

            approval_probability = (
                probabilities[1] * 100
            )

            rejection_probability = (
                probabilities[0] * 100
            )

        except:

            approval_probability = None
            rejection_probability = None


        # =================================================
        # RESULT
        # =================================================

        st.subheader("📊 Prediction Result")


        if prediction == 1:

            st.success(
                "## ✅ Loan Approved"
            )

            if approval_probability is not None:

                st.metric(
                    "Approval Probability",
                    f"{approval_probability:.2f}%"
                )

        else:

            st.error(
                "## ❌ Loan Not Approved"
            )

            if rejection_probability is not None:

                st.metric(
                    "Rejection Probability",
                    f"{rejection_probability:.2f}%"
                )


        # =================================================
        # APPLICATION SUMMARY
        # =================================================

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

        st.write(
            "Model and input features could not be matched."
        )

        st.write(e)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Loan Approval Classification | Logistic Regression"
)

st.caption(
    "Educational Machine Learning Project"
)
