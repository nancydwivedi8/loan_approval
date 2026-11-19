import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.07);
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #172554;
        margin-bottom: 10px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .metric-label {
        color: #64748b;
        font-size: 14px;
    }

    .metric-value {
        color: #172554;
        font-size: 27px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Result */
    .approved {
        background: #ecfdf5;
        border: 1px solid #86efac;
        border-radius: 15px;
        padding: 22px;
        text-align: center;
    }

    .rejected {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 15px;
        padding: 22px;
        text-align: center;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
    }

    .result-subtitle {
        font-size: 16px;
        color: #64748b;
        margin-top: 5px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #e2e8f0;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("loan_approval.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.code(str(e))
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">💰 Loan Approval Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered loan approval prediction using Logistic Regression'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# TOP INFORMATION CARDS
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ML Algorithm</div>
        <div class="metric-value">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Training Dataset</div>
        <div class="metric-value">45,000 Records</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">90%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("""
<div class="card">
<div class="card-title">👤 Applicant Information</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2)

with left:

    credit_score = st.number_input(
        "Credit Score",
        min_value=390,
        max_value=850,
        value=720,
        step=1,
        help="Applicant's credit score"
    )

    loan_int_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=5.42,
        max_value=20.0,
        value=11.5,
        step=0.01,
        help="Interest rate of the requested loan"
    )


with right:

    loan_percent_income = st.number_input(
        "Loan Percent Income",
        min_value=0.0,
        max_value=0.66,
        value=0.15,
        step=0.01,
        help="Loan amount as a proportion of applicant income"
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=2.0,
        max_value=30.0,
        value=4.0,
        step=0.1,
        help="Length of credit history in years"
    )


st.write("")

# =========================================================
# PREDICTION BUTTON
# =========================================================
predict_button = st.button(
    "🔍 Predict Loan Approval"
)


# =========================================================
# PREDICTION
# =========================================================
if predict_button:

    try:

        # -------------------------------------------------
        # Create new applicant
        # Same 4 inputs used in the original notebook
        # -------------------------------------------------
        new_applicant = pd.DataFrame({
            "credit_score": [credit_score],
            "loan_int_rate": [loan_int_rate],
            "loan_percent_income": [loan_percent_income],
            "cb_person_cred_hist_length": [
                cb_person_cred_hist_length
            ]
        })

        # -------------------------------------------------
        # IMPORTANT:
        # Original model was trained after get_dummies()
        # and therefore expects 22 features.
        #
        # Reindex input using the exact feature names
        # stored by the trained model.
        # -------------------------------------------------
        if hasattr(model, "feature_names_in_"):

            new_applicant = new_applicant.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------
        prediction = model.predict(new_applicant)[0]

        probabilities = model.predict_proba(
            new_applicant
        )[0]

        # Classes are [0, 1]
        class_0_index = list(model.classes_).index(0)
        class_1_index = list(model.classes_).index(1)

        rejected_probability = (
            probabilities[class_0_index] * 100
        )

        approved_probability = (
            probabilities[class_1_index] * 100
        )

        st.write("")
        st.markdown("---")
        st.write("")

        # =================================================
        # RESULT
        # =================================================
        if prediction == 1:

            st.markdown("""
            <div class="approved">
                <div class="result-title">
                    ✅ Loan Approved
                </div>
                <div class="result-subtitle">
                    The model predicts that this loan application
                    is likely to be approved.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="rejected">
                <div class="result-title">
                    ❌ Loan Rejected
                </div>
                <div class="result-subtitle">
                    The model predicts that this loan application
                    is likely to be rejected.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # =================================================
        # PROBABILITY METRICS
        # =================================================
        p1, p2 = st.columns(2)

        with p1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    Approval Probability
                </div>
                <div class="metric-value">
                    {approved_probability:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">
                    Rejection Probability
                </div>
                <div class="metric-value">
                    {rejected_probability:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # =================================================
        # PROBABILITY VISUALIZATION
        # =================================================
        st.markdown("""
        <div class="card">
        <div class="card-title">
            📊 Prediction Probability
        </div>
        </div>
        """, unsafe_allow_html=True)

        chart_df = pd.DataFrame({
            "Outcome": [
                "Loan Rejected",
                "Loan Approved"
            ],
            "Probability": [
                rejected_probability,
                approved_probability
            ]
        })

        st.bar_chart(
            chart_df.set_index("Outcome"),
            y="Probability"
        )

        st.caption(
            "Probability represents the model's estimated likelihood "
            "for each outcome."
        )

        # =================================================
        # INPUT SUMMARY
        # =================================================
        st.write("")

        st.markdown("""
        <div class="card">
        <div class="card-title">
            📋 Applicant Input Summary
        </div>
        </div>
        """, unsafe_allow_html=True)

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.write(
                f"**Credit Score:** {credit_score}"
            )
            st.write(
                f"**Loan Interest Rate:** "
                f"{loan_int_rate:.2f}%"
            )

        with summary_col2:
            st.write(
                f"**Loan Percent Income:** "
                f"{loan_percent_income:.2f}"
            )
            st.write(
                f"**Credit History:** "
                f"{cb_person_cred_hist_length:.1f} years"
            )

    except Exception as e:

        st.error("❌ Prediction error")
        st.code(str(e))


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    Loan Approval Prediction | Logistic Regression |
    Machine Learning Project
</div>
""", unsafe_allow_html=True)
