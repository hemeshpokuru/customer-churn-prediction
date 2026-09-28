# ============================================================
# CUSTOMER CHURN PREDICTION - STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(37, 99, 235, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(79, 70, 229, 0.10),
                transparent 30%
            ),
            #080d1a;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #111827 0%,
                #0b1220 100%
            );

        border-right: 1px solid #1e293b;
    }

    .sidebar-brand {
        padding: 0.4rem 0 1.3rem 0;
    }

    .sidebar-brand h1 {
        color: #f8fafc;
        font-size: 1.35rem;
        margin: 0;
        font-weight: 800;
    }

    .sidebar-brand p {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    .sidebar-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 1rem;
        margin: 0.8rem 0;
    }

    .sidebar-card-title {
        color: #f8fafc;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.45rem;
    }

    .sidebar-card-text {
        color: #94a3b8;
        font-size: 0.78rem;
        line-height: 1.55;
    }

    .model-badge {
        display: inline-block;
        background: rgba(37, 99, 235, 0.18);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 999px;
        padding: 0.45rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        position: relative;

        padding: 2.3rem 2.5rem;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                #172554 0%,
                #1e3a8a 50%,
                #312e81 100%
            );

        border: 1px solid rgba(147, 197, 253, 0.15);

        box-shadow:
            0 20px 45px rgba(0, 0, 0, 0.28);

        overflow: hidden;

        margin-bottom: 1.5rem;
    }

    .hero::after {
        content: "";

        position: absolute;

        width: 230px;
        height: 230px;

        right: -80px;
        top: -100px;

        background: rgba(255,255,255,0.06);

        border-radius: 50%;
    }

    .hero-badge {
        display: inline-block;

        color: #bfdbfe;

        background: rgba(255,255,255,0.08);

        border: 1px solid rgba(255,255,255,0.13);

        border-radius: 999px;

        padding: 0.4rem 0.75rem;

        font-size: 0.7rem;

        font-weight: 800;

        letter-spacing: 0.05em;

        margin-bottom: 0.9rem;
    }

    .hero h1 {
        color: #ffffff;

        font-size: 2.55rem;

        font-weight: 850;

        margin: 0;

        line-height: 1.15;
    }

    .hero p {
        color: #dbeafe;

        font-size: 0.98rem;

        line-height: 1.6;

        margin-top: 0.8rem;

        max-width: 800px;
    }


    /* ======================================================
       STEP INDICATOR
       ====================================================== */

    .step-container {
        display: flex;

        align-items: center;

        justify-content: space-between;

        margin: 1.5rem 0 1.2rem 0;

        padding: 0.9rem 1rem;

        border-radius: 14px;

        background: rgba(15, 23, 42, 0.75);

        border: 1px solid #1e293b;
    }

    .step {
        display: flex;

        align-items: center;

        gap: 0.5rem;

        color: #94a3b8;

        font-size: 0.76rem;

        font-weight: 650;
    }

    .step-number {
        width: 27px;
        height: 27px;

        display: flex;

        align-items: center;
        justify-content: center;

        border-radius: 50%;

        background: #1e293b;

        border: 1px solid #334155;

        color: #94a3b8;

        font-size: 0.72rem;

        font-weight: 800;
    }

    .step-active {
        color: #dbeafe;
    }

    .step-active .step-number {
        background: #2563eb;
        border-color: #3b82f6;
        color: white;
    }


    /* ======================================================
       SECTION HEADER
       ====================================================== */

    .section-header {
        margin-top: 1.5rem;

        margin-bottom: 1rem;

        padding: 1rem 1.2rem;

        border-radius: 15px;

        background:
            linear-gradient(
                135deg,
                #111c35,
                #16233e
            );

        border: 1px solid #263754;

        box-shadow:
            0 7px 18px rgba(0, 0, 0, 0.12);
    }

    .section-header h3 {
        color: #f8fafc;

        margin: 0;

        font-size: 1.08rem;

        font-weight: 800;
    }

    .section-header p {
        color: #94a3b8;

        margin: 0.3rem 0 0 0;

        font-size: 0.77rem;
    }


    /* ======================================================
       INPUT LABELS
       ====================================================== */

    label {
        color: #cbd5e1 !important;

        font-weight: 600 !important;

        font-size: 0.78rem !important;
    }


    /* ======================================================
       SELECT BOXES
       ====================================================== */

    div[data-baseweb="select"] > div {

        background-color: #151d2d !important;

        border: 1px solid #29374d !important;

        border-radius: 9px !important;

        min-height: 42px;
    }

    div[data-baseweb="select"] > div:hover {

        border-color: #3b82f6 !important;

        box-shadow:
            0 0 0 1px rgba(59,130,246,0.15);
    }


    /* ======================================================
       NUMBER INPUT
       ====================================================== */

    div[data-testid="stNumberInput"] input {

        background-color: #151d2d !important;

        color: #f8fafc !important;

        border: 1px solid #29374d !important;

        border-radius: 9px !important;
    }


    /* ======================================================
       HELP TEXT
       ====================================================== */

    .input-hint {
        color: #64748b;

        font-size: 0.69rem;

        margin-top: -0.35rem;

        margin-bottom: 0.4rem;
    }


    /* ======================================================
       PREDICT BUTTON
       ====================================================== */

    .stButton > button {

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #4f46e5
            );

        color: white;

        border: none;

        border-radius: 11px;

        min-height: 48px;

        font-weight: 750;

        font-size: 0.9rem;

        box-shadow:
            0 10px 24px rgba(37, 99, 235, 0.25);

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    .stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 14px 30px rgba(37, 99, 235, 0.35);

        border: none;
    }


    /* ======================================================
       RESET BUTTON
       ====================================================== */

    .reset-button button {

        background: #172033 !important;

        color: #94a3b8 !important;

        border: 1px solid #29374d !important;

        box-shadow: none !important;
    }


    /* ======================================================
       RESULT
       ====================================================== */

    .result-wrapper {

        margin-top: 2rem;

        padding: 1.5rem;

        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                #101a30,
                #111c35
            );

        border: 1px solid #263754;

        box-shadow:
            0 15px 35px rgba(0,0,0,0.22);
    }

    .result-label {

        color: #64748b;

        font-size: 0.68rem;

        font-weight: 800;

        letter-spacing: 0.12em;

        text-transform: uppercase;
    }

    .result-title {

        color: #f8fafc;

        font-size: 1.65rem;

        font-weight: 850;

        margin-top: 0.35rem;
    }

    .probability-number {

        font-size: 3rem;

        font-weight: 900;

        margin-top: 0.3rem;
    }

    .result-description {

        color: #94a3b8;

        font-size: 0.84rem;

        line-height: 1.55;
    }


    /* ======================================================
       RISK BADGES
       ====================================================== */

    .risk-badge {

        display: inline-block;

        padding: 0.45rem 0.8rem;

        border-radius: 999px;

        font-size: 0.72rem;

        font-weight: 800;
    }

    .risk-low {

        color: #6ee7b7;

        background: rgba(16,185,129,0.12);

        border: 1px solid rgba(16,185,129,0.25);
    }

    .risk-medium {

        color: #fcd34d;

        background: rgba(245,158,11,0.12);

        border: 1px solid rgba(245,158,11,0.25);
    }

    .risk-high {

        color: #fda4af;

        background: rgba(244,63,94,0.12);

        border: 1px solid rgba(244,63,94,0.25);
    }


    /* ======================================================
       RESULT METRICS
       ====================================================== */

    .metric-card {

        background: #0c1425;

        border: 1px solid #22304a;

        border-radius: 13px;

        padding: 1rem;

        min-height: 100px;
    }

    .metric-label {

        color: #64748b;

        font-size: 0.67rem;

        text-transform: uppercase;

        letter-spacing: 0.08em;

        font-weight: 800;
    }

    .metric-value {

        color: #f8fafc;

        font-size: 1.35rem;

        font-weight: 850;

        margin-top: 0.35rem;
    }


    /* ======================================================
       EXPANDER
       ====================================================== */

    [data-testid="stExpander"] {

        background: #0e1728;

        border: 1px solid #22304a;

        border-radius: 12px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        text-align: center;

        color: #475569;

        margin-top: 3rem;

        padding-top: 1.5rem;

        border-top: 1px solid #1e293b;

        font-size: 0.72rem;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 768px) {

        .hero {
            padding: 1.6rem;
        }

        .hero h1 {
            font-size: 2rem;
        }

        .step {
            font-size: 0.65rem;
        }

        .step-number {
            width: 23px;
            height: 23px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = Path("customer_churn_model.pkl")


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error("❌ Unable to load the trained model.")

    st.code(str(e))

    st.info(
        "Make sure 'customer_churn_model.pkl' is in the same "
        "GitHub repository as app.py and that requirements.txt "
        "uses scikit-learn==1.6.1."
    )

    st.stop()


# ============================================================
# EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:

    st.session_state.prediction_done = False


if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "probability" not in st.session_state:

    st.session_state.probability = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <h1>📊 ChurnAI</h1>

            <p>
            Customer retention intelligence powered by
            machine learning.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                🤖 Final Model
            </div>

            <div class="sidebar-card-text">

                Your trained customer churn pipeline
                using a tuned and class-balanced
                Random Forest model.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                ⚙️ How it works
            </div>

            <div class="sidebar-card-text">

                <b>01</b> Enter customer information.<br><br>

                <b>02</b> Submit the customer profile.<br><br>

                <b>03</b> The trained pipeline processes
                the information.<br><br>

                <b>04</b> The model predicts churn probability.<br><br>

                <b>05</b> Review the estimated customer risk.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                🎯 Prediction Target
            </div>

            <div class="sidebar-card-text">

                <span class="model-badge">
                    CUSTOMER CHURN
                </span>

                <br><br>

                The application estimates the probability
                that a customer belongs to the churn class.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption(
        "Machine Learning Customer Churn Prediction"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            MACHINE LEARNING • CUSTOMER RETENTION
        </div>

        <h1>
            📊 Customer Churn Prediction
        </h1>

        <p>
            Estimate whether a telecom customer is likely to churn
            using their demographic, service, contract and billing
            information.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STEP INDICATOR
# ============================================================

st.markdown(
    """
    <div class="step-container">

        <div class="step step-active">
            <div class="step-number">1</div>
            Customer Profile
        </div>

        <div class="step">
            <div class="step-number">2</div>
            Services
        </div>

        <div class="step">
            <div class="step-number">3</div>
            Billing
        </div>

        <div class="step">
            <div class="step-number">4</div>
            Prediction
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <h3>👤 Customer Profile</h3>

        <p>
        Basic demographic and household information.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
        help="Customer gender."
    )


with col2:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Whether the customer is a senior citizen."
    )


with col3:

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
        help="Whether the customer has a partner."
    )


with col4:

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
        help="Whether the customer has dependents."
    )


# ============================================================
# SERVICE INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <h3>📱 Service Information</h3>

        <p>
        Select the telecom and additional services associated
        with this customer.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
        help="Whether the customer has a phone service."
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"],
        help="Whether the customer has multiple phone lines."
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        help="Type of internet service."
    )


with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"],
        help="Whether online security service is active."
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"],
        help="Whether online backup service is active."
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"],
        help="Whether device protection is active."
    )


with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"],
        help="Whether technical support is active."
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"],
        help="Whether the customer uses streaming TV."
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"],
        help="Whether the customer uses streaming movie services."
    )


# ============================================================
# CONTRACT & BILLING
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <h3>💳 Contract & Billing</h3>

        <p>
        Enter customer tenure, contract and billing information.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1,
        help="Number of months the customer has stayed with the company."
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=500.0,
        value=70.0,
        step=1.0,
        help="Customer's current monthly charges."
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=10.0,
        help="Total amount charged to the customer."
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ],
        help="Current customer contract type."
    )


with col3:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
        help="Whether paperless billing is enabled."
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        help="Customer's payment method."
    )


# ============================================================
# QUICK PROFILE SUMMARY
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-header">

        <h3>📝 Profile Summary</h3>

        <p>
        Review the important customer attributes before prediction.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)


with summary_col1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Contract
            </div>

            <div class="metric-value">
                {contract}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with summary_col2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Tenure
            </div>

            <div class="metric-value">
                {tenure} months
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with summary_col3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Monthly Charges
            </div>

            <div class="metric-value">
                ${monthly_charges:,.2f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with summary_col4:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Internet
            </div>

            <div class="metric-value">
                {internet_service}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ACTION BUTTONS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)


button_col1, button_col2, button_col3 = st.columns([1, 2, 1])


with button_col2:

    predict_button = st.button(
        "🔮  Predict Customer Churn",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame(
    {
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    }
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    with st.spinner("Analyzing customer profile..."):

        try:

            # ------------------------------------------------
            # Validate input columns
            # ------------------------------------------------

            missing_features = [
                feature
                for feature in EXPECTED_FEATURES
                if feature not in input_data.columns
            ]

            if missing_features:

                st.error(
                    f"Missing model features: {missing_features}"
                )

                st.stop()


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(input_data)[0]


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                classes = list(model.classes_)

                # Handle Yes / No target
                if "Yes" in classes:

                    positive_index = classes.index("Yes")

                # Handle numeric encoded target
                elif 1 in classes:

                    positive_index = classes.index(1)

                else:

                    positive_index = len(classes) - 1

                churn_probability = float(
                    probabilities[positive_index]
                )

            else:

                churn_probability = float(
                    prediction
                )


            # ------------------------------------------------
            # Convert prediction into readable label
            # ------------------------------------------------

            if prediction in ["Yes", 1, True]:

                churn_label = "Yes"

            else:

                churn_label = "No"


            # Save result
            st.session_state.prediction_done = True

            st.session_state.prediction = churn_label

            st.session_state.probability = churn_probability


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


# ============================================================
# DISPLAY RESULT
# ============================================================

if st.session_state.prediction_done:

    prediction = st.session_state.prediction

    churn_probability = st.session_state.probability


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    if churn_probability < 0.30:

        risk_level = "Low Risk"

        risk_class = "risk-low"

        risk_icon = "🟢"

        risk_message = (
            "The model estimates a relatively low probability "
            "of customer churn."
        )

    elif churn_probability < 0.60:

        risk_level = "Medium Risk"

        risk_class = "risk-medium"

        risk_icon = "🟠"

        risk_message = (
            "The model estimates a moderate probability "
            "of customer churn."
        )

    else:

        risk_level = "High Risk"

        risk_class = "risk-high"

        risk_icon = "🔴"

        risk_message = (
            "The model estimates a relatively high probability "
            "of customer churn."
        )


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.markdown(
        """
        <div class="section-header">

            <h3>🎯 Prediction Result</h3>

            <p>
            Machine-learning based customer churn assessment.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # MAIN RESULT
    # ========================================================

    if prediction == "Yes":

        result_title = "⚠️ Customer Likely to Churn"

        result_color = "#fb7185"

        result_description = (
            "The model classified this customer in the churn class."
        )

    else:

        result_title = "✅ Customer Likely to Stay"

        result_color = "#34d399"

        result_description = (
            "The model classified this customer in the non-churn class."
        )


    st.markdown(
        f"""
        <div class="result-wrapper">

            <div class="result-label">
                MODEL PREDICTION
            </div>

            <div
                class="result-title"
                style="color:{result_color};"
            >
                {result_title}
            </div>

            <div
                class="probability-number"
                style="color:{result_color};"
            >
                {churn_probability:.2%}
            </div>

            <div class="result-description">

                Estimated probability of churn.

                <br>

                {result_description}

            </div>

            <br>

            <span class="risk-badge {risk_class}">
                {risk_icon} {risk_level}
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown("### 📈 Churn Probability")

    st.progress(
        min(
            max(churn_probability, 0.0),
            1.0
        )
    )

    probability_col1, probability_col2, probability_col3 = st.columns(3)


    with probability_col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Churn Probability
                </div>

                <div class="metric-value">
                    {churn_probability:.2%}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with probability_col2:

        stay_probability = 1 - churn_probability

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Estimated Stay Probability
                </div>

                <div class="metric-value">
                    {stay_probability:.2%}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with probability_col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Risk Level
                </div>

                <div class="metric-value">
                    {risk_level}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # BUSINESS INTERPRETATION
    # ========================================================

    st.markdown("### 💡 Interpretation")

    if risk_level == "High Risk":

        st.error(
            "This customer has a comparatively high estimated "
            "churn probability. The profile may warrant closer "
            "customer-retention attention."
        )

    elif risk_level == "Medium Risk":

        st.warning(
            "This customer falls into a moderate estimated "
            "churn-risk range. The profile may benefit from "
            "additional review."
        )

    else:

        st.success(
            "This customer has a comparatively low estimated "
            "churn probability based on the information provided."
        )


    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    with st.expander(
        "🔎 View Submitted Customer Information"
    ):

        display_data = input_data.T.reset_index()

        display_data.columns = [
            "Feature",
            "Value"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <div>
            Customer Churn Prediction • Machine Learning Application
        </div>

        <div style="margin-top:0.4rem;">
            Tuned & Class-Balanced Random Forest
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
