import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
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

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(67, 56, 202, 0.16), transparent 28%),
            radial-gradient(circle at 15% 20%, rgba(37, 99, 235, 0.10), transparent 25%),
            #080d1c;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0b1122;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-brand {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.35rem;
    }

    .sidebar-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    .sidebar-divider {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 1.3rem 0;
    }

    .model-card {
        background: linear-gradient(
            135deg,
            rgba(37,99,235,0.25),
            rgba(79,70,229,0.18)
        );
        border: 1px solid rgba(96,165,250,0.22);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .model-card-title {
        color: #60a5fa;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .model-card-text {
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .metric-box {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.7rem;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 0.2rem;
    }

    .workflow-item {
        color: #cbd5e1;
        font-size: 0.84rem;
        margin: 0.55rem 0;
    }

    /* ---------- HERO ---------- */

    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(
                135deg,
                rgba(30,64,175,0.90),
                rgba(49,46,129,0.92)
            );
        border: 1px solid rgba(147,197,253,0.20);
        border-radius: 22px;
        padding: 2.25rem 2.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        border-radius: 50%;
        background: rgba(255,255,255,0.07);
        right: -90px;
        top: -110px;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.15);
        color: #dbeafe;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 850;
        color: #ffffff;
        margin: 0;
        line-height: 1.1;
    }

    .hero-text {
        color: #dbeafe;
        font-size: 1rem;
        margin-top: 0.8rem;
        max-width: 850px;
        line-height: 1.6;
    }

    /* ---------- SECTION HEADERS ---------- */

    .section-header {
        background: linear-gradient(
            135deg,
            rgba(22,38,75,0.95),
            rgba(16,28,57,0.95)
        );
        border: 1px solid rgba(96,165,250,0.12);
        border-radius: 15px;
        padding: 1rem 1.25rem;
        margin-top: 1.3rem;
        margin-bottom: 0.8rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 750;
        margin: 0;
    }

    .section-description {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }

    /* ---------- INPUTS ---------- */

    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #171d2e !important;
        border-color: rgba(148,163,184,0.15) !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #171d2e !important;
        border-color: rgba(148,163,184,0.15) !important;
        border-radius: 10px !important;
    }

    input {
        color: #f8fafc !important;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 750;
        font-size: 0.95rem;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        box-shadow: 0 8px 25px rgba(37,99,235,0.25);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(37,99,235,0.35);
    }

    /* ---------- SUMMARY CARDS ---------- */

    .summary-card {
        background: #11182b;
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        min-height: 90px;
    }

    .summary-label {
        color: #94a3b8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .summary-value {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }

    /* ---------- RESULT ---------- */

    .result-card {
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1.4rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .result-safe {
        background: linear-gradient(
            135deg,
            rgba(6,78,59,0.40),
            rgba(15,23,42,0.96)
        );
    }

    .result-risk {
        background: linear-gradient(
            135deg,
            rgba(127,29,29,0.42),
            rgba(15,23,42,0.96)
        );
    }

    .result-label {
        color: #94a3b8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 700;
    }

    .result-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 850;
        margin-top: 0.35rem;
    }

    .result-description {
        color: #cbd5e1;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    .probability-number {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 850;
        line-height: 1;
        margin: 0.7rem 0;
    }

    .risk-low {
        color: #34d399;
    }

    .risk-medium {
        color: #fbbf24;
    }

    .risk-high {
        color: #fb7185;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.3rem;
        border-top: 1px solid rgba(255,255,255,0.07);
    }

    /* ---------- HIDE STREAMLIT BRANDING ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">📊 Churn Predictor</div>
        <div class="sidebar-subtitle">
        An interactive machine-learning application for
        predicting telecom customer churn.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🤖 Model")

    st.markdown(
        """
        <div class="model-card">
            <div class="model-card-title">Final Model</div>
            <div class="model-card-text">
                Tuned + Class-Balanced<br>
                <b>Random Forest Classifier</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📈 Model Performance")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">76.51%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col2:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">62.85%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    metric_col3, metric_col4 = st.columns(2)

    with metric_col3:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">Recall</div>
                <div class="metric-value">74.87%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric_col4:
        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">ROC-AUC</div>
                <div class="metric-value">84.10%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🔄 How It Works")

    workflow = [
        "1. Enter customer information",
        "2. Review the customer profile",
        "3. Submit the prediction",
        "4. Model processes the inputs",
        "5. Review churn probability"
    ]

    for item in workflow:
        st.markdown(
            f'<div class="workflow-item">{item}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.caption(
        "Built with Python • Scikit-learn • Random Forest • Streamlit"
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

        <div class="hero-title">
            📊 Customer Churn Prediction
        </div>

        <div class="hero-text">
            Estimate whether a telecom customer is likely to churn
            using demographic, service, contract and billing information.
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
        <div class="section-title">👤 Customer Profile</div>
        <div class="section-description">
            Basic demographic and household information.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

with col4:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

# ============================================================
# SERVICE INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">📡 Service Information</div>
        <div class="section-description">
            Select the telecom and additional services associated with this customer.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

# ============================================================
# CONTRACT AND BILLING
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">💳 Contract & Billing</div>
        <div class="section-description">
            Enter customer tenure, contract and billing information.
        </div>
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
        step=1
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0,
        format="%.2f"
    )

with col3:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=5.0,
        format="%.2f"
    )

with col2:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ============================================================
# CUSTOMER SUMMARY
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">🔎 Customer Summary</div>
        <div class="section-description">
            Review the main attributes before running the prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Contract</div>
            <div class="summary-value">{contract}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary2:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Tenure</div>
            <div class="summary-value">{tenure} months</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary3:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Monthly Charges</div>
            <div class="summary-value">₹{monthly_charges:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary4:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Internet</div>
            <div class="summary-value">{internet_service}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_col, reset_col, empty_col = st.columns([5, 2, 3])

with predict_col:
    predict_button = st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )

with reset_col:
    reset_button = st.button(
        "↻ Reset",
        use_container_width=True
    )

if reset_button:
    st.rerun()

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create dataframe
    # IMPORTANT:
    # Column names must match the training dataset.
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
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
    })

    try:

        with st.spinner("Analyzing customer profile..."):

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        # ----------------------------------------------------
        # Convert prediction
        # ----------------------------------------------------

        prediction_text = str(prediction)

        # Handle either Yes/No or 1/0 model outputs
        if prediction_text.lower() in ["yes", "1", "true"]:
            churn_prediction = "Yes"
        else:
            churn_prediction = "No"

        churn_probability = float(probability)

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if churn_probability < 0.30:
            risk_level = "Low Risk"
            risk_class = "risk-low"
            risk_message = (
                "The estimated churn probability is relatively low "
                "based on the information provided."
            )

        elif churn_probability < 0.60:
            risk_level = "Moderate Risk"
            risk_class = "risk-medium"
            risk_message = (
                "The customer shows a moderate estimated likelihood "
                "of churn and may benefit from retention attention."
            )

        else:
            risk_level = "High Risk"
            risk_class = "risk-high"
            risk_message = (
                "The customer has a relatively high estimated churn "
                "probability and may require retention attention."
            )

        # ----------------------------------------------------
        # Result card
        # ----------------------------------------------------

        if churn_prediction == "Yes":

            st.markdown(
                f"""
                <div class="result-card result-risk">

                    <div class="result-label">
                        PREDICTION RESULT
                    </div>

                    <div class="result-title">
                        ⚠️ Customer Likely to Churn
                    </div>

                    <div class="result-description">
                        The trained model predicts that this customer
                        is likely to churn based on the information provided.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-card result-safe">

                    <div class="result-label">
                        PREDICTION RESULT
                    </div>

                    <div class="result-title">
                        ✅ Customer Likely to Stay
                    </div>

                    <div class="result-description">
                        The trained model predicts that this customer
                        is unlikely to churn based on the information provided.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    ESTIMATED CHURN PROBABILITY
                </div>

                <div class="probability-number {risk_class}">
                    {churn_probability * 100:.2f}%
                </div>

                <div class="{risk_class}"
                     style="font-weight:750; margin-bottom:0.6rem;">
                    {risk_level}
                </div>

                <div class="result-description">
                    {risk_message}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        st.progress(
            min(max(churn_probability, 0.0), 1.0),
            text=f"Churn probability: {churn_probability * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Business interpretation
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">
                    💡 Prediction Interpretation
                </div>
                <div class="section-description">
                    A simple business-oriented interpretation of the result.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if churn_prediction == "Yes":

            st.warning(
                "⚠️ This customer has been classified as likely to churn. "
                "The probability represents the model's estimated likelihood "
                "based on the supplied customer attributes."
            )

            st.info(
                "Potential business action: consider reviewing the customer's "
                "contract, service usage, billing experience and available "
                "retention options."
            )

        else:

            st.success(
                "✅ This customer has been classified as unlikely to churn "
                "based on the supplied information."
            )

            st.info(
                "Potential business action: continue normal customer "
                "engagement and monitor future changes in customer behavior."
            )

        # ----------------------------------------------------
        # Submitted information
        # ----------------------------------------------------

        with st.expander("🔍 View submitted customer information"):

            display_data = pd.DataFrame({
                "Attribute": [
                    "Gender",
                    "Senior Citizen",
                    "Partner",
                    "Dependents",
                    "Tenure",
                    "Phone Service",
                    "Multiple Lines",
                    "Internet Service",
                    "Online Security",
                    "Online Backup",
                    "Device Protection",
                    "Tech Support",
                    "Streaming TV",
                    "Streaming Movies",
                    "Contract",
                    "Paperless Billing",
                    "Payment Method",
                    "Monthly Charges",
                    "Total Charges"
                ],
                "Value": [
                    gender,
                    senior_citizen,
                    partner,
                    dependents,
                    f"{tenure} months",
                    phone_service,
                    multiple_lines,
                    internet_service,
                    online_security,
                    online_backup,
                    device_protection,
                    tech_support,
                    streaming_tv,
                    streaming_movies,
                    contract,
                    paperless_billing,
                    payment_method,
                    f"{monthly_charges:.2f}",
                    f"{total_charges:.2f}"
                ]
            })

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.error(
            "The prediction could not be generated. "
            "Please check that the input columns match the model's training data."
        )

        st.code(str(e))

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer Churn Prediction • Machine Learning Application<br>
        Tuned & Class-Balanced Random Forest • Built with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
