import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,0.12), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(14,165,233,0.10), transparent 25%),
            #0b1020;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        padding: 2rem 2.2rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.88));
        box-shadow: 0 18px 50px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0;
        color: #f8fafc;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.02rem;
        margin-top: 0.55rem;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 0.32rem 0.75rem;
        border-radius: 999px;
        background: rgba(99,102,241,0.16);
        border: 1px solid rgba(129,140,248,0.28);
        color: #c7d2fe;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    /* Section cards */
    .section-card {
        background: rgba(15,23,42,0.76);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 20px;
        padding: 1.25rem 1.35rem 0.8rem 1.35rem;
        margin-bottom: 1rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.18rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .section-caption {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }

    /* Inputs */
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: rgba(30,41,59,0.82);
        border-radius: 10px;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        border-radius: 13px;
        min-height: 3.2rem;
        font-size: 1.05rem;
        font-weight: 750;
        border: 0;
        color: white;
        background: linear-gradient(90deg, #6366f1, #2563eb);
        box-shadow: 0 10px 25px rgba(37,99,235,0.28);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(37,99,235,0.36);
    }

    /* Result cards */
    .result-card {
        border-radius: 20px;
        padding: 1.45rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(15,23,42,0.88);
    }

    .result-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 850;
        margin-top: 0.25rem;
    }

    .risk-high {
        color: #fb7185;
    }

    .risk-low {
        color: #34d399;
    }

    .probability {
        font-size: 2.65rem;
        font-weight: 850;
        color: #e2e8f0;
    }

    .info-box {
        padding: 0.9rem 1rem;
        border-radius: 13px;
        background: rgba(30,41,59,0.70);
        border: 1px solid rgba(148,163,184,0.12);
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a0f1c;
        border-right: 1px solid rgba(148,163,184,0.12);
    }

    .sidebar-brand {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .sidebar-text {
        color: #94a3b8;
        font-size: 0.87rem;
        line-height: 1.55;
    }

    /* Hide unnecessary Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.code(str(e))
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📊 ChurnAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">'
        'A machine-learning application for predicting customer churn '
        'from customer profile, service, contract, and billing information.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Model")
    st.caption("Logistic Regression pipeline")
    st.caption("Classification task: Customer Churn")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        """
        <div class="sidebar-text">
        1. Enter customer information<br>
        2. Submit the form<br>
        3. The trained model processes the inputs<br>
        4. View the predicted churn status<br>
        5. Review the estimated churn probability
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="badge">MACHINE LEARNING • CUSTOMER RETENTION</div>
    <div class="hero-title">📊 Customer Churn Prediction</div>
    <div class="hero-subtitle">
        Predict whether a customer is likely to churn using demographic,
        service, contract, and billing information.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CUSTOMER INPUTS
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Basic customer and household information.</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])

with c2:
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

with c3:
    partner = st.selectbox("Partner", ["No", "Yes"])

with c4:
    dependents = st.selectbox("Dependents", ["No", "Yes"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SERVICE INFORMATION
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📱 Service Information</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Select the services currently associated with the customer.</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with c2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

with c3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# BILLING INFORMATION
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💳 Billing & Tenure</div>', unsafe_allow_html=True)
st.markdown("<div class='section-caption'>Enter the customer's tenure and billing values.</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

with c2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=500.0,
        value=70.0,
        step=0.01,
        format="%.2f"
    )

with c3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=0.01,
        format="%.2f"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🔮 Predict Customer Churn")

if predict:
    # Keep the exact feature names/order used during model training.
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        prediction = model.predict(input_data)[0]

        # Convert model output safely to Yes/No.
        prediction_text = str(prediction)

        # Probability, if supported by the saved classifier.
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if "Yes" in classes:
                probability = float(probabilities[classes.index("Yes")])
            elif 1 in classes:
                probability = float(probabilities[classes.index(1)])

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------
        if prediction_text.lower() in ["yes", "1", "true"]:
            result_class = "risk-high"
            result_icon = "⚠️"
            result_message = "The model predicts that this customer is likely to churn."
        else:
            result_class = "risk-low"
            result_icon = "✅"
            result_message = "The model predicts that this customer is likely to stay."

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Prediction Result</div>
                <div class="result-value {result_class}">
                    {result_icon} {prediction_text}
                </div>
                <div style="color:#cbd5e1; margin-top:0.35rem;">
                    {result_message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if probability is not None:
            churn_percentage = probability * 100
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated Churn Probability</div>
                    <div class="probability">{churn_percentage:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(probability)

            if probability >= 0.70:
                st.warning(
                    "Higher estimated churn probability. Consider reviewing the customer's "
                    "contract, service usage, and billing experience."
                )
            elif probability >= 0.40:
                st.info(
                    "Moderate estimated churn probability. The customer may benefit from "
                    "closer retention monitoring."
                )
            else:
                st.success(
                    "Lower estimated churn probability based on the information provided."
                )

        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------
        with st.expander("🔎 View submitted customer information"):
            st.dataframe(
                input_data.T.rename(columns={0: "Value"}),
                use_container_width=True
            )

    except Exception as e:
        st.error("Prediction failed. Please verify the input values and model compatibility.")
        st.code(str(e))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.78rem;">
        Customer Churn Prediction • Machine Learning Application
    </div>
    """,
    unsafe_allow_html=True
)
