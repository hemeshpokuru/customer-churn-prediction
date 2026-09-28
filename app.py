import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
    margin-bottom: 1.5rem;
}

.hero h1 {
    font-size: 2.6rem;
    margin-bottom: 0.4rem;
}

.hero p {
    font-size: 1.05rem;
    opacity: 0.9;
}

.section {
    background: white;
    padding: 1.4rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border: 1px solid #e5e7eb;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.result-card {
    padding: 1.8rem;
    border-radius: 18px;
    text-align: center;
    margin-top: 1.5rem;
}

.result-title {
    font-size: 2rem;
    font-weight: 800;
}

.probability {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

.info-card {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.small-text {
    color: #64748b;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path("customer_churn_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.info(
        "Make sure customer_churn_model.pkl is in the same GitHub repository "
        "as app.py and that requirements.txt uses scikit-learn==1.6.1."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Churn Predictor")

    st.markdown("---")

    st.markdown("### About the Model")

    st.write(
        "This application uses the trained Customer Churn Machine Learning "
        "pipeline to predict whether a customer is likely to churn."
    )

    st.markdown("### Final Model")

    st.info(
        "Tuned + Class-Balanced Random Forest"
    )

    st.markdown("### Model Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", "76.51%")
        st.metric("Recall", "74.87%")

    with col2:
        st.metric("F1 Score", "62.85%")
        st.metric("ROC-AUC", "84.10%")

    st.markdown("---")

    st.markdown(
        "<div class='small-text'>"
        "Model trained using the Telco Customer Churn dataset."
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>📊 Customer Churn Prediction</h1>

<p>
Predict whether a telecom customer is likely to churn using
customer demographics, services, contract information and billing details.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

st.markdown("""
<div class="section">
<div class="section-title">👤 Customer Profile</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col4:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


# ============================================================
# SERVICE INFORMATION
# ============================================================

st.markdown("""
<div class="section">
<div class="section-title">📱 Service Information</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# CONTRACT AND BILLING
# ============================================================

st.markdown("""
<div class="section">
<div class="section-title">💳 Contract & Billing</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=10.0
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col3:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
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


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:

    predict_button = st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({
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
    })

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        # Find probability associated with Yes
        classes = list(model.classes_)

        if "Yes" in classes:
            churn_probability = probabilities[classes.index("Yes")]
        else:
            # Fallback for encoded target
            churn_probability = probabilities[-1]

        churn_probability = float(churn_probability)

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == "Yes":

            st.markdown(
                f"""
                <div class="result-card"
                     style="background:#fff1f2;border:1px solid #fecdd3;">

                    <div class="result-title" style="color:#be123c;">
                        ⚠️ Customer Likely to Churn
                    </div>

                    <div class="probability" style="color:#be123c;">
                        {churn_probability:.2%}
                    </div>

                    <p>
                        Estimated probability of customer churn
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.warning(
                "This customer shows characteristics associated with "
                "a higher probability of churn."
            )

        else:

            st.markdown(
                f"""
                <div class="result-card"
                     style="background:#ecfdf5;border:1px solid #a7f3d0;">

                    <div class="result-title" style="color:#047857;">
                        ✅ Customer Likely to Stay
                    </div>

                    <div class="probability" style="color:#047857;">
                        {churn_probability:.2%}
                    </div>

                    <p>
                        Estimated probability of customer churn
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "This customer shows characteristics associated with "
                "a lower probability of churn."
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.markdown("### 📈 Churn Probability")

        st.progress(
            min(max(churn_probability, 0.0), 1.0)
        )

        st.write(
            f"**Churn probability:** {churn_probability:.2%}"
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if churn_probability >= 0.70:

            risk = "High Risk"

        elif churn_probability >= 0.40:

            risk = "Medium Risk"

        else:

            risk = "Low Risk"

        st.markdown("### 🎯 Risk Classification")

        if risk == "High Risk":

            st.error(f"🔴 {risk}")

        elif risk == "Medium Risk":

            st.warning(f"🟠 {risk}")

        else:

            st.success(f"🟢 {risk}")


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander("🔎 View Customer Information"):

            display_data = input_data.T.reset_index()

            display_data.columns = ["Feature", "Value"]

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
        <p>
        Customer Churn Prediction • Machine Learning Project
        </p>
        <p>
        Tuned & Class-Balanced Random Forest
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
