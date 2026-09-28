import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(37, 99, 235, 0.13),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(99, 102, 241, 0.12),
                transparent 28%
            ),
            #07101f;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #091222;
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .sidebar-description {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.55;
    }


    /* ---------- HERO ---------- */

    .hero-box {
        padding: 2.2rem;
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.15);

        background:
            linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.82),
                rgba(15, 23, 42, 0.92)
            );

        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.25);

        margin-bottom: 1.4rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;

        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.15);

        color: #dbeafe;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 850;
        color: white;
        margin-top: 0.7rem;
        letter-spacing: -1px;
    }

    .hero-description {
        color: #dbeafe;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 850px;
        margin-top: 0.5rem;
    }


    /* ---------- CARDS ---------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(148, 163, 184, 0.14) !important;
        border-radius: 18px !important;
        background: rgba(15, 23, 42, 0.52);
    }


    /* ---------- METRICS ---------- */

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 0.85rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }


    /* ---------- INPUTS ---------- */

    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: rgba(15, 23, 42, 0.85);
        border-radius: 10px;
        border-color: rgba(148, 163, 184, 0.18);
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 11px;
        font-weight: 750;
        min-height: 3rem;
    }


    /* ---------- RESULT ---------- */

    .risk-number {
        font-size: 3rem;
        font-weight: 850;
        letter-spacing: -1px;
        margin-top: 0.2rem;
    }

    .risk-high {
        color: #fb7185;
    }

    .risk-low {
        color: #34d399;
    }


    /* ---------- INFO ---------- */

    .small-text {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.55;
    }


    /* ---------- FOOTER ---------- */

    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "customer_churn_model.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Trained model file not found."
    )

    st.info(
        "Make sure customer_churn_model.pkl is in the same GitHub "
        "repository folder as app.py."
    )

    st.stop()


try:

    model = load_model()

except Exception as error:

    st.error(
        "❌ The trained model could not be loaded."
    )

    st.code(str(error))

    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def is_churn_prediction(prediction):

    value = str(prediction).strip().lower()

    return value in {
        "yes",
        "1",
        "true",
        "churn",
        "churned"
    }


def get_churn_probability(model, input_data):

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_data)[0]

    classes = list(
        getattr(model, "classes_", [])
    )

    positive_classes = [
        "Yes",
        "yes",
        "Churn",
        "churn",
        1,
        True
    ]

    for positive_class in positive_classes:

        if positive_class in classes:

            index = classes.index(
                positive_class
            )

            return float(
                probabilities[index]
            )

    # Binary fallback
    if len(probabilities) == 2:

        return float(
            probabilities[1]
        )

    return None


def get_feature_importance(model):

    try:

        classifier = model

        # If the model is a Pipeline
        if hasattr(model, "named_steps"):

            for name, step in model.named_steps.items():

                if hasattr(
                    step,
                    "feature_importances_"
                ):

                    classifier = step
                    break

        if not hasattr(
            classifier,
            "feature_importances_"
        ):

            return None

        importance_values = np.asarray(
            classifier.feature_importances_,
            dtype=float
        )

        feature_names = []

        # Extract feature names from preprocessing pipeline
        if hasattr(model, "named_steps"):

            preprocessor = None

            for name, step in model.named_steps.items():

                if (
                    "preprocess" in name.lower()
                    or "transform" in name.lower()
                ):

                    preprocessor = step
                    break

            if (
                preprocessor is not None
                and hasattr(
                    preprocessor,
                    "get_feature_names_out"
                )
            ):

                feature_names = list(
                    preprocessor.get_feature_names_out()
                )

        if len(feature_names) != len(
            importance_values
        ):

            feature_names = [
                f"Feature {i + 1}"
                for i in range(
                    len(importance_values)
                )
            ]

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(12)
            .reset_index(drop=True)
        )

        return importance_df

    except Exception:

        return None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">'
        '📊 Churn Predictor'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'An interactive machine-learning application '
        'for telecom customer churn prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🎯 Final Model")

    st.info(
        "Tuned + Class-Balanced Random Forest"
    )

    st.markdown("### 📈 Model Performance")

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Accuracy",
        "76.51%"
    )

    metric_col2.metric(
        "F1 Score",
        "62.85%"
    )

    metric_col3, metric_col4 = st.columns(2)

    metric_col3.metric(
        "Recall",
        "74.87%"
    )

    metric_col4.metric(
        "ROC-AUC",
        "84.10%"
    )

    st.divider()

    st.markdown("### ⚙️ How It Works")

    st.markdown(
        """
        **1.** Enter customer information

        **2.** Submit the prediction form

        **3.** The trained pipeline processes the data

        **4.** View churn classification

        **5.** Review estimated churn probability
        """
    )

    st.divider()

    st.caption(
        "Scikit-learn pipeline"
    )

    st.caption(
        "Prediction is model-based decision support."
    )


# =========================================================
# HERO
# =========================================================




# =========================================================
# TABS
# =========================================================

prediction_tab, model_tab, business_tab = st.tabs(
    [
        "🔮 Predict Churn",
        "📊 Model Analysis",
        "💡 Business Insights"
    ]
)


# =========================================================
# PREDICTION TAB
# =========================================================

with prediction_tab:

    st.subheader(
        "👤 Customer Profile"
    )

    st.caption(
        "Enter the customer's demographic and household information."
    )

    with st.form(
        "customer_churn_form"
    ):

        # ---------------------------------------------
        # CUSTOMER PROFILE
        # ---------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            gender = st.selectbox(
                "Gender",
                [
                    "Female",
                    "Male"
                ]
            )

        with col2:

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [
                    "No",
                    "Yes"
                ]
            )

        with col3:

            partner = st.selectbox(
                "Partner",
                [
                    "No",
                    "Yes"
                ]
            )

        with col4:

            dependents = st.selectbox(
                "Dependents",
                [
                    "No",
                    "Yes"
                ]
            )


        st.divider()


        # ---------------------------------------------
        # SERVICE INFORMATION
        # ---------------------------------------------

        st.subheader(
            "📱 Service Information"
        )

        st.caption(
            "Select the services currently associated with the customer."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            phone_service = st.selectbox(
                "Phone Service",
                [
                    "No",
                    "Yes"
                ]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service"
                ]
            )

            internet_service = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            online_security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )


        with col2:

            device_protection = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )


        with col3:

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                [
                    "No",
                    "Yes"
                ]
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


        st.divider()


        # ---------------------------------------------
        # BILLING
        # ---------------------------------------------

        st.subheader(
            "💳 Contract & Billing"
        )

        st.caption(
            "Enter tenure and billing information."
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

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=500.0,
                value=70.0,
                step=0.01,
                format="%.2f"
            )

        with col3:

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=840.0,
                step=0.01,
                format="%.2f"
            )


        st.divider()

        st.caption(
            "💡 The model evaluates all supplied customer attributes together."
        )


        submitted = st.form_submit_button(
            "🔮 Predict Customer Churn",
            use_container_width=True,
            type="primary"
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if submitted:

        input_data = pd.DataFrame(
            [
                {
                    "gender": gender,

                    "SeniorCitizen":
                        1
                        if senior_citizen == "Yes"
                        else 0,

                    "Partner":
                        partner,

                    "Dependents":
                        dependents,

                    "tenure":
                        tenure,

                    "PhoneService":
                        phone_service,

                    "MultipleLines":
                        multiple_lines,

                    "InternetService":
                        internet_service,

                    "OnlineSecurity":
                        online_security,

                    "OnlineBackup":
                        online_backup,

                    "DeviceProtection":
                        device_protection,

                    "TechSupport":
                        tech_support,

                    "StreamingTV":
                        streaming_tv,

                    "StreamingMovies":
                        streaming_movies,

                    "Contract":
                        contract,

                    "PaperlessBilling":
                        paperless_billing,

                    "PaymentMethod":
                        payment_method,

                    "MonthlyCharges":
                        monthly_charges,

                    "TotalCharges":
                        total_charges
                }
            ]
        )


        try:

            prediction = model.predict(
                input_data
            )[0]

            churn_probability = (
                get_churn_probability(
                    model,
                    input_data
                )
            )

            churn = is_churn_prediction(
                prediction
            )


            st.divider()

            st.subheader(
                "🎯 Prediction Result"
            )


            # -----------------------------------------
            # RESULT
            # -----------------------------------------

            if churn:

                st.error(
                    "⚠️ Likely to Churn"
                )

                st.markdown(
                    '<div class="risk-number risk-high">'
                    'Customer at Risk'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    "The trained model classified this "
                    "customer as likely to churn based "
                    "on the supplied information."
                )

            else:

                st.success(
                    "✅ Likely to Stay"
                )

                st.markdown(
                    '<div class="risk-number risk-low">'
                    'Lower Churn Signal'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    "The trained model classified this "
                    "customer as likely to stay based "
                    "on the supplied information."
                )


            # -----------------------------------------
            # PROBABILITY
            # -----------------------------------------

            if churn_probability is not None:

                st.markdown(
                    "### 📊 Estimated Churn Probability"
                )

                percentage = (
                    churn_probability * 100
                )

                st.markdown(
                    f'<div class="risk-number">'
                    f'{percentage:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(
                        max(
                            churn_probability,
                            0.0
                        ),
                        1.0
                    )
                )


                # Risk interpretation

                if percentage >= 70:

                    st.error(
                        "🔴 High estimated churn risk"
                    )

                    st.write(
                        "This profile has a relatively "
                        "high estimated probability of churn. "
                        "It may deserve closer retention attention."
                    )

                elif percentage >= 40:

                    st.warning(
                        "🟠 Moderate estimated churn risk"
                    )

                    st.write(
                        "The estimated churn probability is "
                        "moderate. Consider monitoring the "
                        "customer's future behaviour."
                    )

                else:

                    st.success(
                        "🟢 Lower estimated churn risk"
                    )

                    st.write(
                        "The supplied customer profile has a "
                        "lower estimated probability of churn."
                    )

            else:

                st.info(
                    "The saved model does not provide prediction probabilities."
                )


            # -----------------------------------------
            # CUSTOMER SUMMARY
            # -----------------------------------------

            st.divider()

            st.subheader(
                "📋 Customer Summary"
            )

            summary1, summary2, summary3, summary4 = st.columns(4)

            summary1.metric(
                "Contract",
                contract
            )

            summary2.metric(
                "Tenure",
                f"{tenure} months"
            )

            summary3.metric(
                "Monthly Charges",
                f"₹{monthly_charges:.2f}"
            )

            summary4.metric(
                "Internet",
                internet_service
            )


            # -----------------------------------------
            # FULL DATA
            # -----------------------------------------

            with st.expander(
                "🔎 View submitted customer information"
            ):

                st.dataframe(
                    input_data.T.rename(
                        columns={
                            0: "Value"
                        }
                    ),
                    use_container_width=True,
                    hide_index=False
                )


        except Exception as error:

            st.error(
                "❌ Prediction failed."
            )

            st.code(
                str(error)
            )

            st.info(
                "The saved pipeline must use the same "
                "feature names and categories used during training."
            )


# =========================================================
# MODEL ANALYSIS TAB
# =========================================================

with model_tab:

    st.subheader(
        "📊 Model Performance"
    )

    st.caption(
        "Final evaluation results of the tuned and class-balanced Random Forest model."
    )


    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        "76.51%"
    )

    col2.metric(
        "Precision",
        "—"
    )

    col3.metric(
        "Recall",
        "74.87%"
    )

    col4.metric(
        "F1 Score",
        "62.85%"
    )


    st.divider()


    st.subheader(
        "ROC-AUC"
    )

    st.metric(
        "ROC-AUC",
        "84.10%"
    )

    st.progress(
        0.841
    )


    st.divider()


    # ---------------------------------------------
    # METRIC EXPLANATION
    # ---------------------------------------------

    st.subheader(
        "📚 Understanding the Metrics"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.markdown(
            """
            **Accuracy**

            Measures the overall percentage of correct predictions.

            **Precision**

            Of customers predicted as churners, precision tells us
            how many actually churned.

            **Recall**

            Of all customers who actually churned, recall tells us
            how many were successfully identified.
            """
        )

    with metric_col2:

        st.markdown(
            """
            **F1 Score**

            Combines precision and recall into a single balanced metric.

            **ROC-AUC**

            Measures how well the model separates churn and
            non-churn customers across different thresholds.

            For churn detection, recall is particularly important
            because missing a real churner can mean losing an
            opportunity for retention.
            """
        )


    st.divider()


    # ---------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------

    st.subheader(
        "🌟 Feature Importance"
    )

    feature_importance = get_feature_importance(
        model
    )

    if (
        feature_importance is not None
        and not feature_importance.empty
    ):

        st.dataframe(
            feature_importance.style.format(
                {
                    "Importance": "{:.4f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Feature importance is extracted directly from the loaded tree-based model."
        )

    else:

        st.info(
            "Feature importance could not be extracted automatically from this saved model."
        )


# =========================================================
# BUSINESS INSIGHTS TAB
# =========================================================

with business_tab:

    st.subheader(
        "💡 Business Insights"
    )

    st.caption(
        "How the model output can support customer-retention analysis."
    )


    # ---------------------------------------------
    # INSIGHT 1
    # ---------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🎯 1. Identify customers at risk"
        )

        st.write(
            "The model can identify customer profiles that resemble "
            "historical churn patterns in the training data."
        )


    # ---------------------------------------------
    # INSIGHT 2
    # ---------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 📌 2. Prioritize retention analysis"
        )

        st.write(
            "The estimated churn probability can help prioritize "
            "customers for additional retention analysis."
        )


    # ---------------------------------------------
    # INSIGHT 3
    # ---------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 📱 3. Service behaviour matters"
        )

        st.write(
            "Service-related variables such as internet service, "
            "support, security and streaming services are evaluated "
            "alongside customer and billing information."
        )


    # ---------------------------------------------
    # INSIGHT 4
    # ---------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 💳 4. Billing context matters"
        )

        st.write(
            "Monthly charges, total charges, tenure and contract "
            "information provide important account-level context."
        )


    # ---------------------------------------------
    # INSIGHT 5
    # ---------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 📈 5. Probability is not certainty"
        )

        st.write(
            "A churn probability is a model estimate based on "
            "historical patterns. It does not guarantee that a "
            "specific customer will leave."
        )


    st.divider()


    st.subheader(
        "⚠️ Important Model Limitation"
    )

    st.warning(
        "The model identifies statistical patterns in the training data. "
        "It does not prove that a particular customer attribute causes churn."
    )


    st.subheader(
        "🎯 Recommended Business Interpretation"
    )

    st.write(
        "Use the prediction as decision support. Review the customer's "
        "profile and business context before taking any retention action."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer-text">'
    'Customer Churn Prediction • Machine Learning Application • '
    'Built with Python & Streamlit'
    '</div>',
    unsafe_allow_html=True
)
