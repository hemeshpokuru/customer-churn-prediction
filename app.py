import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("customer_churn_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to churn based on "
    "their demographic, service, contract, and billing information."
)