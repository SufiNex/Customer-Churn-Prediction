
import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    with open("models/customer_churn_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #17365D;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Customer Churn Prediction")
st.write("Analyze customer information and predict churn probability.")
st.divider()

# Sidebar
st.sidebar.title("About Project")
st.sidebar.info(
    "This Machine Learning application predicts "
    "whether a telecom customer is likely to leave the service."
)

st.sidebar.markdown("### Model Information")
st.sidebar.write("Algorithm: Tuned ML Model")
st.sidebar.write("Dataset: Telco Customer Churn")

# Customer Information
st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

st.divider()

# Contract Information
st.header("📄 Contract Information")

col4, col5 = st.columns(2)

with col4:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col5:
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.divider()

# Billing Information
st.header("💳 Billing Information")

col6, col7 = st.columns(2)

with col6:
    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

with col7:
    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

st.divider()

# Prediction
if st.button("🔍 Predict Customer Churn", use_container_width=True):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.header("📈 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    st.progress(float(probability))

    if probability >= 0.70:
        st.warning("High estimated churn risk")
    elif probability >= 0.40:
        st.info("Medium estimated churn risk")
    else:
        st.success("Low estimated churn risk")

    st.caption(
        "Prediction is an estimate from the trained model, "
        "not a guarantee of customer behavior."
    )
