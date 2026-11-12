import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Credit Risk Classification",
    page_icon="🏦",
    layout="wide"
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# =====================================
# APPLICATION HEADER
# =====================================

st.title("🏦 Credit Risk Classification Engine")

st.write(
    "An ML-powered application to classify "
    "customers into Low Risk and High Risk categories."
)

st.divider()

# =====================================
# CUSTOMER INFORMATION
# =====================================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    person_age = st.number_input(
        "Person Age",
        min_value=21,
        max_value=100,
        value=30
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=1,
        max_value=10000000,
        value=60000
    )

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0,
        max_value=60,
        value=5
    )

    credit_history_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0,
        max_value=60,
        value=5
    )

with col2:

    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

# =====================================
# LOAN INFORMATION
# =====================================

st.divider()

st.subheader("💰 Loan Information")

col3, col4 = st.columns(2)

with col3:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=1,
        max_value=10000000,
        value=15000
    )

    loan_int_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=12.0,
        step=0.1
    )

with col4:

    debt_to_income = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=10.0,
        value=0.3,
        step=0.01
    )

    loan_percent_income = loan_amnt / person_income

    st.metric(
        "Loan-to-Income Ratio",
        f"{loan_percent_income:.3f}"
    )

# =====================================
# PREDICTION
# =====================================

st.divider()

if st.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
):

    # Create input DataFrame
    input_data = pd.DataFrame({

        "person_age": [person_age],

        "person_income": [person_income],

        "person_emp_length": [person_emp_length],

        "loan_amnt": [loan_amnt],

        "loan_int_rate": [loan_int_rate],

        "loan_percent_income": [loan_percent_income],

        "credit_history_length": [credit_history_length],

        "debt_to_income": [debt_to_income],

        "home_ownership": [home_ownership],

        "loan_intent": [loan_intent]

    })

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Generate prediction probability
    probabilities = model.predict_proba(input_data)[0]

    high_risk_probability = probabilities[1] * 100

    low_risk_probability = probabilities[0] * 100

    # Display results
    st.divider()

    st.subheader("📊 Credit Risk Prediction")

    col5, col6 = st.columns(2)

    with col5:

        st.metric(
            "Low Risk Probability",
            f"{low_risk_probability:.2f}%"
        )

    with col6:

        st.metric(
            "High Risk Probability",
            f"{high_risk_probability:.2f}%"
        )

    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK CUSTOMER"
        )

        st.warning(
            "The model classifies this customer as High Risk."
        )

    else:

        st.success(
            "✅ LOW RISK CUSTOMER"
        )

        st.info(
            "The model classifies this customer as Low Risk."
        )

    st.caption(
        "This is an educational prediction based on a synthetic "
        "dataset, not a real lending or credit approval decision."
    )