import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_customer


st.set_page_config(page_title="Customer Churn Prediction", page_icon="CC", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1260px;}
    .app-card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.30);
    }
    .risk-high {color: #ff4b4b; font-weight: 700;}
    .risk-medium {color: #f59e0b; font-weight: 700;}
    .risk-low {color: #22c55e; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.70:
        return "High", "risk-high"
    if probability >= 0.40:
        return "Medium", "risk-medium"
    return "Low", "risk-low"


def risk_drivers(customer: dict) -> list[str]:
    drivers = []
    if customer["Age"] >= 50:
        drivers.append("Age profile is above the common mid-career customer range.")
    if customer["IsActiveMember"] == 0:
        drivers.append("Inactive membership is a strong churn warning signal.")
    if customer["NumOfProducts"] >= 3:
        drivers.append("Higher product count can indicate friction or complex account needs.")
    if customer["Balance"] > 100000:
        drivers.append("Large account balance makes retention action commercially important.")
    if customer["Geography"] == "Germany":
        drivers.append("Germany segment often shows stronger churn in this dataset.")
    if not drivers:
        drivers.append("No obvious manual risk trigger stands out from the selected profile.")
    return drivers


st.title("Customer Churn Prediction")
st.caption("Bank customer retention dashboard")

with st.sidebar:
    st.header("Model")
    st.write("Saved scikit-learn pipeline")
    st.write("Threshold: 50% churn probability")
    st.divider()
    scenario = st.selectbox(
        "Scenario",
        ["Balanced customer", "High value inactive", "Low risk loyal"],
    )

defaults = {
    "Balanced customer": {
        "credit_score": 650,
        "geography": "France",
        "gender": "Female",
        "age": 35,
        "tenure": 5,
        "balance": 60000.0,
        "num_products": 1,
        "has_card": True,
        "is_active": True,
        "estimated_salary": 75000.0,
    },
    "High value inactive": {
        "credit_score": 590,
        "geography": "Germany",
        "gender": "Male",
        "age": 54,
        "tenure": 2,
        "balance": 145000.0,
        "num_products": 3,
        "has_card": True,
        "is_active": False,
        "estimated_salary": 93000.0,
    },
    "Low risk loyal": {
        "credit_score": 735,
        "geography": "Spain",
        "gender": "Female",
        "age": 31,
        "tenure": 8,
        "balance": 25000.0,
        "num_products": 2,
        "has_card": True,
        "is_active": True,
        "estimated_salary": 81000.0,
    },
}[scenario]

with st.form("customer_form"):
    st.subheader("Customer Profile")
    c1, c2, c3 = st.columns(3)
    credit_score = c1.slider("Credit Score", 300, 900, defaults["credit_score"])
    geography = c2.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(defaults["geography"]))
    gender = c3.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(defaults["gender"]))
    age = c1.slider("Age", 18, 95, defaults["age"])
    tenure = c2.slider("Tenure", 0, 10, defaults["tenure"])
    balance = c3.number_input("Balance", min_value=0.0, value=defaults["balance"], step=1000.0)
    num_products = c1.selectbox("Number of Products", [1, 2, 3, 4], index=[1, 2, 3, 4].index(defaults["num_products"]))
    has_card = c2.toggle("Has Credit Card", value=defaults["has_card"])
    is_active = c3.toggle("Active Member", value=defaults["is_active"])
    estimated_salary = c1.number_input("Estimated Salary", min_value=0.0, value=defaults["estimated_salary"], step=1000.0)
    submitted = st.form_submit_button("Predict Churn", type="primary")

customer = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": int(has_card),
    "IsActiveMember": int(is_active),
    "EstimatedSalary": estimated_salary,
}

if submitted:
    result = predict_customer(customer)
    probability = result["churn_probability"]
    stay_probability = 1 - probability
    label, css_class = risk_label(probability)
    prediction_text = "Likely to Churn" if result["prediction"] else "Likely to Stay"

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn Probability", f"{probability * 100:.2f}%")
    m2.metric("Stay Probability", f"{stay_probability * 100:.2f}%")
    m3.metric("Risk Level", label)
    m4.metric("Prediction", prediction_text)

    st.progress(min(max(probability, 0.0), 1.0))

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Probability Breakdown")
        chart_df = pd.DataFrame(
            {"Outcome": ["Stay", "Churn"], "Probability": [stay_probability, probability]}
        )
        st.bar_chart(chart_df, x="Outcome", y="Probability", color="#ff4b4b")

        st.subheader("Input Summary")
        st.dataframe(pd.DataFrame([customer]), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Retention Notes")
        st.markdown(f"<p class='{css_class}'>{label} churn risk</p>", unsafe_allow_html=True)
        for driver in risk_drivers(customer):
            st.write(f"- {driver}")

        recommended_action = (
            "Prioritize a retention call, fee review, or loyalty offer."
            if probability >= 0.70
            else "Monitor the customer and review product fit."
            if probability >= 0.40
            else "Maintain regular engagement and service quality."
        )
        st.info(recommended_action)

    export = pd.DataFrame([{**customer, **result, "risk_level": label}])
    st.download_button(
        "Download Prediction",
        export.to_csv(index=False),
        "customer_churn_prediction.csv",
        mime="text/csv",
    )
else:
    st.info("Choose a customer profile and run a prediction.")
