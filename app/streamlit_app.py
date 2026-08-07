from __future__ import annotations

import html
import importlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODULES_TO_CLEAR = [
    "predict",
    "config",
    "utils",
    "preprocessing",
    "feature_engineering",
    "evaluate",
]


PROJECTS = {
    "Home": {
        "slug": None,
        "title": "Codsoft ML Project Dashboard",
        "description": "One professional Streamlit app for Customer Churn, Spam SMS Detection, and Movie Genre Classification.",
    },
    "Customer Churn": {
        "slug": "customer_churn",
        "title": "Customer Churn Prediction",
        "description": "Estimate whether a bank customer is likely to leave and review retention signals.",
    },
    "Spam SMS Detection": {
        "slug": "spam_sms_detection",
        "title": "Spam SMS Detection",
        "description": "Classify SMS messages with spam probability, diagnostics, and highlighted signals.",
    },
    "Movie Genre Classification": {
        "slug": "movie_genre_classification",
        "title": "Movie Genre Classification",
        "description": "Predict top movie genres from plot descriptions with confidence ranking.",
    },
}


st.set_page_config(
    page_title="Codsoft ML Dashboard",
    page_icon="ML",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 2.2rem; max-width: 1280px;}
    .panel {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.25);
    }
    .chip {
        display: inline-block;
        padding: 0.35rem 0.6rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: rgba(56, 189, 248, 0.12);
        color: #7dd3fc;
        font-weight: 700;
    }
    .signal {
        display: inline-block;
        padding: 0.2rem 0.45rem;
        margin: 0.1rem;
        border-radius: 6px;
        background: rgba(255, 75, 75, 0.16);
        border: 1px solid rgba(255, 75, 75, 0.32);
        color: #ff8a8a;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_predict_module(slug: str):
    src_path = ROOT / slug / "src"
    for module_name in MODULES_TO_CLEAR:
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(src_path))
    try:
        return importlib.import_module("predict")
    finally:
        try:
            sys.path.remove(str(src_path))
        except ValueError:
            pass


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.70:
        return "High", "Prioritize a retention call, fee review, or loyalty offer."
    if probability >= 0.40:
        return "Medium", "Monitor the account and review product fit."
    return "Low", "Maintain regular engagement and service quality."


def customer_drivers(customer: dict) -> list[str]:
    drivers = []
    if customer["Age"] >= 50:
        drivers.append("Age profile is above the common mid-career customer range.")
    if customer["IsActiveMember"] == 0:
        drivers.append("Inactive membership is a strong churn warning signal.")
    if customer["NumOfProducts"] >= 3:
        drivers.append("Higher product count can indicate friction or complex account needs.")
    if customer["Balance"] > 100000:
        drivers.append("Large balance makes retention action commercially important.")
    if customer["Geography"] == "Germany":
        drivers.append("Germany segment often shows stronger churn in this dataset.")
    return drivers or ["No obvious manual risk trigger stands out from the selected profile."]


def confidence_label(score: float) -> str:
    if score >= 0.80 or score <= 0.20:
        return "High"
    if score >= 0.60 or score <= 0.40:
        return "Medium"
    return "Uncertain"


def highlight_message(message: str, keywords: list[str]) -> str:
    safe = html.escape(message)
    for keyword in sorted(keywords, key=len, reverse=True):
        safe = safe.replace(keyword, f"<span class='signal'>{keyword}</span>")
        safe = safe.replace(keyword.upper(), f"<span class='signal'>{keyword.upper()}</span>")
        safe = safe.replace(keyword.title(), f"<span class='signal'>{keyword.title()}</span>")
    return safe


def confidence_text(score: float) -> str:
    if score >= 0.50:
        return "Strong match"
    if score >= 0.25:
        return "Possible match"
    return "Weak signal"


with st.sidebar:
    st.title("Codsoft ML")
    selected_project = st.radio(
        "Select Project",
        list(PROJECTS.keys()),
        index=0,
    )
    st.divider()
    st.caption(PROJECTS[selected_project]["description"])


def render_home() -> None:
    st.title("Codsoft ML Project Dashboard")
    st.caption("Unified interface for three machine learning projects")

    c1, c2, c3 = st.columns(3)
    c1.metric("Projects", "3")
    c2.metric("App Entry Point", "1")
    c3.metric("Model Type", "scikit-learn")

    st.subheader("Project Overview")
    overview = pd.DataFrame(
        [
            {
                "Project": "Customer Churn",
                "Input": "Bank customer profile",
                "Output": "Churn probability and risk level",
                "Use Case": "Retention targeting",
            },
            {
                "Project": "Spam SMS Detection",
                "Input": "SMS message text",
                "Output": "Spam probability and detected signals",
                "Use Case": "Message filtering",
            },
            {
                "Project": "Movie Genre Classification",
                "Input": "Movie plot description",
                "Output": "Top genre predictions",
                "Use Case": "Content tagging",
            },
        ]
    )
    st.dataframe(overview, use_container_width=True, hide_index=True)

    st.subheader("Repository Workflow")
    st.markdown(
        """
        ```text
        Raw Data -> Preprocessing -> Feature Engineering -> Model Prediction -> Dashboard Output
        ```
        """
    )

    st.info("Use the sidebar to open a project dashboard.")


def render_customer_churn() -> None:
    predict = load_predict_module("customer_churn")
    st.title("Customer Churn Prediction")
    st.caption("Bank customer retention dashboard")

    scenarios = {
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
    }

    scenario = st.selectbox("Scenario", list(scenarios))
    defaults = scenarios[scenario]

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
        result = predict.predict_customer(customer)
        churn_probability = result["churn_probability"]
        stay_probability = 1 - churn_probability
        level, action = risk_label(churn_probability)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn Probability", f"{churn_probability * 100:.2f}%")
        m2.metric("Stay Probability", f"{stay_probability * 100:.2f}%")
        m3.metric("Risk Level", level)
        m4.metric("Prediction", "Likely to Churn" if result["prediction"] else "Likely to Stay")
        st.progress(churn_probability)

        left, right = st.columns([1.1, 0.9])
        with left:
            st.subheader("Probability Breakdown")
            st.bar_chart(pd.DataFrame({"Outcome": ["Stay", "Churn"], "Probability": [stay_probability, churn_probability]}), x="Outcome", y="Probability")
            st.subheader("Input Summary")
            st.dataframe(pd.DataFrame([customer]), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Retention Notes")
            for driver in customer_drivers(customer):
                st.write(f"- {driver}")
            st.info(action)

        export = pd.DataFrame([{**customer, **result, "risk_level": level}])
        st.download_button("Download Prediction", export.to_csv(index=False), "customer_churn_prediction.csv", mime="text/csv")


def render_spam_detection() -> None:
    predict = load_predict_module("spam_sms_detection")
    st.title("Spam SMS Detection")
    st.caption("Message risk analysis dashboard")

    samples = {
        "Friendly message": "Hi, how are u",
        "Reward claim": "URGENT! You have won a free prize. Call now to claim.",
        "Account warning": "Your account is temporarily blocked. Click this link now to verify your details.",
    }
    sample_name = st.selectbox("Sample", list(samples))
    message = st.text_area("Message", samples[sample_name], height=170)

    if st.button("Analyze Message", type="primary"):
        result = predict.predict_message(message)
        score = result["spam_probability"]
        ham_score = 1 - score

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spam Probability", f"{score * 100:.2f}%")
        m2.metric("Ham Probability", f"{ham_score * 100:.2f}%")
        m3.metric("Prediction", result["prediction"].upper())
        m4.metric("Confidence", confidence_label(score))
        st.progress(score)

        left, right = st.columns([1.05, 0.95])
        with left:
            st.subheader("Class Scores")
            st.bar_chart(pd.DataFrame({"Class": ["Ham", "Spam"], "Score": [ham_score, score]}), x="Class", y="Score")
            st.subheader("Cleaned Text")
            st.code(result["cleaned_text"], language="text")
        with right:
            st.subheader("Message Diagnostics")
            d1, d2, d3 = st.columns(3)
            d1.metric("Characters", len(message))
            d2.metric("Words", len(message.split()))
            d3.metric("Signals", len(result["highlighted_keywords"]))
            st.markdown("<div class='panel'>" + highlight_message(message, result["highlighted_keywords"]) + "</div>", unsafe_allow_html=True)
            st.write("Detected keywords:", ", ".join(result["highlighted_keywords"]) or "None")

        export = pd.DataFrame([{**result, "message": message, "confidence": confidence_label(score)}])
        st.download_button("Download Prediction", export.to_csv(index=False), "sms_prediction.csv", mime="text/csv")


def render_movie_genre() -> None:
    predict = load_predict_module("movie_genre_classification")
    preprocessing = importlib.import_module("preprocessing")
    st.title("Movie Genre Classification")
    st.caption("Plot-to-genre analysis dashboard")

    examples = {
        "Mystery thriller": "A retired detective returns to solve a mysterious disappearance in a coastal town while old secrets begin to surface.",
        "Romantic drama": "Two childhood friends reconnect after years apart and must choose between ambition, family expectations, and love.",
        "Science fiction": "A crew of astronauts discovers a signal from a distant planet that changes humanity's understanding of time.",
        "Comedy": "A nervous office worker accidentally becomes the face of a citywide campaign after one chaotic interview.",
    }
    c1, c2 = st.columns([0.75, 0.25])
    example_name = c1.selectbox("Example", list(examples))
    top_k = c2.slider("Genres to show", 2, 5, 3)
    description = st.text_area("Movie Description", examples[example_name], height=190)

    if st.button("Predict Genres", type="primary"):
        results = predict.predict_genres(description, top_k=top_k)
        df = pd.DataFrame(results)
        top = df.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top Genre", str(top["genre"]).title())
        m2.metric("Top Confidence", f"{float(top['confidence']) * 100:.2f}%")
        m3.metric("Words", len(description.split()))
        m4.metric("Characters", len(description))

        chips = "".join(f"<span class='chip'>{row.genre.title()} - {row.confidence * 100:.1f}%</span>" for row in df.itertuples())
        st.markdown(chips, unsafe_allow_html=True)

        left, right = st.columns([1.1, 0.9])
        with left:
            st.subheader("Genre Ranking")
            st.dataframe(df.assign(confidence_percent=(df["confidence"] * 100).round(2)), use_container_width=True, hide_index=True)
            st.bar_chart(df, x="genre", y="confidence")
        with right:
            st.subheader("Interpretation")
            for row in df.itertuples():
                st.write(f"- {row.genre.title()}: {confidence_text(float(row.confidence))}")
            st.subheader("Cleaned Description")
            st.code(preprocessing.clean_text(description), language="text")

        export = df.copy()
        export["description"] = description
        st.download_button("Download Prediction", export.to_csv(index=False), "movie_genre_prediction.csv", mime="text/csv")


if selected_project == "Home":
    render_home()
elif selected_project == "Customer Churn":
    render_customer_churn()
elif selected_project == "Spam SMS Detection":
    render_spam_detection()
else:
    render_movie_genre()
