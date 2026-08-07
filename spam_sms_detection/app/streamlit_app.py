import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import KEYWORDS, predict_message


st.set_page_config(page_title="Spam SMS Detection", page_icon="SMS", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1260px;}
    .keyword {
        display: inline-block;
        padding: 0.2rem 0.45rem;
        margin: 0.1rem;
        border-radius: 6px;
        background: rgba(255, 75, 75, 0.16);
        border: 1px solid rgba(255, 75, 75, 0.32);
        color: #ff8a8a;
        font-weight: 700;
    }
    .message-box {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: rgba(15, 23, 42, 0.28);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def confidence_label(score: float) -> str:
    if score >= 0.80 or score <= 0.20:
        return "High"
    if score >= 0.60 or score <= 0.40:
        return "Medium"
    return "Uncertain"


def highlight_message(message: str, keywords: list[str]) -> str:
    safe = html.escape(message)
    for keyword in sorted(keywords, key=len, reverse=True):
        safe = safe.replace(keyword, f"<span class='keyword'>{keyword}</span>")
        safe = safe.replace(keyword.upper(), f"<span class='keyword'>{keyword.upper()}</span>")
        safe = safe.replace(keyword.title(), f"<span class='keyword'>{keyword.title()}</span>")
    return safe


st.title("Spam SMS Detection")
st.caption("Message risk analysis dashboard")

samples = {
    "Friendly message": "Hi, how are u",
    "Reward claim": "URGENT! You have won a free prize. Call now to claim.",
    "Account warning": "Your account is temporarily blocked. Click this link now to verify your details.",
}

with st.sidebar:
    st.header("Text Profile")
    sample_name = st.selectbox("Sample", list(samples))
    st.write("Tracked signals")
    st.write(", ".join(KEYWORDS))

message = st.text_area("Message", samples[sample_name], height=170)
analyze = st.button("Analyze Message", type="primary")

if analyze:
    result = predict_message(message)
    score = result["spam_probability"]
    ham_score = 1 - score
    confidence = confidence_label(score)
    word_count = len(message.split())
    char_count = len(message)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spam Probability", f"{score * 100:.2f}%")
    m2.metric("Ham Probability", f"{ham_score * 100:.2f}%")
    m3.metric("Prediction", result["prediction"].upper())
    m4.metric("Confidence", confidence)
    st.progress(min(max(score, 0.0), 1.0))

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Class Scores")
        chart_df = pd.DataFrame({"Class": ["Ham", "Spam"], "Score": [ham_score, score]})
        st.bar_chart(chart_df, x="Class", y="Score", color="#38bdf8")

        st.subheader("Cleaned Text")
        st.code(result["cleaned_text"], language="text")

    with right:
        st.subheader("Message Diagnostics")
        d1, d2, d3 = st.columns(3)
        d1.metric("Characters", char_count)
        d2.metric("Words", word_count)
        d3.metric("Signals", len(result["highlighted_keywords"]))

        st.markdown("<div class='message-box'>" + highlight_message(message, result["highlighted_keywords"]) + "</div>", unsafe_allow_html=True)
        st.write("Detected keywords:", ", ".join(result["highlighted_keywords"]) or "None")

        if result["prediction"] == "spam":
            st.error("This message should be reviewed before trusting links, rewards, or contact requests.")
        else:
            st.success("This message looks closer to normal conversation.")

    export = pd.DataFrame([{**result, "message": message, "confidence": confidence}])
    st.download_button("Download Prediction", export.to_csv(index=False), "sms_prediction.csv", mime="text/csv")
else:
    st.info("Enter an SMS message and run the analysis.")
