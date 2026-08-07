import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_genres
from preprocessing import clean_text


st.set_page_config(page_title="Movie Genre Classification", page_icon="MG", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1260px;}
    .genre-chip {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: rgba(56, 189, 248, 0.12);
        color: #7dd3fc;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


EXAMPLES = {
    "Mystery thriller": "A retired detective returns to solve a mysterious disappearance in a coastal town while old secrets begin to surface.",
    "Romantic drama": "Two childhood friends reconnect after years apart and must choose between ambition, family expectations, and love.",
    "Science fiction": "A crew of astronauts discovers a signal from a distant planet that changes humanity's understanding of time.",
    "Comedy": "A nervous office worker accidentally becomes the face of a citywide campaign after one chaotic interview.",
}


def confidence_text(score: float) -> str:
    if score >= 0.50:
        return "Strong match"
    if score >= 0.25:
        return "Possible match"
    return "Weak signal"


st.title("Movie Genre Classification")
st.caption("Plot-to-genre analysis dashboard")

with st.sidebar:
    st.header("Model")
    st.write("TF-IDF text classifier")
    top_k = st.slider("Genres to show", 2, 5, 3)
    example_name = st.selectbox("Example", list(EXAMPLES))

description = st.text_area("Movie Description", EXAMPLES[example_name], height=190)
predict = st.button("Predict Genres", type="primary")

if predict:
    results = predict_genres(description, top_k=top_k)
    df = pd.DataFrame(results)
    cleaned = clean_text(description)
    top = df.iloc[0]

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top Genre", str(top["genre"]).title())
    m2.metric("Top Confidence", f"{float(top['confidence']) * 100:.2f}%")
    m3.metric("Words", len(description.split()))
    m4.metric("Characters", len(description))

    chips = "".join(f"<span class='genre-chip'>{row.genre.title()} - {row.confidence * 100:.1f}%</span>" for row in df.itertuples())
    st.markdown(chips, unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Genre Ranking")
        st.dataframe(df.assign(confidence_percent=(df["confidence"] * 100).round(2)), use_container_width=True, hide_index=True)
        st.bar_chart(df, x="genre", y="confidence", color="#7dd3fc")

    with right:
        st.subheader("Interpretation")
        for row in df.itertuples():
            st.write(f"- {row.genre.title()}: {confidence_text(float(row.confidence))}")
        st.subheader("Cleaned Description")
        st.code(cleaned, language="text")

    export = df.copy()
    export["description"] = description
    st.download_button("Download Prediction", export.to_csv(index=False), "movie_genre_prediction.csv", mime="text/csv")
else:
    st.info("Enter a plot description and run genre prediction.")
