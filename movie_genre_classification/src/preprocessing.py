from __future__ import annotations

import re

import pandas as pd


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_movie_data(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = [part.strip() for part in line.split(" ::: ")]
            if len(parts) >= 4:
                rows.append({"id": parts[0], "title": parts[1], "genre": parts[2], "description": parts[3]})
    df = pd.DataFrame(rows)
    df["clean_description"] = df["description"].apply(clean_text)
    return df
