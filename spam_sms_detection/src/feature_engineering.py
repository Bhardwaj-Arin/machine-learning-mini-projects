from __future__ import annotations

import pandas as pd


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["message_length"] = data["message"].str.len()
    data["word_count"] = data["message"].str.split().str.len()
    data["has_url"] = data["message"].str.contains(r"http|www", case=False, regex=True).astype(int)
    return data
