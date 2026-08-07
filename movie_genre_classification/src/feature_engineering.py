from __future__ import annotations

import pandas as pd


def add_description_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["description_length"] = data["description"].str.len()
    data["word_count"] = data["description"].str.split().str.len()
    return data
