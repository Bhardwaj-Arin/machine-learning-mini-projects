from __future__ import annotations

import pandas as pd


DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET = "Exited"


def load_customer_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols + [TARGET])
    y = df[TARGET].astype(int)
    return X, y
