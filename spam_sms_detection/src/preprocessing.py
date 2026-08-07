from __future__ import annotations

import html
import re
import string

import pandas as pd


CONTRACTIONS = {"can't": "cannot", "won't": "will not", "n't": " not", "'re": " are", "'s": " is", "'ll": " will"}


def clean_text(text: str) -> str:
    text = html.unescape(str(text)).lower()
    for old, new in CONTRACTIONS.items():
        text = text.replace(old, new)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\d+", " number ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def load_spam_data(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1")
    df = raw[["v1", "v2"]].copy()
    df.columns = ["label", "message"]
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["clean_message"] = df["message"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    return df
