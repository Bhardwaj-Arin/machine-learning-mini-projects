from __future__ import annotations

import math

import joblib

from config import load_config
from preprocessing import clean_text
from utils import PROJECT_ROOT


KEYWORDS = ["free", "win", "winner", "urgent", "call", "claim", "prize", "cash", "now", "reply"]


def load_model():
    cfg = load_config()
    return joblib.load(PROJECT_ROOT / cfg["model"]["path"])


def spam_probability(model, text: str) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([text])[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function([text])[0])
        return 1 / (1 + math.exp(-score))
    return float(model.predict([text])[0])


def predict_message(message: str) -> dict:
    cleaned = clean_text(message)
    model = load_model()
    prediction = int(model.predict([cleaned])[0])
    return {
        "prediction": "spam" if prediction else "ham",
        "spam_probability": spam_probability(model, cleaned),
        "highlighted_keywords": [word for word in KEYWORDS if word in cleaned.split()],
        "cleaned_text": cleaned,
    }
