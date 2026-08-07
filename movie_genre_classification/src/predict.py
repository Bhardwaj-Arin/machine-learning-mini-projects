from __future__ import annotations

import math

import joblib

from config import load_config
from preprocessing import clean_text
from utils import PROJECT_ROOT


def load_model():
    cfg = load_config()
    return joblib.load(PROJECT_ROOT / cfg["model"]["path"])


def predict_genres(description: str, top_k: int = 3) -> list[dict]:
    model = load_model()
    cleaned = clean_text(description)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        classes = model.classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:top_k]
        return [{"genre": genre, "confidence": float(score)} for genre, score in ranked]
    if hasattr(model, "decision_function"):
        scores = model.decision_function([cleaned])[0]
        classes = model.classes_
        if getattr(scores, "ndim", 1) == 0:
            positive = 1 / (1 + math.exp(-float(scores)))
            negative = 1 - positive
            ranked = sorted(zip(classes, [negative, positive]), key=lambda item: item[1], reverse=True)
        else:
            exp_scores = [math.exp(float(score) - max(scores)) for score in scores]
            total = sum(exp_scores) or 1
            ranked = sorted(zip(classes, [score / total for score in exp_scores]), key=lambda item: item[1], reverse=True)
        return [{"genre": str(genre), "confidence": float(score)} for genre, score in ranked[:top_k]]
    prediction = str(model.predict([cleaned])[0])
    return [{"genre": prediction, "confidence": 1.0}]
