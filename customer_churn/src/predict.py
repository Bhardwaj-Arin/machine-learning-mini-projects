from __future__ import annotations

import joblib
import pandas as pd

from config import load_config
from feature_engineering import add_customer_features
from utils import PROJECT_ROOT


def load_model():
    cfg = load_config()
    return joblib.load(PROJECT_ROOT / cfg["model"]["path"])


def predict_customer(customer: dict) -> dict[str, float | int]:
    model = load_model()
    row = add_customer_features(pd.DataFrame([customer]))
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else float(prediction)
    return {"prediction": prediction, "churn_probability": probability}
