from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import load_config
from evaluate import save_classification_outputs
from feature_engineering import add_customer_features
from preprocessing import load_customer_data, split_features_target
from utils import PROJECT_ROOT, ensure_dirs, set_seed, setup_logging


LOGGER = logging.getLogger(__name__)


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(estimator) -> Pipeline:
    numeric = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "BalanceToSalaryRatio", "BalanceBucket"]
    categorical = ["Geography", "Gender", "TenureGroup"]
    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_encoder())]), categorical),
                ],
                remainder="drop",
            )),
            ("model", estimator),
        ]
    )


def candidate_models(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=seed, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=seed),
    }


def train() -> pd.DataFrame:
    setup_logging()
    cfg = load_config()
    seed = int(cfg.get("random_state", 42))
    set_seed(seed)
    ensure_dirs()

    df = load_customer_data(PROJECT_ROOT / cfg["data"]["raw"])
    X, y = split_features_target(df)
    X = add_customer_features(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg["training"]["test_size"]), random_state=seed, stratify=y
    )

    leaderboard = []
    best_model = None
    best_score = -1.0
    for name, estimator in candidate_models(seed).items():
        LOGGER.info("Training %s", name)
        model = build_pipeline(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        leaderboard.append(row)
        if row["f1"] > best_score:
            best_score = row["f1"]
            best_model = model

    output_dir = PROJECT_ROOT / "outputs"
    pd.DataFrame(leaderboard).sort_values("f1", ascending=False).to_csv(output_dir / "model_leaderboard.csv", index=False)
    save_classification_outputs(y_test, best_model.predict(X_test), labels=[0, 1], output_dir=output_dir)
    joblib.dump(best_model, PROJECT_ROOT / cfg["model"]["path"])
    LOGGER.info("Saved model to %s", cfg["model"]["path"])
    return pd.DataFrame(leaderboard)


if __name__ == "__main__":
    print(train())
