from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import load_config
from evaluate import save_classification_outputs
from preprocessing import load_spam_data
from utils import PROJECT_ROOT, ensure_dirs, set_seed, setup_logging


LOGGER = logging.getLogger(__name__)


def candidates(seed: int) -> dict[str, object]:
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=seed),
    }


def train() -> pd.DataFrame:
    setup_logging()
    cfg = load_config()
    seed = int(cfg.get("random_state", 42))
    set_seed(seed)
    ensure_dirs()
    df = load_spam_data(PROJECT_ROOT / cfg["data"]["raw"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["target"], test_size=float(cfg["training"]["test_size"]), random_state=seed, stratify=df["target"]
    )
    leaderboard = []
    best_model = None
    best_f1 = -1.0
    for name, estimator in candidates(seed).items():
        LOGGER.info("Training %s", name)
        model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95, max_features=20000, sublinear_tf=True)),
            ("model", estimator),
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        row = {"model": name, "accuracy": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred, zero_division=0)}
        leaderboard.append(row)
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_model = model
    output_dir = PROJECT_ROOT / "outputs"
    pd.DataFrame(leaderboard).sort_values("f1", ascending=False).to_csv(output_dir / "model_leaderboard.csv", index=False)
    save_classification_outputs(y_test, best_model.predict(X_test), labels=[0, 1], output_dir=output_dir)
    joblib.dump(best_model, PROJECT_ROOT / cfg["model"]["path"])
    LOGGER.info("Saved model to %s", cfg["model"]["path"])
    return pd.DataFrame(leaderboard)


if __name__ == "__main__":
    print(train())
