# CodSoft Machine Learning Projects

Three end-to-end machine learning projects completed as part of the CodSoft ML Internship, restructured into a single, production-style repository with modular pipelines, configuration-driven training, saved models, evaluation artifacts, and one unified Streamlit dashboard.

| Project | Task Type | Best Model | Headline Metric |
|---|---|---|---|
| [Customer Churn Prediction](#1-customer-churn-prediction) | Binary classification | Random Forest | ROC-AUC 0.860 |
| [Spam SMS Detection](#2-spam-sms-detection) | Binary text classification | Linear SVM | F1 0.927, ROC-AUC 0.994 |
| [Movie Genre Classification](#3-movie-genre-classification) | 27-class text classification | Linear SVM | Weighted F1 0.580 |

---

## Repository Structure

```text
codsoft/
├── customer_churn/
│   ├── data/raw/                  # Churn_Modelling.csv
│   ├── notebooks/                 # EDA + experimentation
│   ├── src/                       # config, preprocessing, feature_engineering, train, evaluate, predict, utils
│   ├── models/                    # saved .joblib pipeline
│   ├── docs/                      # Business_Problem, Architecture, Data_Dictionary, Deployment, User_Guide
│   ├── reports/                   # Project_Report.pdf
│   └── config.yaml
├── spam_sms_detection/            # same structure as above
├── movie_genre_classification/    # same structure as above
├── app/
│   └── streamlit_app.py           # unified dashboard for all three projects
├── requirements.txt
├── Makefile
└── LICENSE
```

Each project is self-contained (its own data, config, and model) but shares a common architecture, so any one of them can be trained, evaluated, or served independently.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a model (repeat for spam_sms_detection / movie_genre_classification)
cd customer_churn
python src/train.py
cd ..

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

Or use the Makefile shortcuts:

```bash
make train-customer   # trains the churn model
make train-spam       # trains the spam detector
make train-movie      # trains the genre classifier
make app              # launches the Streamlit dashboard
```

Use the sidebar in the app to switch between Customer Churn, Spam SMS Detection, and Movie Genre Classification.

---

## 1. Customer Churn Prediction

**Problem:** Predict whether a bank customer will exit (`Exited` = 1), using account, demographic, and activity signals from a 10,000-row banking dataset (`Churn_Modelling.csv`).

**Pipeline:** raw data → feature engineering (balance-to-salary ratio, tenure/balance buckets) → `ColumnTransformer` (median/most-frequent imputation, scaling, one-hot encoding) → model comparison → best model by F1.

**Models compared:** Logistic Regression, Random Forest, Gradient Boosting — all trained with `class_weight="balanced"` to counter the dataset's ~20% churn rate.

**Result — Random Forest selected (highest F1):**

| Metric | Value |
|---|---|
| Accuracy | 80.95% |
| Precision (churn) | 52.27% |
| Recall (churn) | 73.46% |
| F1 (churn) | 61.08% |
| ROC-AUC | 0.860 |

**Top drivers of churn:** `Age`, `NumOfProducts`, `Balance`, `IsActiveMember`, `EstimatedSalary`, and `Geography = Germany` (by feature importance).

**Note on the precision/recall trade-off:** class balancing was used deliberately to prioritize *catching* likely churners (recall) over minimizing false alarms, which is usually the more valuable trade-off for a retention use case — flagging a customer who doesn't churn costs a discount email, missing one who does costs the customer.

---

## 2. Spam SMS Detection

**Problem:** Classify SMS messages as spam or ham using a cleaned NLP pipeline and TF-IDF features, on the classic UCI SMS Spam Collection (`spam.csv`).

**Pipeline:** raw text → cleaning/normalization → TF-IDF vectorization → model comparison → best model by F1.

**Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVM.

**Result — Linear SVM selected (highest F1):**

| Metric | Value |
|---|---|
| Accuracy | 98.16% |
| Precision (spam) | 93.08% |
| Recall (spam) | 92.37% |
| F1 (spam) | 92.72% |
| ROC-AUC | 0.994 |

This is the strongest-performing model in the repo, which is consistent with this being the most well-studied and separable of the three datasets.

---

## 3. Movie Genre Classification

**Problem:** Predict a movie's genre from its plot description — a 27-class, heavily imbalanced text classification task (documentary and drama dominate; genres like biography or war have very few examples).

**Pipeline:** raw plot text → cleaning → TF-IDF vectorization → model comparison → best model by weighted F1, retrained on full training data before final test evaluation.

**Models compared:** Dummy Baseline, Multinomial Naive Bayes, Logistic Regression, Linear SVM.

**Result — Linear SVM selected (evaluated on the held-out test set, 54,200 rows):**

| Metric | Value |
|---|---|
| Test Accuracy | 58.28% |
| Macro F1 | 0.393 |
| Weighted F1 | 0.580 |

**Honest note on performance:** this is the hardest of the three problems by a wide margin. Well-performing majority classes (documentary F1 ≈ 0.77, comedy F1 ≈ 0.58) pull weighted F1 up, but minority genres (biography, war, crime) score poorly (F1 well under 0.25) because of severe class imbalance and short, often ambiguous plot summaries. The macro-F1 gap versus weighted-F1 reflects this directly. This matches results reported by other public solutions on the same dataset — it is a genuinely hard multiclass problem, not a broken pipeline.

---

## Unified Streamlit Dashboard

`app/streamlit_app.py` serves all three projects from a single app with sidebar navigation, showing:
- Prediction with class probability / confidence
- Interpretation notes (key signals behind a prediction)
- Downloadable predictions

```bash
streamlit run app/streamlit_app.py
```

---

## Tech Stack

`Python` · `scikit-learn` · `pandas` / `numpy` · `Streamlit` · `matplotlib` / `seaborn` · `PyYAML` · `joblib`

See `requirements.txt` for pinned versions.

---

## Known Limitations & Next Steps

- Model comparison uses a single train/test split; k-fold cross-validation would give more reliable metric estimates, especially for the smaller churn dataset.
- No automated test suite or CI pipeline yet.
- Movie genre classification would benefit from a more expressive text representation (e.g. embeddings) given the ceiling TF-IDF + linear models hit on minority classes.

---

## License

MIT License. See [LICENSE](LICENSE) for details.