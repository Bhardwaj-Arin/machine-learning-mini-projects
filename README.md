# Codsoft Machine Learning Projects

This repository contains three polished machine learning projects:

- Customer Churn Prediction
- Spam SMS Detection
- Movie Genre Classification

Each project has been reorganized into a professional structure with modular Python code, configuration files, documentation, saved models, reproducible outputs, and Streamlit app entry points.

## Repository Structure

```text
codsoft_professional_projects/
|-- customer_churn/
|-- spam_sms_detection/
|-- movie_genre_classification/
|-- requirements.txt
|-- Makefile
|-- LICENSE
`-- README.md
```

## Quick Start

```bash
pip install -r requirements.txt
cd customer_churn
python src/train.py
streamlit run app/streamlit_app.py
```

Repeat the same workflow for `spam_sms_detection` and `movie_genre_classification`.

## What Changed

- Clean folder structure for data, notebooks, source code, models, outputs, apps, docs, images, and reports.
- Modular training, prediction, preprocessing, feature engineering, evaluation, and utility files.
- YAML configuration for reproducible training parameters.
- Logging-based training scripts.
- Model comparison leaderboards and saved evaluation outputs.
- Detailed documentation for business problem, architecture, data dictionary, deployment, and user guide.
- Streamlit apps with probabilities, confidence indicators, and downloadable predictions.

## License

MIT License.
