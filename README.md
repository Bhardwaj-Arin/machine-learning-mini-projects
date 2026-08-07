# Codsoft Machine Learning Projects

This repository contains three polished machine learning projects:

- Customer Churn Prediction
- Spam SMS Detection
- Movie Genre Classification

Each project has been reorganized into a professional structure with modular Python code, configuration files, documentation, saved models, reproducible outputs, and one unified Streamlit app.

## Repository Structure

```text
codsoft_professional_projects/
|-- customer_churn/
|-- spam_sms_detection/
|-- movie_genre_classification/
|-- app/
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
cd ..
streamlit run app/streamlit_app.py
```

Use the sidebar in the app to switch between Customer Churn, Spam SMS Detection, and Movie Genre Classification.

## What Changed

- Clean folder structure for data, notebooks, source code, models, outputs, apps, docs, images, and reports.
- Modular training, prediction, preprocessing, feature engineering, evaluation, and utility files.
- YAML configuration for reproducible training parameters.
- Logging-based training scripts.
- Model comparison leaderboards and saved evaluation outputs.
- Detailed documentation for business problem, architecture, data dictionary, deployment, and user guide.
- One unified Streamlit app with sidebar navigation for all three projects.
- Dashboard views with probabilities, confidence indicators, interpretation notes, and downloadable predictions.

## License

MIT License.
