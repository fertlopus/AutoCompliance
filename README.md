# Automatic Compliance Service

---


## Overview
This project is an email classification system that categorizes emails into different categories based on their content. It uses Natural Language Processing (NLP) techniques and a machine learning model to automate the classification process.

## Features
- Email text classification using a Logistic Regression model.
- REST API built with FastAPI for interacting with the model.
- Gradio UI for easy interaction with the model through a web interface.
- MLflow for tracking experiments and model management.

## Installation
To set up the project environment, follow these steps:

```bash
git clone https://github.com/your-repo/email-classification.git
cd email-classification
pip install -r requirements.txt
```

---

Usage:

Starting the FastAPI Server

Run the following command to start the FastAPI server:

```bash
uvicorn api.app:app --reload
```

Using the Gradio Interface

```bash
python ./frontend/ui_classification.py
```

Making Predictions

Make predictions using the REST API:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Sample email text"}'
```

### Project Structure

* `src/`: Core source code for the project.
* `api/`: FastAPI application files.
* `frontend/`: Gradio interface files.
* `model_artifacts/`: Saved models and vectorizers.
* `mlruns/`: MLflow tracking and experiment data.

