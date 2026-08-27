# Churn Prediction FastAPI

A small end-to-end machine learning deployment example reconstructed from the presentation **Deploying a simple ML model using FastAPI**.

The project trains a `RandomForestClassifier`, saves it with Joblib, and serves predictions through FastAPI.

## Project structure

```text
.
├── app.py
├── train.py
├── model.pkl
├── requirements.txt
├── Procfile
└── .gitignore
```

## Setup

```bash
uv sync
```

Format and lint the project:

```bash
uv run ruff format .
uv run ruff check .
```

## Train and save the model

```bash
python train.py
```

This creates `model.pkl`.

## Run locally

```bash
uvicorn app:app --reload
```

Open:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Prediction endpoint

`POST /predict`

Example:

```bash
curl -X POST "http://127.0.0.1:98000/predict?age=45&monthly_charges=80&contract_length=24&support_calls=3"
```

Example response:

```json
{"churn_prediction": 1}
```

## Render deployment

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```
