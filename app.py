from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI

app = FastAPI()

model = None


@app.get("/")
def home():
    return {"message": "Churn Prediction API running"}


@app.post("/predict")
def predict(
    age: int,
    monthly_charges: float,
    contract_length: int,
    support_calls: int,
):
    global model
    if model is None:
        model = joblib.load(Path(__file__).with_name("model.pkl"))

    data = np.array([[age, monthly_charges, contract_length, support_calls]])
    prediction = model.predict(data)[0]
    return {"churn_prediction": int(prediction)}
