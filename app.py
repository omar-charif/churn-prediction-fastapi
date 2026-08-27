from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")


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
    data = np.array([[age, monthly_charges, contract_length, support_calls]])
    prediction = model.predict(data)[0]
    return {"churn_prediction": int(prediction)}
