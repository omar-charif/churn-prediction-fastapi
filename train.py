import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# sample dataset
data = {
    "age": [25, 45, 30, 50, 23, 40, 60, 48],
    "monthly_charges": [50, 80, 60, 90, 45, 70, 100, 85],
    "contract_length": [12, 24, 12, 36, 6, 24, 36, 24],
    "support_calls": [1, 3, 0, 5, 0, 2, 6, 4],
    "churn": [0, 1, 0, 1, 0, 0, 1, 1],
}

df = pd.DataFrame(data)

X = df.drop("churn", axis=1)
y = df["churn"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained & saved!")
