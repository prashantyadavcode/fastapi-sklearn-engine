from fastapi import FastAPI
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}