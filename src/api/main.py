import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic_models import CustomerData
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

MODEL_NAME = "GradientBoosting"
MODEL_PATH = f"../../models/{MODEL_NAME}_best_model.pkl"

# Load the model once at startup
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")  # if MLflow registry used
# Or fallback to local:
import joblib
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict_risk(customer: CustomerData):
    # Convert Pydantic model to dataframe
    data_dict = customer.dict()
    df = pd.DataFrame([data_dict])
    try:
        proba = model.predict_proba(df)[:, 1][0]
        return {"risk_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
