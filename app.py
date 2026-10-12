import os
import csv
import json
import joblib

from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Iris Neural Network API",
    description="Predict Iris flower species using a trained Neural Network",
    version="1.0.0",
)

MODEL_PATH = "artifacts/iris_model.pkl"
ENCODER_PATH = "artifacts/label_encoder.pkl"
METRICS_PATH = "artifacts/metrics.json"
LOG_PATH = "logs/predictions.csv"

model = None
encoder = None
metrics = {}

# Load the trained model and encoder
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

# Load model evaluation metrics
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)


# Input features
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {
        "message": "Iris Neural Network API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict")
def predict(data: IrisInput):
    if model is None or encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run python train.py first.",
        )

    # Prepare input features in the same order as training
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    # Neural Network prediction
    predicted_number = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    species = str(
        encoder.inverse_transform([predicted_number])[0]
    )

    probability_details = {
        str(encoder.classes_[i]): round(float(probabilities[i]), 4)
        for i in range(len(encoder.classes_))
    }

    result = {
        "predicted_species": species,
        "confidence": round(float(max(probabilities)), 4),
        "probabilities": probability_details,
    }

    # Store prediction history in CSV
    os.makedirs("logs", exist_ok=True)
    file_exists = (
        os.path.exists(LOG_PATH)
        and os.path.getsize(LOG_PATH) > 0
    )

    with open(LOG_PATH, "a", newline="") as file:
        fieldnames = [
            "timestamp",
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "predicted_species",
            "confidence",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            **data.model_dump(),
            "predicted_species": species,
            "confidence": result["confidence"],
        })

    return result