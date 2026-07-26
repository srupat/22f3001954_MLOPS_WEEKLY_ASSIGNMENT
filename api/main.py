from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


app = FastAPI(title="IRIS Classifier API", version="1.0.0")


class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


class BatchIrisInput(BaseModel):
    samples: List[IrisInput]


model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "IRIS Classifier API is running",
        "model_path": str(MODEL_PATH),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: IrisInput):
    input_df = pd.DataFrame([data.model_dump()])[FEATURE_COLUMNS]
    prediction = model.predict(input_df)[0]

    return {
        "prediction": str(prediction),
        "input": data.model_dump(),
    }


@app.post("/predict-batch")
def predict_batch(data: BatchIrisInput):
    input_df = pd.DataFrame([sample.model_dump() for sample in data.samples])[FEATURE_COLUMNS]
    predictions = model.predict(input_df)

    return {
        "predictions": [str(pred) for pred in predictions],
        "count": len(predictions),
    }
