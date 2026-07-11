import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score


MODEL_PATH = "models/iris_model.joblib"
EVAL_PATH = "data/eval.csv"
REPORT_PATH = "reports/metrics.json"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

MIN_ACCURACY = 0.80
MIN_PRECISION = 0.80


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_accuracy_and_precision_thresholds():
    os.makedirs("reports", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    X_eval = eval_df[FEATURE_COLUMNS]
    y_true = eval_df[TARGET_COLUMN]

    y_pred = model.predict(X_eval)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "min_accuracy_required": MIN_ACCURACY,
        "min_precision_required": MIN_PRECISION,
        "num_eval_rows": int(len(eval_df)),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    assert accuracy >= MIN_ACCURACY, f"Accuracy too low: {accuracy}"
    assert precision >= MIN_PRECISION, f"Precision too low: {precision}"
