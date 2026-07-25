from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import accuracy_score


ROOT_DIR = Path(__file__).resolve().parents[1]

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MODEL_URI = "models:/iris-classifier@champion"

EVAL_PATH = ROOT_DIR / "data" / "eval.csv"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

MIN_ACCURACY = 0.80


def test_champion_model_can_be_loaded_from_mlflow_registry():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.pyfunc.load_model(MODEL_URI)

    assert model is not None


def test_champion_model_accuracy_threshold():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.pyfunc.load_model(MODEL_URI)

    eval_df = pd.read_csv(EVAL_PATH)

    X_eval = eval_df[FEATURE_COLUMNS]
    y_true = eval_df[TARGET_COLUMN]

    y_pred = model.predict(X_eval)

    accuracy = accuracy_score(y_true, y_pred)

    assert accuracy >= MIN_ACCURACY
