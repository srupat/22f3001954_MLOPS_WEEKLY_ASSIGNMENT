import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]

EVAL_PATH = ROOT_DIR / "data" / "eval.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "iris-classifier"
MODEL_ALIAS = "champion"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    print(f"Loading model from MLflow registry: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    eval_df = pd.read_csv(EVAL_PATH)
    X_eval = eval_df[FEATURE_COLUMNS]
    y_true = eval_df[TARGET_COLUMN]

    y_pred = model.predict(X_eval)

    metrics = {
        "model_uri": model_uri,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "num_eval_rows": len(eval_df),
    }

    client = MlflowClient()
    champion_version = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    metrics["registered_model_name"] = REGISTERED_MODEL_NAME
    metrics["resolved_model_version"] = champion_version.version
    metrics["resolved_run_id"] = champion_version.run_id

    metrics_path = OUTPUT_DIR / "mlflow_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation completed")
    print(json.dumps(metrics, indent=4))
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
