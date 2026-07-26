from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


ROOT_DIR = Path(__file__).resolve().parents[1]

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "iris-classifier"
MODEL_ALIAS = "champion"

OUTPUT_MODEL_PATH = ROOT_DIR / "api" / "model.joblib"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from MLflow registry: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_MODEL_PATH)

    print(f"Exported MLflow champion model to: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
