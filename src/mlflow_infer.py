import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    print(f"Loading model from MLflow registry: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    eval_df = pd.read_csv(EVAL_PATH)

    if len(sys.argv) > 1:
        row_count = int(sys.argv[1])
    else:
        row_count = 5

    input_df = eval_df[FEATURE_COLUMNS].head(row_count)
    predictions = model.predict(input_df)

    output_df = input_df.copy()
    output_df["prediction"] = predictions

    output_path = OUTPUT_DIR / "mlflow_predictions.csv"
    output_df.to_csv(output_path, index=False)

    print("Inference completed")
    print(output_df)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
