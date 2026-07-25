import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "train.csv"
EVAL_PATH = ROOT_DIR / "data" / "eval.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
EXPERIMENT_NAME = "iris-mlflow-experiments"
REGISTERED_MODEL_NAME = "iris-classifier"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_eval = eval_df[FEATURE_COLUMNS]
    y_eval = eval_df[TARGET_COLUMN]

    return X_train, y_train, X_eval, y_eval


def evaluate_model(model, X_eval, y_eval):
    y_pred = model.predict(X_eval)

    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision_macro": precision_score(y_eval, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_eval, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_eval, y_pred, average="macro", zero_division=0),
    }

    return metrics, y_pred


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_eval, y_eval = load_data()

    # At least two hyperparameters are varied:
    # n_estimators and max_depth. criterion is also varied.
    param_grid = [
        {"n_estimators": 50, "max_depth": 2, "criterion": "gini", "random_state": 42},
        {"n_estimators": 100, "max_depth": 3, "criterion": "gini", "random_state": 42},
        {"n_estimators": 150, "max_depth": 4, "criterion": "entropy", "random_state": 42},
        {"n_estimators": 200, "max_depth": None, "criterion": "gini", "random_state": 42},
    ]

    best_run_id = None
    best_metrics = None
    best_params = None
    best_model_uri = None
    best_accuracy = -1.0

    all_results = []

    for params in param_grid:
        with mlflow.start_run(run_name=f"rf_{params['n_estimators']}_{params['max_depth']}_{params['criterion']}") as run:
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            metrics, y_pred = evaluate_model(model, X_eval, y_eval)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            mlflow.set_tag("model_type", "RandomForestClassifier")
            mlflow.set_tag("dataset", "IRIS")
            mlflow.set_tag("assignment_week", "week_5")

            signature = infer_signature(X_eval, y_pred)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_eval.head(3),
            )

            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"

            result = {
                "run_id": run_id,
                "model_uri": model_uri,
                "params": params,
                "metrics": metrics,
            }

            all_results.append(result)

            print("Run completed")
            print(f"Run ID: {run_id}")
            print(f"Params: {params}")
            print(f"Metrics: {metrics}")
            print("-" * 80)

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_run_id = run_id
                best_metrics = metrics
                best_params = params
                best_model_uri = model_uri

    client = MlflowClient()

    print("Registering best model")
    print(f"Best run ID: {best_run_id}")
    print(f"Best model URI: {best_model_uri}")
    print(f"Best params: {best_params}")
    print(f"Best metrics: {best_metrics}")

    model_version = mlflow.register_model(
        model_uri=best_model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # Set a stable alias for the best model.
    # Evaluation and inference can now load models:/iris-classifier@champion
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version.version,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=model_version.version,
        key="validation_status",
        value="approved",
    )

    summary = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "champion_version": model_version.version,
        "best_run_id": best_run_id,
        "best_model_uri": best_model_uri,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "all_results": all_results,
    }

    summary_path = OUTPUT_DIR / "mlflow_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Training summary saved to: {summary_path}")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Champion version: {model_version.version}")


if __name__ == "__main__":
    main()
