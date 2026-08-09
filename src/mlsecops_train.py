import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

DATA_DIR = Path("data/mlsecops")
REPORT_DIR = Path("reports/mlsecops")
MODEL_DIR = Path("models/mlsecops")

EXPERIMENT_NAME = "week_8_mlsecops_iris_poisoning"


def load_dataset(poisoning_percent: int):
    train_path = DATA_DIR / f"iris_train_poison_{poisoning_percent}.csv"
    test_path = DATA_DIR / "iris_test_clean.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def train_and_evaluate(poisoning_percent: int):
    train_df, test_df = load_dataset(poisoning_percent)

    X_train = train_df[FEATURE_COLUMNS]
    y_train_text = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test_text = test_df[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    return model, label_encoder, metrics, X_test, y_test, y_pred


def save_confusion_matrix(poisoning_percent: int, label_encoder, y_test, y_pred):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax)
    ax.set_title(f"IRIS Confusion Matrix - Poisoning {poisoning_percent}%")

    output_path = REPORT_DIR / f"confusion_matrix_poison_{poisoning_percent}.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def save_model(poisoning_percent: int, model, label_encoder):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }

    model_path = MODEL_DIR / f"iris_model_poison_{poisoning_percent}.joblib"
    joblib.dump(artifact, model_path)

    return model_path


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    poisoning_percents = [0, 5, 10, 50]
    all_results = []

    for poisoning_percent in poisoning_percents:
        poisoning_level = poisoning_percent / 100

        with mlflow.start_run(run_name=f"iris_poison_{poisoning_percent}_percent"):
            model, label_encoder, metrics, X_test, y_test, y_pred = train_and_evaluate(
                poisoning_percent=poisoning_percent
            )

            confusion_matrix_path = save_confusion_matrix(
                poisoning_percent=poisoning_percent,
                label_encoder=label_encoder,
                y_test=y_test,
                y_pred=y_pred,
            )

            model_path = save_model(
                poisoning_percent=poisoning_percent,
                model=model,
                label_encoder=label_encoder,
            )

            summary_path = DATA_DIR / f"poison_summary_{poisoning_percent}.json"
            with open(summary_path) as f:
                poison_summary = json.load(f)

            mlflow.log_param("poisoning_level", poisoning_level)
            mlflow.log_param("poisoning_percent", poisoning_percent)
            mlflow.log_param("num_poisoned_rows", poison_summary["num_poisoned_rows"])
            mlflow.log_param("num_train_rows", poison_summary["num_train_rows"])
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 4)
            mlflow.log_param("random_state", 42)

            mlflow.log_metrics(metrics)

            mlflow.log_artifact(str(confusion_matrix_path))
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(model_path))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_test.head(5),
            )

            result = {
                "poisoning_percent": poisoning_percent,
                "poisoning_level": poisoning_level,
                "num_poisoned_rows": poison_summary["num_poisoned_rows"],
                **metrics,
            }

            all_results.append(result)

            print(json.dumps(result, indent=4))

    results_df = pd.DataFrame(all_results)
    results_csv = REPORT_DIR / "mlsecops_poisoning_results.csv"
    results_json = REPORT_DIR / "mlsecops_poisoning_results.json"

    results_df.to_csv(results_csv, index=False)

    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nFinal MLSecOps poisoning results:")
    print(results_df)
    print(f"\nSaved results to {results_csv} and {results_json}")


if __name__ == "__main__":
    main()
