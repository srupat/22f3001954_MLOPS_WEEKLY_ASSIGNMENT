import json
from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "feast_data" / "iris_data_adapted_for_feast.csv"
MODEL_PATH = ROOT_DIR / "models" / "iris_feast_model.joblib"
TRAINING_DATA_OUTPUT = ROOT_DIR / "outputs" / "feast_training_data.csv"
METRICS_OUTPUT = ROOT_DIR / "outputs" / "feast_training_metrics.json"

FEATURE_REPO_PATH = ROOT_DIR / "feature_repo"

FEATURE_REFS = [
    "iris_features:sepal_length",
    "iris_features:sepal_width",
    "iris_features:petal_length",
    "iris_features:petal_width",
]

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(CSV_PATH)
    raw_df["event_timestamp"] = pd.to_datetime(raw_df["event_timestamp"])

    entity_df = raw_df[["iris_id", "event_timestamp", "species"]].copy()

    store = FeatureStore(repo_path=str(FEATURE_REPO_PATH))

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    training_df = training_df.dropna(subset=FEATURE_COLUMNS + ["species"])

    X = training_df[FEATURE_COLUMNS]
    y = training_df["species"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X, y)

    train_predictions = model.predict(X)
    train_accuracy = accuracy_score(y, train_predictions)

    joblib.dump(model, MODEL_PATH)
    training_df.to_csv(TRAINING_DATA_OUTPUT, index=False)

    metrics = {
        "training_rows": int(len(training_df)),
        "num_classes": int(y.nunique()),
        "classes": sorted(y.unique().tolist()),
        "training_accuracy": float(train_accuracy),
        "feature_source": "Feast offline store via get_historical_features",
    }

    with open(METRICS_OUTPUT, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Training completed using Feast historical features.")
    print(f"Training data saved to: {TRAINING_DATA_OUTPUT}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_OUTPUT}")
    print(metrics)


if __name__ == "__main__":
    main()
