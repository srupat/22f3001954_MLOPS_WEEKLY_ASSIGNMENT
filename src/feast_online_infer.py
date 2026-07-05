import sys
from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "feast_data" / "iris_data_adapted_for_feast.csv"
MODEL_PATH = ROOT_DIR / "models" / "iris_feast_model.joblib"
OUTPUT_PATH = ROOT_DIR / "outputs" / "feast_online_predictions.csv"
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        iris_ids = [int(x) for x in sys.argv[1:]]
    else:
        iris_ids = [1001, 1002, 1003]

    store = FeatureStore(repo_path=str(FEATURE_REPO_PATH))

    entity_rows = [{"iris_id": iris_id} for iris_id in iris_ids]

    online_features = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=entity_rows,
    ).to_df()

    model = joblib.load(MODEL_PATH)

    online_features = online_features.sort_values("iris_id").reset_index(drop=True)

    X_online = online_features[FEATURE_COLUMNS]
    online_predictions = model.predict(X_online)

    result_df = online_features.copy()
    result_df["online_prediction"] = online_predictions

    raw_df = pd.read_csv(CSV_PATH)
    raw_df["event_timestamp"] = pd.to_datetime(raw_df["event_timestamp"])

    latest_raw_df = (
        raw_df[raw_df["iris_id"].isin(iris_ids)]
        .sort_values(["iris_id", "event_timestamp"])
        .groupby("iris_id")
        .tail(1)
        .sort_values("iris_id")
        .reset_index(drop=True)
    )

    raw_predictions = model.predict(latest_raw_df[FEATURE_COLUMNS])

    comparison_df = latest_raw_df[
        ["iris_id", "event_timestamp", "species"] + FEATURE_COLUMNS
    ].copy()
    comparison_df["raw_latest_prediction"] = raw_predictions

    final_df = result_df.merge(
        comparison_df,
        on="iris_id",
        suffixes=("_online", "_raw_latest"),
    )

    final_df["prediction_match"] = (
        final_df["online_prediction"] == final_df["raw_latest_prediction"]
    )

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Online inference completed using Feast online store.")
    print(f"Predictions saved to: {OUTPUT_PATH}")
    print(final_df)


if __name__ == "__main__":
    main()
