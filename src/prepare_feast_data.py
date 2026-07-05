from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "feast_data" / "iris_data_adapted_for_feast.csv"
OUTPUT_PARQUET = ROOT_DIR / "feast_data" / "iris_features.parquet"


def main():
    df = pd.read_csv(INPUT_CSV)

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["created_timestamp"] = pd.to_datetime(df["created_timestamp"])

    df["iris_id"] = df["iris_id"].astype("int64")
    df["sepal_length"] = df["sepal_length"].astype("float32")
    df["sepal_width"] = df["sepal_width"].astype("float32")
    df["petal_length"] = df["petal_length"].astype("float32")
    df["petal_width"] = df["petal_width"].astype("float32")

    df = df.sort_values(["iris_id", "event_timestamp"]).reset_index(drop=True)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Prepared Feast parquet file: {OUTPUT_PARQUET}")
    print(df.head())
    print(df.dtypes)


if __name__ == "__main__":
    main()
