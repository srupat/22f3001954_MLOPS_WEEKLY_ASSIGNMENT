import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"


def load_clean_iris(random_state: int):
    iris = load_iris(as_frame=True)

    df = iris.frame.copy()
    df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    # Convert numeric target to readable labels.
    target_names = dict(enumerate(iris.target_names))
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_names)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def poison_training_data(
    train_df: pd.DataFrame,
    poisoning_level: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(random_state)

    poisoned_df = train_df.copy()

    n_rows = len(poisoned_df)
    n_poisoned = int(round(n_rows * poisoning_level))

    if n_poisoned == 0:
        summary = {
            "poisoning_level": poisoning_level,
            "num_train_rows": n_rows,
            "num_poisoned_rows": 0,
            "poisoned_indices": [],
        }
        return poisoned_df, summary

    poisoned_indices = rng.choice(n_rows, size=n_poisoned, replace=False)

    # Generate random feature values inside broad IRIS-like ranges.
    # This simulates injected noisy samples while still looking numerically plausible.
    feature_ranges = {
        "sepal_length": (4.0, 8.5),
        "sepal_width": (1.8, 4.8),
        "petal_length": (1.0, 7.5),
        "petal_width": (0.1, 3.0),
    }

    for col in FEATURE_COLUMNS:
        low, high = feature_ranges[col]
        poisoned_df.loc[poisoned_indices, col] = rng.uniform(
            low=low,
            high=high,
            size=n_poisoned,
        )

    class_labels = sorted(train_df[TARGET_COLUMN].unique())
    poisoned_df.loc[poisoned_indices, TARGET_COLUMN] = rng.choice(
        class_labels,
        size=n_poisoned,
        replace=True,
    )

    summary = {
        "poisoning_level": poisoning_level,
        "num_train_rows": n_rows,
        "num_poisoned_rows": n_poisoned,
        "poisoned_indices": poisoned_indices.tolist(),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "class_labels": class_labels,
    }

    return poisoned_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/mlsecops")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_clean_iris(random_state=args.random_state)

    # Keep test set clean for fair evaluation.
    test_path = output_dir / "iris_test_clean.csv"
    test_df.to_csv(test_path, index=False)

    poisoning_levels = [0.0, 0.05, 0.10, 0.50]

    all_summaries = []

    for level in poisoning_levels:
        poisoned_train_df, summary = poison_training_data(
            train_df=train_df,
            poisoning_level=level,
            random_state=args.random_state + int(level * 1000),
        )

        level_name = int(level * 100)
        train_path = output_dir / f"iris_train_poison_{level_name}.csv"
        summary_path = output_dir / f"poison_summary_{level_name}.json"

        poisoned_train_df.to_csv(train_path, index=False)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        all_summaries.append(summary)

        print(
            f"Created {train_path}: poisoning_level={level}, "
            f"poisoned_rows={summary['num_poisoned_rows']}"
        )

    with open(output_dir / "all_poisoning_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=4)

    print(f"Saved clean test data to {test_path}")
    print(f"Saved poisoning summaries to {output_dir}")


if __name__ == "__main__":
    main()
