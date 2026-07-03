import sys
import pandas as pd


TRAIN_PATH = "data/train.csv"


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python src/augment_data.py <number_of_rows_to_add>")

    num_rows = int(sys.argv[1])

    train_df = pd.read_csv(TRAIN_PATH)

    sampled_rows = train_df.sample(
        n=num_rows,
        replace=True,
        random_state=42 + num_rows
    )

    augmented_df = pd.concat([train_df, sampled_rows], ignore_index=True)

    augmented_df.to_csv(TRAIN_PATH, index=False)

    print(f"Added {num_rows} rows to training data.")
    print(f"New training data size: {len(augmented_df)}")


if __name__ == "__main__":
    main()
