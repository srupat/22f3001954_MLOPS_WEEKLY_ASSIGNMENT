import os

import pandas as pd


TRAIN_PATH = "data/train.csv"
EVAL_PATH = "data/eval.csv"

EXPECTED_COLUMNS = {
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
}

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def load_dataset(path):
    assert os.path.exists(path), f"Dataset not found: {path}"
    return pd.read_csv(path)


def test_train_and_eval_files_exist():
    assert os.path.exists(TRAIN_PATH)
    assert os.path.exists(EVAL_PATH)


def test_train_schema_is_valid():
    df = load_dataset(TRAIN_PATH)
    assert set(df.columns) == EXPECTED_COLUMNS


def test_eval_schema_is_valid():
    df = load_dataset(EVAL_PATH)
    assert set(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values():
    train_df = load_dataset(TRAIN_PATH)
    eval_df = load_dataset(EVAL_PATH)

    assert train_df.isnull().sum().sum() == 0
    assert eval_df.isnull().sum().sum() == 0


def test_feature_columns_are_numeric():
    train_df = load_dataset(TRAIN_PATH)
    eval_df = load_dataset(EVAL_PATH)

    for col in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(train_df[col]), f"{col} is not numeric in train data"
        assert pd.api.types.is_numeric_dtype(eval_df[col]), f"{col} is not numeric in eval data"


def test_species_values_are_valid():
    valid_species = {"setosa", "versicolor", "virginica"}

    train_df = load_dataset(TRAIN_PATH)
    eval_df = load_dataset(EVAL_PATH)

    assert set(train_df["species"].unique()).issubset(valid_species)
    assert set(eval_df["species"].unique()).issubset(valid_species)


def test_feature_value_ranges_are_reasonable():
    train_df = load_dataset(TRAIN_PATH)
    eval_df = load_dataset(EVAL_PATH)

    combined_df = pd.concat([train_df, eval_df], ignore_index=True)

    assert combined_df["sepal_length"].between(3.0, 9.0).all()
    assert combined_df["sepal_width"].between(1.0, 6.0).all()
    assert combined_df["petal_length"].between(0.5, 8.0).all()
    assert combined_df["petal_width"].between(0.0, 4.0).all()
