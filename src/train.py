import json
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


TRAIN_PATH = "data/train.csv"
EVAL_PATH = "data/eval.csv"
MODEL_PATH = "models/iris_model.joblib"
METRICS_PATH = "outputs/metrics.json"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)

    target_col = "species" if "species" in train_df.columns else train_df.columns[-1]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_eval = eval_df.drop(columns=[target_col])
    y_eval = eval_df[target_col]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_pred)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df))
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()
