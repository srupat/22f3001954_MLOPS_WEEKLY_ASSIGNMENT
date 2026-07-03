import os
import joblib
import pandas as pd


EVAL_PATH = "data/eval.csv"
MODEL_PATH = "models/iris_model.joblib"
PREDICTIONS_PATH = "outputs/predictions.csv"


def main():
    os.makedirs("outputs", exist_ok=True)

    eval_df = pd.read_csv(EVAL_PATH)

    target_col = "species" if "species" in eval_df.columns else eval_df.columns[-1]

    X_eval = eval_df.drop(columns=[target_col])
    y_true = eval_df[target_col]

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_eval)

    predictions_df = eval_df.copy()
    predictions_df["prediction"] = y_pred
    predictions_df["correct"] = predictions_df["prediction"] == y_true

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print("Inference completed.")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(predictions_df.head())


if __name__ == "__main__":
    main()
