import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import MetricFrame
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"
SENSITIVE_COLUMN = "location"

REPORT_DIR = Path("reports/week9")
MODEL_DIR = Path("models/week9")
DATA_DIR = Path("data/week9")

EXPERIMENT_NAME = "week_9_explainability_fairness_drift"


def precision_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def load_iris_with_location():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    target_names = dict(enumerate(iris.target_names))
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_names)

    rng = np.random.default_rng(RANDOM_STATE)
    df[SENSITIVE_COLUMN] = rng.integers(0, 2, size=len(df))

    return df, list(iris.target_names)


def train_model(train_df):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=4,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, test_df):
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_macro(y_test, y_pred),
        "recall_macro": recall_macro(y_test, y_pred),
        "f1_macro": f1_macro(y_test, y_pred),
    }

    predictions_df = test_df.copy()
    predictions_df["prediction"] = y_pred

    return metrics, predictions_df


def run_fairness_audit(predictions_df):
    y_true = predictions_df[TARGET_COLUMN]
    y_pred = predictions_df["prediction"]
    sensitive_features = predictions_df[SENSITIVE_COLUMN]

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    by_group_df = metric_frame.by_group.reset_index()
    by_group_df = by_group_df.rename(columns={SENSITIVE_COLUMN: "location"})

    difference = metric_frame.difference(method="between_groups")
    ratio = metric_frame.ratio(method="between_groups")

    fairness_summary = {
        "overall": metric_frame.overall.to_dict(),
        "difference_between_groups": difference.to_dict(),
        "ratio_between_groups": ratio.to_dict(),
    }

    return by_group_df, fairness_summary


def extract_shap_values_for_class(shap_values, class_index):
    """
    Handles both common SHAP formats:
    1. list[class_index] -> array(n_samples, n_features)
    2. array(n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        return shap_values[class_index]

    values = np.asarray(shap_values)

    if values.ndim == 3:
        return values[:, :, class_index]

    if values.ndim == 2:
        return values

    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def generate_shap_plots(model, X, class_names):
    shap_dir = REPORT_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_summary = {}

    for class_index, class_name in enumerate(class_names):
        class_shap_values = extract_shap_values_for_class(shap_values, class_index)

        plt.figure()
        shap.summary_plot(
            class_shap_values,
            X,
            feature_names=FEATURE_COLUMNS,
            show=False,
            plot_type="dot",
        )
        plt.title(f"SHAP Summary Plot - {class_name}")
        output_path = shap_dir / f"shap_summary_{class_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        mean_abs = np.abs(class_shap_values).mean(axis=0)
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "mean_absolute_shap": mean_abs,
                    "class_name": class_name,
                }
            )
            .sort_values("mean_absolute_shap", ascending=False)
            .reset_index(drop=True)
        )

        feature_importance.to_csv(
            shap_dir / f"shap_importance_{class_name}.csv",
            index=False,
        )

        shap_summary[class_name] = feature_importance.to_dict(orient="records")

    return shap_summary


def simulate_production_drift(test_df):
    production_df = test_df.copy()

    # Simulate data drift: production flowers have shifted petal measurements.
    production_df["petal_length"] = production_df["petal_length"] + 1.2
    production_df["petal_width"] = production_df["petal_width"] + 0.4

    # Small sepal drift to make the comparison visible but less severe.
    production_df["sepal_length"] = production_df["sepal_length"] + 0.2

    return production_df


def run_drift_detection(reference_df, production_df):
    drift_rows = []

    for feature in FEATURE_COLUMNS:
        statistic, p_value = ks_2samp(reference_df[feature], production_df[feature])
        drift_detected = bool(p_value < 0.05)

        drift_rows.append(
            {
                "feature": feature,
                "ks_statistic": statistic,
                "p_value": p_value,
                "drift_detected_p_lt_0_05": drift_detected,
                "reference_mean": reference_df[feature].mean(),
                "production_mean": production_df[feature].mean(),
                "mean_shift": production_df[feature].mean() - reference_df[feature].mean(),
            }
        )

    drift_df = pd.DataFrame(drift_rows)
    return drift_df


def save_drift_plots(reference_df, production_df):
    drift_dir = REPORT_DIR / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for feature in FEATURE_COLUMNS:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(reference_df[feature], bins=15, alpha=0.6, label="training/reference")
        ax.hist(production_df[feature], bins=15, alpha=0.6, label="simulated production")
        ax.set_title(f"Data Drift Check - {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("count")
        ax.legend()

        output_path = drift_dir / f"drift_distribution_{feature}.png"
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

        output_paths.append(output_path)

    return output_paths


def write_model_card(metrics, fairness_by_group_df, fairness_summary, drift_df, shap_summary):
    model_card_path = REPORT_DIR / "model_card_iris_classifier.md"

    virginica_features = shap_summary["virginica"][:4]

    card = f"""# Model Card: IRIS Classifier

## Model Details

- Model name: IRIS RandomForestClassifier
- Assignment: Week 9 Explainability, Fairness, and Drift
- Model type: Random Forest classifier
- Target: IRIS species
- Classes: setosa, versicolor, virginica
- Training features: sepal_length, sepal_width, petal_length, petal_width
- Sensitive attribute used for audit: location
- Sensitive attribute included in training: No

## Intended Use

This model is intended for educational demonstration of explainability, fairness auditing, and drift detection in an MLOps pipeline. It predicts the IRIS flower species from four numeric flower measurements.

## Training Data

The model uses the sklearn IRIS dataset. A synthetic `location` column is randomly assigned with values 0 and 1. The location column is not used as a model input. It is used only to audit whether performance differs across groups.

## Overall Performance

| Metric | Value |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision Macro | {metrics["precision_macro"]:.4f} |
| Recall Macro | {metrics["recall_macro"]:.4f} |
| F1 Macro | {metrics["f1_macro"]:.4f} |

## Fairness Audit by Location

{fairness_by_group_df.to_markdown(index=False)}

## Fairness Summary

- Accuracy difference between location groups: {fairness_summary["difference_between_groups"]["accuracy"]:.4f}
- Precision macro difference between location groups: {fairness_summary["difference_between_groups"]["precision_macro"]:.4f}
- Recall macro difference between location groups: {fairness_summary["difference_between_groups"]["recall_macro"]:.4f}
- F1 macro difference between location groups: {fairness_summary["difference_between_groups"]["f1_macro"]:.4f}

Because location is randomly assigned and excluded from training, large fairness gaps are not expected. However, this workflow demonstrates how fairness audits can be added to a production ML pipeline.

## Explainability Summary for Virginica

The SHAP summary plot for virginica explains which feature values push predictions toward or away from the virginica class.

Top SHAP features for virginica in this run:

{pd.DataFrame(virginica_features).to_markdown(index=False)}

Plain-language interpretation:

- Points on the right side of the SHAP plot push the model toward predicting virginica.
- Points on the left side push the model away from predicting virginica.
- Red points indicate high feature values.
- Blue points indicate low feature values.
- If high petal length or high petal width values appear on the right, that means those high values are strong evidence for virginica.

## Drift Analysis

The production dataset was simulated by shifting petal_length, petal_width, and sepal_length. The Kolmogorov-Smirnov test was used to compare reference training distributions with simulated production distributions.

{drift_df.to_markdown(index=False)}

Features marked as drifted have distributions that differ significantly from the training/reference data at p < 0.05.

## Limitations

- IRIS is a small educational dataset.
- The sensitive attribute `location` is synthetic and randomly assigned.
- Fairness results are therefore illustrative, not a real demographic audit.
- The production drift is simulated, not collected from a real deployment.
- The model should not be used for real-world biological or business decisions.

## Governance Notes

Before deploying a model like this in production, the team should maintain:

- Data schema validation
- Sensitive group performance monitoring
- SHAP or other explainability reports
- Data drift and concept drift monitoring
- Model cards for accountability
- Approval workflow before promotion to production
"""

    model_card_path.write_text(card)
    return model_card_path


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df, class_names = load_iris_with_location()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(DATA_DIR / "iris_train_with_location.csv", index=False)
    test_df.to_csv(DATA_DIR / "iris_test_with_location.csv", index=False)

    model = train_model(train_df)
    metrics, predictions_df = evaluate_model(model, test_df)

    predictions_df.to_csv(REPORT_DIR / "predictions_with_location.csv", index=False)

    fairness_by_group_df, fairness_summary = run_fairness_audit(predictions_df)
    fairness_by_group_df.to_csv(REPORT_DIR / "fairness_by_location.csv", index=False)

    with open(REPORT_DIR / "fairness_summary.json", "w") as f:
        json.dump(fairness_summary, f, indent=4)

    X_test = test_df[FEATURE_COLUMNS]
    shap_summary = generate_shap_plots(model, X_test, class_names)

    with open(REPORT_DIR / "shap_summary.json", "w") as f:
        json.dump(shap_summary, f, indent=4)

    production_df = simulate_production_drift(test_df)
    production_df.to_csv(DATA_DIR / "iris_simulated_production_drift.csv", index=False)

    drift_df = run_drift_detection(train_df, production_df)
    drift_df.to_csv(REPORT_DIR / "drift_report.csv", index=False)

    drift_plot_paths = save_drift_plots(train_df, production_df)

    model_path = MODEL_DIR / "iris_week9_random_forest.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "sensitive_column": SENSITIVE_COLUMN,
        },
        model_path,
    )

    model_card_path = write_model_card(
        metrics=metrics,
        fairness_by_group_df=fairness_by_group_df,
        fairness_summary=fairness_summary,
        drift_df=drift_df,
        shap_summary=shap_summary,
    )

    with open(REPORT_DIR / "overall_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="iris_fairness_shap_drift"):
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 150,
                "max_depth": 4,
                "sensitive_attribute": SENSITIVE_COLUMN,
                "sensitive_attribute_used_for_training": False,
                "drift_simulation": "petal_length_plus_1.2_petal_width_plus_0.4_sepal_length_plus_0.2",
                "random_state": RANDOM_STATE,
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(REPORT_DIR / "overall_metrics.json"))
        mlflow.log_artifact(str(REPORT_DIR / "fairness_by_location.csv"))
        mlflow.log_artifact(str(REPORT_DIR / "fairness_summary.json"))
        mlflow.log_artifact(str(REPORT_DIR / "drift_report.csv"))
        mlflow.log_artifact(str(REPORT_DIR / "shap_summary.json"))
        mlflow.log_artifact(str(model_card_path))
        mlflow.log_artifact(str(model_path))

        for path in (REPORT_DIR / "shap").glob("*.png"):
            mlflow.log_artifact(str(path), artifact_path="shap_plots")

        for path in drift_plot_paths:
            mlflow.log_artifact(str(path), artifact_path="drift_plots")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(5),
        )

    print("\nOverall metrics:")
    print(json.dumps(metrics, indent=4))

    print("\nFairness by location:")
    print(fairness_by_group_df)

    print("\nFairness summary:")
    print(json.dumps(fairness_summary, indent=4))

    print("\nDrift report:")
    print(drift_df)

    print(f"\nModel card written to: {model_card_path}")
    print("Week 9 pipeline completed successfully.")


if __name__ == "__main__":
    main()
