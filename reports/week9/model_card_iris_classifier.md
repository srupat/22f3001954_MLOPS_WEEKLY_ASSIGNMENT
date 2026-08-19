# Model Card: IRIS Classifier

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
| Accuracy | 0.9333 |
| Precision Macro | 0.9333 |
| Recall Macro | 0.9333 |
| F1 Macro | 0.9333 |

## Fairness Audit by Location

|   location |   accuracy |   precision_macro |   recall_macro |   f1_macro |
|-----------:|-----------:|------------------:|---------------:|-----------:|
|          0 |   0.846154 |          0.877778 |       0.877778 |   0.877778 |
|          1 |   1        |          1        |       1        |   1        |

## Fairness Summary

- Accuracy difference between location groups: 0.1538
- Precision macro difference between location groups: 0.1222
- Recall macro difference between location groups: 0.1222
- F1 macro difference between location groups: 0.1222

Because location is randomly assigned and excluded from training, large fairness gaps are not expected. However, this workflow demonstrates how fairness audits can be added to a production ML pipeline.

## Explainability Summary for Virginica

The SHAP summary plot for virginica explains which feature values push predictions toward or away from the virginica class.

Top SHAP features for virginica in this run:

| feature      |   mean_absolute_shap | class_name   |
|:-------------|---------------------:|:-------------|
| petal_length |           0.212281   | virginica    |
| petal_width  |           0.174864   | virginica    |
| sepal_length |           0.0396654  | virginica    |
| sepal_width  |           0.00387807 | virginica    |

Plain-language interpretation:

- Points on the right side of the SHAP plot push the model toward predicting virginica.
- Points on the left side push the model away from predicting virginica.
- Red points indicate high feature values.
- Blue points indicate low feature values.
- If high petal length or high petal width values appear on the right, that means those high values are strong evidence for virginica.

## Drift Analysis

The production dataset was simulated by shifting petal_length, petal_width, and sepal_length. The Kolmogorov-Smirnov test was used to compare reference training distributions with simulated production distributions.

| feature      |   ks_statistic |     p_value | drift_detected_p_lt_0_05   |   reference_mean |   production_mean |   mean_shift |
|:-------------|---------------:|------------:|:---------------------------|-----------------:|------------------:|-------------:|
| sepal_length |       0.183333 | 0.370345    | False                      |          5.84167 |           6.05    |     0.208333 |
| sepal_width  |       0.183333 | 0.370345    | False                      |          3.04833 |           3.09333 |     0.045    |
| petal_length |       0.425    | 0.000230205 | True                       |          3.77    |           4.91    |     1.14     |
| petal_width  |       0.333333 | 0.00793551  | True                       |          1.205   |           1.57667 |     0.371667 |

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
