# Week 11 LLM Governance Guardrails Report

## Objective

This report evaluates the Week 10 fine-tuned Gemini IRIS pipeline against prompt injection and prompt leakage risks.
It compares unguarded behavior with a guarded pipeline using input and output guardrails.

## Guardrail Design

### Input Guardrails

- Rule-based regex detection for injection and leakage phrases.
- Structural IRIS schema validation requiring all four flower measurements.
- Numeric range validation for sepal and petal measurements.

### Output Guardrails

- Leakage scan for system/context/training-data fragments.
- Format compliance check for allowed IRIS species outputs.
- Non-compliant output is replaced with a safe fallback.

## Effectiveness Metrics

| metric                                   |   value |
|:-----------------------------------------|--------:|
| Injection block rate                     |     1   |
| Leakage block rate                       |     1   |
| False positive rate on legitimate inputs |     0   |
| Unguarded prompt injection success rate  |     1   |
| Unguarded prompt leakage success rate    |     0.3 |

## Guarded Accuracy by Model

| model_version   |   guarded_accuracy |   false_positive_rate |
|:----------------|-------------------:|----------------------:|
| v1              |           0.933333 |                     0 |
| v2              |           0.333333 |                     0 |

## Interpretation

A strong guardrail should block most adversarial prompts while allowing legitimate IRIS classification prompts.
The false positive rate is important because a guardrail that blocks everything would be secure but unusable.
The goal is high injection/leakage block rate with low false positives and minimal accuracy degradation.