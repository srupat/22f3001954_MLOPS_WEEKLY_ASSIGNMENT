import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple


VALID_SPECIES = ["setosa", "versicolor", "virginica"]

FEATURE_RANGES = {
    "sepal_length": (3.0, 9.0),
    "sepal_width": (1.0, 6.0),
    "petal_length": (0.0, 8.5),
    "petal_width": (0.0, 4.5),
}


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str
    matched_rule: Optional[str] = None


class AuditLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: Dict):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }

        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")


class InputGuardrail:
    """
    Input guardrail with two mechanisms:
    1. Rule-based injection/leakage keyword and regex matching.
    2. Structural schema validation for IRIS measurement prompts.
    """

    def __init__(self, logger: AuditLogger):
        self.logger = logger

        self.block_patterns = [
            ("ignore_instructions", r"\bignore\b.*\b(instruction|instructions|previous|above)\b"),
            ("system_prompt_request", r"\b(system prompt|developer message|hidden instruction|initial instruction)\b"),
            ("context_window_request", r"\b(context window|everything above|repeat above|print context)\b"),
            ("roleplay_override", r"\b(you are now|act as|roleplay as|pretend to be)\b"),
            ("jailbreak_marker", r"\b(jailbreak|do anything now|dan mode|unrestricted mode)\b"),
            ("delimiter_escape", r"(```|###|<system>|</system>|<developer>|</developer>)"),
            ("task_override", r"\b(answer this instead|new task|forget the iris task|general assistant)\b"),
            ("training_data_probe", r"\b(training examples|few-shot examples|tuning dataset|fine[- ]tuning data)\b"),
        ]

    def validate(self, raw_input: str, representation: str) -> GuardrailResult:
        lowered = raw_input.lower()

        for rule_name, pattern in self.block_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
                result = GuardrailResult(
                    blocked=True,
                    reason="Input blocked due to prompt injection or leakage pattern.",
                    matched_rule=rule_name,
                )
                self.logger.log(
                    "input_blocked",
                    {
                        "representation": representation,
                        "reason": result.reason,
                        "matched_rule": rule_name,
                        "raw_input": raw_input,
                    },
                )
                return result

        structural_ok, structural_reason = self._validate_iris_schema(
            raw_input=raw_input,
            representation=representation,
        )

        if not structural_ok:
            result = GuardrailResult(
                blocked=True,
                reason=structural_reason,
                matched_rule="iris_schema_validation",
            )
            self.logger.log(
                "input_blocked",
                {
                    "representation": representation,
                    "reason": result.reason,
                    "matched_rule": "iris_schema_validation",
                    "raw_input": raw_input,
                },
            )
            return result

        return GuardrailResult(blocked=False, reason="Input passed guardrails.")

    def _validate_iris_schema(self, raw_input: str, representation: str) -> Tuple[bool, str]:
        features = extract_iris_features(raw_input)

        missing = [feature for feature in FEATURE_RANGES if feature not in features]
        if missing:
            return False, f"Input blocked because required IRIS features are missing: {missing}"

        for feature, value in features.items():
            low, high = FEATURE_RANGES[feature]
            if value < low or value > high:
                return (
                    False,
                    f"Input blocked because {feature}={value} is outside expected range [{low}, {high}].",
                )

        if representation == "v1_raw":
            required_tokens = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]
            if not all(token in raw_input.lower() for token in required_tokens):
                return False, "Input blocked because v1 raw prompt does not contain required feature keys."

        elif representation == "v2_description":
            required_tokens = [
                "sepal length",
                "sepal width",
                "petal length",
                "petal width",
            ]
            if not all(token in raw_input.lower() for token in required_tokens):
                return False, "Input blocked because v2 description prompt does not contain required feature phrases."

        else:
            return False, f"Unknown representation: {representation}"

        return True, "IRIS schema valid."


class OutputGuardrail:
    """
    Output guardrail checks:
    1. Context leakage indicators.
    2. Format compliance for v1 and v2 output formats.
    """

    def __init__(self, logger: AuditLogger):
        self.logger = logger

        self.leakage_patterns = [
            ("system_prompt_leakage", r"\b(system prompt|developer message|hidden instruction|initial instruction)\b"),
            ("context_leakage", r"\b(context window|everything above|conversation context|internal context)\b"),
            ("training_example_leakage", r"\b(training example|few-shot|tuning dataset|fine[- ]tuning data)\b"),
            ("json_context_leakage", r"\b(contents|parts|role|input_text|output_text)\b"),
            ("instruction_leakage", r"\b(classify the iris species using these measurements|respond with exactly one word)\b"),
        ]

    def filter(self, raw_output: Optional[str], representation: str) -> Dict:
        if raw_output is None:
            return self._block_output(
                raw_output="",
                representation=representation,
                reason="Output was empty or model call failed.",
                matched_rule="empty_output",
            )

        lowered = raw_output.lower()

        for rule_name, pattern in self.leakage_patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
                return self._block_output(
                    raw_output=raw_output,
                    representation=representation,
                    reason="Output blocked due to possible prompt/context leakage.",
                    matched_rule=rule_name,
                )

        if not is_format_compliant(raw_output, representation):
            return self._block_output(
                raw_output=raw_output,
                representation=representation,
                reason="Output blocked due to format violation.",
                matched_rule="format_violation",
            )

        return {
            "filtered": False,
            "safe_response": raw_output.strip(),
            "reason": "Output passed guardrails.",
            "matched_rule": None,
        }

    def _block_output(self, raw_output: str, representation: str, reason: str, matched_rule: str) -> Dict:
        fallback = "Blocked by output guardrail: response was not compliant with the allowed IRIS classification format."

        self.logger.log(
            "output_filtered",
            {
                "representation": representation,
                "reason": reason,
                "matched_rule": matched_rule,
                "raw_output": raw_output,
                "safe_response": fallback,
            },
        )

        return {
            "filtered": True,
            "safe_response": fallback,
            "reason": reason,
            "matched_rule": matched_rule,
        }


def extract_iris_features(raw_input: str) -> Dict[str, float]:
    text = raw_input.lower()

    patterns = {
        "sepal_length": [
            r"sepal_length\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            r"sepal length of\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "sepal_width": [
            r"sepal_width\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            r"sepal width of\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "petal_length": [
            r"petal_length\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            r"petal length of\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "petal_width": [
            r"petal_width\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            r"petal width of\s*([0-9]+(?:\.[0-9]+)?)",
        ],
    }

    features = {}

    for feature, feature_patterns in patterns.items():
        for pattern in feature_patterns:
            match = re.search(pattern, text)
            if match:
                features[feature] = float(match.group(1))
                break

    return features


def normalize_species(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    cleaned = text.strip().lower()

    for species in VALID_SPECIES:
        if re.search(rf"\b{species}\b", cleaned):
            return species

    return None


def is_format_compliant(text: Optional[str], representation: str) -> bool:
    if text is None:
        return False

    cleaned = text.strip().lower()

    if representation == "v1_raw":
        return cleaned in VALID_SPECIES

    if representation == "v2_description":
        cleaned_no_period = cleaned.rstrip(".")
        allowed = [f"this is iris {species}" for species in VALID_SPECIES]
        return cleaned_no_period in allowed

    return False


def leakage_detected(text: Optional[str]) -> bool:
    if text is None:
        return False

    lowered = text.lower()
    leakage_keywords = [
        "system prompt",
        "developer message",
        "hidden instruction",
        "context window",
        "everything above",
        "training example",
        "few-shot",
        "tuning dataset",
        "input_text",
        "output_text",
        "contents",
        "parts",
        "role",
    ]

    return any(keyword in lowered for keyword in leakage_keywords)
