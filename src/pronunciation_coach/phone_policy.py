"""Frozen L2-ARCTIC phone-error policy used for supplemental warnings."""

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, cast

from pronunciation_coach.errors import CoachError, ErrorCode

PHONE_FEATURE_NAMES = (
    "expected_mean_log_probability",
    "alternative_mean_log_probability",
    "log_posterior_margin",
    "log_frame_count",
)


@dataclass(frozen=True, slots=True)
class PhonePolicy:
    id: str
    version: int
    model_id: str
    model_revision: str
    calibration_report_sha256: str
    final_report_sha256: str
    warning_probability_threshold: float
    candidate_model: dict[str, Any]


def load_phone_policy() -> PhonePolicy:
    resource = files("pronunciation_coach.data").joinpath("phone_policy.v1.json")
    try:
        raw: object = json.loads(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("phone policy must be an object")
        value = cast(dict[str, Any], raw)
        candidate = value["candidate_model"]
        if not isinstance(candidate, dict):
            raise ValueError("candidate model must be an object")
        candidate_model = cast(dict[str, Any], candidate)
        validate_phone_model(candidate_model)
        policy = PhonePolicy(
            id=str(value["id"]),
            version=int(value["version"]),
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            calibration_report_sha256=str(value["calibration_report_sha256"]),
            final_report_sha256=str(value["final_report_sha256"]),
            warning_probability_threshold=float(value["warning_probability_threshold"]),
            candidate_model=candidate_model,
        )
        if not 0.0 < policy.warning_probability_threshold < 1.0:
            raise ValueError("warning threshold must be between zero and one")
        for digest in (policy.calibration_report_sha256, policy.final_report_sha256):
            if len(digest) != 64 or not set(digest) <= set("0123456789abcdef"):
                raise ValueError("phone policy provenance is invalid")
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        raise CoachError(ErrorCode.CATALOGUE_INVALID, "Phone policy is invalid") from error
    return policy


def validate_phone_model(model: dict[str, Any]) -> None:
    if model.get("version") != 1 or model.get("method") != "l2-logistic-regression":
        raise ValueError("unsupported phone model")
    if tuple(model.get("continuous_feature_names", ())) != PHONE_FEATURE_NAMES:
        raise ValueError("phone model uses different continuous features")
    expected = _categories(model, "expected_phone_categories")
    alternatives = _categories(model, "alternative_phone_categories")
    means = _numbers(model, "continuous_means")
    scales = _numbers(model, "continuous_scales")
    coefficients = _numbers(model, "coefficients")
    if len(means) != len(PHONE_FEATURE_NAMES) or len(scales) != len(PHONE_FEATURE_NAMES):
        raise ValueError("phone model has invalid continuous parameters")
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("phone model scales must be positive")
    if len(coefficients) != len(PHONE_FEATURE_NAMES) + len(expected) + len(alternatives):
        raise ValueError("phone model has invalid coefficient count")
    float(model["intercept"])


def predict_phone_errors(
    phone_scores: list[dict[str, Any]], model: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_phone_model(model)
    expected_categories = _categories(model, "expected_phone_categories")
    alternative_categories = _categories(model, "alternative_phone_categories")
    expected_indexes = {phone: index for index, phone in enumerate(expected_categories)}
    alternative_indexes = {phone: index for index, phone in enumerate(alternative_categories)}
    expected_other = expected_indexes["<other>"]
    alternative_other = alternative_indexes["<other>"]
    means = _numbers(model, "continuous_means")
    scales = _numbers(model, "continuous_scales")
    coefficients = _numbers(model, "coefficients")
    expected_offset = len(PHONE_FEATURE_NAMES)
    alternative_offset = expected_offset + len(expected_categories)
    predictions: list[dict[str, Any]] = []
    for score in phone_scores:
        features = (
            float(score["expected_mean_log_probability"]),
            float(score["alternative_mean_log_probability"]),
            float(score["log_posterior_margin"]),
            math.log1p(int(score["frame_count"])),
        )
        logit = float(model["intercept"])
        logit += sum(
            coefficient * ((feature - mean) / scale)
            for coefficient, feature, mean, scale in zip(
                coefficients[:expected_offset], features, means, scales, strict=True
            )
        )
        expected_index = expected_indexes.get(str(score["phone"]), expected_other)
        alternative_index = alternative_indexes.get(
            str(score["alternative_phone"]), alternative_other
        )
        logit += coefficients[expected_offset + expected_index]
        logit += coefficients[alternative_offset + alternative_index]
        predictions.append(
            {
                "word_index": int(score["word_index"]),
                "phone": str(score["phone"]),
                "alternative_phone": str(score["alternative_phone"]),
                "probability": _sigmoid(logit),
            }
        )
    return predictions


def _categories(model: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = model.get(key)
    if not isinstance(raw, list) or "<other>" not in raw:
        raise ValueError(f"phone model has invalid {key}")
    values = tuple(str(value) for value in cast(list[object], raw))
    if len(values) != len(set(values)):
        raise ValueError(f"phone model has duplicate {key}")
    return values


def _numbers(model: dict[str, Any], key: str) -> tuple[float, ...]:
    raw = model.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"phone model has invalid {key}")
    numeric_values: list[float] = []
    for value in cast(list[object], raw):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"phone model has invalid {key}")
        numeric_values.append(float(value))
    values = tuple(numeric_values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"phone model has non-finite {key}")
    return values


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
