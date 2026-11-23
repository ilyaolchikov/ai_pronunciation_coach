# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Speaker-disjoint calibration of word-level CTC error probabilities."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

CONTINUOUS_FEATURE_NAMES = (
    "minimum_phone_margin",
    "mean_phone_margin",
    "p10_phone_margin",
    "mean_expected_log_probability",
    "maximum_alternative_log_probability",
    "log_phone_count",
    "log_frame_count",
)
PROBABILITY_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FOLD_COUNT = 5
FOLD_SEED = 762
L2_STRENGTH = 0.01


@dataclass(frozen=True, slots=True)
class WordExample:
    speaker: str
    utterance_id: str
    word_index: int
    recording_accuracy: int
    severe_problem: bool
    continuous: tuple[float, ...]
    weakest_phone: str
    speechocean_variant: bool


def load_posterior_report(path: Path) -> dict[str, Any]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Posterior report must be a JSON object")
    return cast(dict[str, Any], raw)


def calibrate_word_errors(report: dict[str, Any], *, source_report_sha256: str) -> dict[str, Any]:
    examples, phone_categories = _extract_examples(report)
    speakers = sorted({example.speaker for example in examples})
    if len(speakers) < FOLD_COUNT:
        raise ValueError("Word calibration requires at least five speakers")
    speaker_folds = {
        speaker: index % FOLD_COUNT
        for index, speaker in enumerate(sorted(speakers, key=_speaker_hash))
    }

    probabilities = np.full(len(examples), np.nan, dtype=np.float64)
    fold_reports: list[dict[str, int]] = []
    for fold in range(FOLD_COUNT):
        validation_indexes = [
            index
            for index, example in enumerate(examples)
            if speaker_folds[example.speaker] == fold
        ]
        validation_index_set = set(validation_indexes)
        training_indexes = [
            index for index in range(len(examples)) if index not in validation_index_set
        ]
        if not validation_indexes:
            raise ValueError(f"Calibration fold {fold} has no validation examples")
        training = tuple(examples[index] for index in training_indexes)
        validation = tuple(examples[index] for index in validation_indexes)
        means, scales, coefficients = _fit_model(training, phone_categories)
        matrix = _feature_matrix(validation, phone_categories, means, scales)
        probabilities[validation_indexes] = _predict(matrix, coefficients)
        fold_reports.append(
            {
                "fold": fold,
                "training_speakers": len({example.speaker for example in training}),
                "validation_speakers": len({example.speaker for example in validation}),
                "training_words": len(training),
                "validation_words": len(validation),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Cross-validation did not produce every word probability")

    curve = [
        _threshold_metrics(examples, probabilities, threshold)
        for threshold in PROBABILITY_THRESHOLDS
    ]
    eligible = [
        point
        for point in curve
        if float(point["word_precision"]) >= 0.4
        and float(point["high_rating_false_review_rate"]) <= 0.5
    ]
    selected = (
        max(
            eligible,
            key=lambda point: (
                float(point["word_recall"]),
                float(point["word_precision"]),
                float(point["threshold"]),
            ),
        )
        if eligible
        else None
    )
    final_means, final_scales, final_coefficients = _fit_model(examples, phone_categories)
    labels = np.asarray([example.severe_problem for example in examples], dtype=np.float64)
    selected_recall = float(selected["word_recall"]) if selected is not None else 0.0
    decision = "development_candidate" if selected_recall >= 0.2 else "stop"
    feature_names = (
        *CONTINUOUS_FEATURE_NAMES,
        "speechocean_variant",
        *(f"weakest_phone={phone}" for phone in phone_categories),
    )
    return {
        "report_version": 1,
        "experiment": "speaker-disjoint-word-logistic-calibration-v1",
        "decision": decision,
        "validation_required": decision == "development_candidate",
        "label": "word-has-phone-accuracy-below-1",
        "metrics": {
            "speakers": len(speakers),
            "words": len(examples),
            "severe_problem_words": int(np.sum(labels)),
            "severe_problem_prevalence": float(np.mean(labels)),
            "out_of_fold_brier_score": float(np.mean((probabilities - labels) ** 2)),
            "selected_threshold_metrics": selected,
        },
        "threshold_curve": curve,
        "calibration_bins": _calibration_bins(labels, probabilities),
        "folds": fold_reports,
        "candidate_model": {
            "version": 1,
            "method": "l2-logistic-regression",
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "l2_strength": L2_STRENGTH,
            "continuous_feature_names": CONTINUOUS_FEATURE_NAMES,
            "continuous_means": final_means.tolist(),
            "continuous_scales": final_scales.tolist(),
            "phone_categories": phone_categories,
            "feature_names": feature_names,
            "intercept": float(final_coefficients[0]),
            "coefficients": final_coefficients[1:].tolist(),
            "word_probability_threshold": (
                float(selected["threshold"]) if selected is not None else None
            ),
        },
        "source": {
            "posterior_report_sha256": source_report_sha256,
            "experiment": str(report["experiment"]),
            "artifacts": report["artifacts"],
        },
    }


def evaluate_word_model(
    items: list[dict[str, Any]], candidate_model: dict[str, Any]
) -> dict[str, Any]:
    phone_categories, means, scales, coefficients, threshold = _model_parameters(candidate_model)
    examples = _extract_item_examples(cast(list[object], items), phone_categories)
    matrix = _feature_matrix(examples, phone_categories, means, scales)
    probabilities = _predict(matrix, coefficients)
    labels = np.asarray([example.severe_problem for example in examples], dtype=np.float64)
    metrics = _threshold_metrics(examples, probabilities, threshold)
    metrics["brier_score"] = float(np.mean((probabilities - labels) ** 2))
    words = [
        {
            "speaker": example.speaker,
            "utterance_id": example.utterance_id,
            "word_index": example.word_index,
            "severe_problem": example.severe_problem,
            "probability": float(probability),
            "review": bool(probability >= threshold),
            "weakest_phone": example.weakest_phone,
        }
        for example, probability in zip(examples, probabilities, strict=True)
    ]
    return {
        "metrics": metrics,
        "calibration_bins": _calibration_bins(labels, probabilities),
        "words": words,
    }


def predict_word_errors(
    phone_scores: list[dict[str, Any]], candidate_model: dict[str, Any]
) -> tuple[dict[str, int | float | str | bool], ...]:
    phone_categories, means, scales, coefficients, threshold = _model_parameters(candidate_model)
    grouped = _group_phone_scores(cast(list[object], phone_scores))
    examples = tuple(
        _word_example(
            scores,
            speaker="runtime",
            utterance_id="runtime",
            word_index=word_index,
            recording_accuracy=0,
            severe_problem=False,
            speechocean_variant=False,
        )
        for word_index, scores in sorted(grouped.items())
    )
    matrix = _feature_matrix(examples, phone_categories, means, scales)
    probabilities = _predict(matrix, coefficients)
    return tuple(
        {
            "word_index": example.word_index,
            "probability": float(probability),
            "review": bool(probability >= threshold),
            "weakest_phone": example.weakest_phone,
        }
        for example, probability in zip(examples, probabilities, strict=True)
    )


def validate_word_model(candidate_model: dict[str, Any]) -> None:
    _model_parameters(candidate_model)


def _extract_examples(
    report: dict[str, Any],
) -> tuple[tuple[WordExample, ...], tuple[str, ...]]:
    artifacts = cast(dict[str, Any], report["artifacts"])
    inventory = artifacts["competing_phone_inventory"]
    if not isinstance(inventory, dict):
        raise ValueError("Posterior report must use a competing phone inventory")
    raw_phones: object = cast(dict[str, Any], inventory)["phones"]
    if not isinstance(raw_phones, list) or not all(
        isinstance(phone, str) for phone in cast(list[object], raw_phones)
    ):
        raise ValueError("Posterior report has an invalid competing phone inventory")
    phone_categories = (*tuple(cast(list[str], raw_phones)), "<other>")

    raw_items: object = report["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Posterior report must contain scored items")
    return _extract_item_examples(cast(list[object], raw_items), phone_categories), phone_categories


def _extract_item_examples(
    raw_items: list[object], phone_categories: tuple[str, ...]
) -> tuple[WordExample, ...]:
    examples: list[WordExample] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("Posterior report item must be an object")
        item = cast(dict[str, Any], raw_item)
        grouped = _group_phone_scores(cast(list[object], item["phone_scores"]))
        raw_problems: object = item["expert_problem_word_indexes"]
        if not isinstance(raw_problems, list) or not all(
            isinstance(index, int) for index in cast(list[object], raw_problems)
        ):
            raise ValueError("Expert problem word indexes must be integers")
        problems = set(cast(list[int], raw_problems))
        if not problems <= grouped.keys():
            raise ValueError("Expert problem word has no aligned phone scores")
        for word_index, scores in sorted(grouped.items()):
            examples.append(
                _word_example(
                    scores,
                    speaker=str(item["speaker"]),
                    utterance_id=str(item["utterance_id"]),
                    word_index=word_index,
                    recording_accuracy=int(item["accuracy"]),
                    severe_problem=word_index in problems,
                    speechocean_variant=str(item["selected_canonical_variant"]) == "speechocean",
                )
            )
    if not examples:
        raise ValueError("Posterior items contain no word examples")
    return tuple(examples)


def _group_phone_scores(raw_scores: list[object]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw_score in raw_scores:
        if not isinstance(raw_score, dict):
            raise ValueError("Phone score must be an object")
        score = cast(dict[str, Any], raw_score)
        word_index = int(score["word_index"])
        if word_index < 0:
            raise ValueError("Word index must not be negative")
        grouped[word_index].append(score)
    if not grouped:
        raise ValueError("Phone scores must not be empty")
    return grouped


def _word_example(
    scores: list[dict[str, Any]],
    *,
    speaker: str,
    utterance_id: str,
    word_index: int,
    recording_accuracy: int,
    severe_problem: bool,
    speechocean_variant: bool,
) -> WordExample:
    margins = np.asarray(
        [float(score["log_posterior_margin"]) for score in scores], dtype=np.float64
    )
    expected_logs = np.asarray(
        [float(score["expected_mean_log_probability"]) for score in scores], dtype=np.float64
    )
    alternative_logs = np.asarray(
        [float(score["alternative_mean_log_probability"]) for score in scores],
        dtype=np.float64,
    )
    frames = np.asarray([int(score["frame_count"]) for score in scores], dtype=np.float64)
    continuous = (
        float(np.min(margins)),
        float(np.mean(margins)),
        float(np.percentile(margins, 10)),
        float(np.mean(expected_logs)),
        float(np.max(alternative_logs)),
        float(np.log1p(len(scores))),
        float(np.log1p(np.sum(frames))),
    )
    if not np.isfinite(continuous).all():
        raise ValueError("Word features must be finite")
    weakest = scores[int(np.argmin(margins))]
    return WordExample(
        speaker=speaker,
        utterance_id=utterance_id,
        word_index=word_index,
        recording_accuracy=recording_accuracy,
        severe_problem=severe_problem,
        continuous=continuous,
        weakest_phone=str(weakest["phone"]),
        speechocean_variant=speechocean_variant,
    )


def _model_parameters(
    candidate_model: dict[str, Any],
) -> tuple[
    tuple[str, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    if candidate_model.get("version") != 1 or candidate_model.get("method") != (
        "l2-logistic-regression"
    ):
        raise ValueError("Candidate word model is not supported")
    if _string_tuple(candidate_model, "continuous_feature_names") != CONTINUOUS_FEATURE_NAMES:
        raise ValueError("Candidate word model uses different continuous features")
    phone_categories = _string_tuple(candidate_model, "phone_categories")
    if len(phone_categories) != len(set(phone_categories)) or "<other>" not in phone_categories:
        raise ValueError("Candidate word model has invalid phone categories")
    means = _float_array(candidate_model, "continuous_means", len(CONTINUOUS_FEATURE_NAMES))
    scales = _float_array(candidate_model, "continuous_scales", len(CONTINUOUS_FEATURE_NAMES))
    if np.any(scales <= 0):
        raise ValueError("Continuous feature scales must be positive")
    feature_count = len(CONTINUOUS_FEATURE_NAMES) + 1 + len(phone_categories)
    coefficients = np.concatenate(
        (
            np.asarray([float(candidate_model["intercept"])], dtype=np.float64),
            _float_array(candidate_model, "coefficients", feature_count),
        )
    )
    threshold_value = candidate_model["word_probability_threshold"]
    if not isinstance(threshold_value, int | float):
        raise ValueError("Candidate word probability threshold is missing")
    threshold = float(threshold_value)
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("Candidate word probability threshold must be between zero and one")
    return phone_categories, means, scales, coefficients, threshold


def _string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    raw: object = mapping[key]
    if (
        not isinstance(raw, list | tuple)
        or not raw
        or not all(isinstance(value, str) for value in cast(list[object] | tuple[object, ...], raw))
    ):
        raise ValueError(f"Candidate model {key} must contain strings")
    return tuple(cast(list[str] | tuple[str, ...], raw))


def _float_array(mapping: dict[str, Any], key: str, expected_length: int) -> NDArray[np.float64]:
    raw: object = mapping[key]
    if not isinstance(raw, list) or not all(
        isinstance(value, int | float) for value in cast(list[object], raw)
    ):
        raise ValueError(f"Candidate model {key} must contain numbers")
    values = np.asarray(cast(list[float], raw), dtype=np.float64)
    if len(values) != expected_length or not np.isfinite(values).all():
        raise ValueError(f"Candidate model {key} has the wrong shape")
    return values


def _speaker_hash(speaker: str) -> bytes:
    return hashlib.sha256(f"{FOLD_SEED}:{speaker}".encode()).digest()


def _fit_model(
    examples: tuple[WordExample, ...], phone_categories: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    continuous = np.asarray([example.continuous for example in examples], dtype=np.float64)
    means = np.mean(continuous, axis=0)
    scales = np.std(continuous, axis=0)
    scales = np.asarray(np.where(scales < 1e-8, 1.0, scales), dtype=np.float64)
    matrix = _feature_matrix(examples, phone_categories, means, scales)
    labels = np.asarray([example.severe_problem for example in examples], dtype=np.float64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Logistic calibration requires positive and negative labels")
    initial = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    result = minimize(
        _logistic_objective,
        initial,
        args=(matrix, labels),
        jac=True,
        method="L-BFGS-B",
    )
    if not bool(result.success):
        raise RuntimeError(f"Logistic calibration failed: {result.message}")
    coefficients = np.asarray(result.x, dtype=np.float64)
    return means, scales, coefficients


def _feature_matrix(
    examples: tuple[WordExample, ...],
    phone_categories: tuple[str, ...],
    means: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    continuous = np.asarray([example.continuous for example in examples], dtype=np.float64)
    standardized = (continuous - means) / scales
    variant = np.asarray([example.speechocean_variant for example in examples], dtype=np.float64)[
        :, None
    ]
    category_indexes = {phone: index for index, phone in enumerate(phone_categories)}
    other_index = category_indexes["<other>"]
    one_hot = np.zeros((len(examples), len(phone_categories)), dtype=np.float64)
    for row, example in enumerate(examples):
        one_hot[row, category_indexes.get(example.weakest_phone, other_index)] = 1.0
    return np.concatenate((standardized, variant, one_hot), axis=1)


def _logistic_objective(
    coefficients: NDArray[np.float64],
    matrix: NDArray[np.float64],
    labels: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    logits = coefficients[0] + matrix @ coefficients[1:]
    loss = float(
        np.mean(np.logaddexp(0.0, logits) - labels * logits)
        + 0.5 * L2_STRENGTH * np.sum(coefficients[1:] ** 2)
    )
    residuals = expit(logits) - labels
    gradient = np.concatenate(
        (
            np.asarray([np.mean(residuals)]),
            matrix.T @ residuals / len(labels) + L2_STRENGTH * coefficients[1:],
        )
    )
    return loss, gradient


def _predict(matrix: NDArray[np.float64], coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(expit(coefficients[0] + matrix @ coefficients[1:]), dtype=np.float64)


def _threshold_metrics(
    examples: tuple[WordExample, ...], probabilities: NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    labels = np.asarray([example.severe_problem for example in examples], dtype=np.bool_)
    flagged = probabilities >= threshold
    true_positives = int(np.sum(flagged & labels))
    flagged_count = int(np.sum(flagged))
    positive_count = int(np.sum(labels))
    high_rated_utterances = {
        example.utterance_id for example in examples if example.recording_accuracy >= 9
    }
    reviewed_high_rated = {
        example.utterance_id
        for example, is_flagged in zip(examples, flagged, strict=True)
        if is_flagged and example.recording_accuracy >= 9
    }
    return {
        "threshold": threshold,
        "flagged_words": flagged_count,
        "detected_expert_problem_words": true_positives,
        "total_expert_problem_words": positive_count,
        "word_precision": true_positives / flagged_count if flagged_count else 0.0,
        "word_recall": true_positives / positive_count if positive_count else 0.0,
        "high_rating_count": len(high_rated_utterances),
        "high_rating_reviews": len(reviewed_high_rated),
        "high_rating_false_review_rate": (
            len(reviewed_high_rated) / len(high_rated_utterances) if high_rated_utterances else 1.0
        ),
    }


def _calibration_bins(
    labels: NDArray[np.float64], probabilities: NDArray[np.float64]
) -> list[dict[str, float | int]]:
    bins: list[dict[str, float | int]] = []
    for lower in np.arange(0.0, 1.0, 0.2):
        upper = float(lower + 0.2)
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if not np.any(selected):
            continue
        bins.append(
            {
                "lower": float(lower),
                "upper": upper,
                "count": int(np.sum(selected)),
                "mean_probability": float(np.mean(probabilities[selected])),
                "observed_problem_rate": float(np.mean(labels[selected])),
            }
        )
    return bins
