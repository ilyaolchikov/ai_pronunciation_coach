# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Isolated L2-ARCTIC phone-error calibration using the frozen LV-60 scorer."""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from pronunciation_coach.audio import load_audio
from pronunciation_coach.benchmarking import CompetingPhoneInventory
from pronunciation_coach.calibrated_analysis import load_word_policy
from pronunciation_coach.config import DEFAULT_MODEL_CACHE, MODEL_ID, MODEL_REVISION
from pronunciation_coach.l2arctic import (
    L2ARCTIC_DATASET_ID,
    L2ArcticSplitManifest,
    Partition,
    iter_l2arctic_utterances,
)
from pronunciation_coach.recognizers.wav2vec2_espeak import Wav2Vec2EspeakRecognizer
from pronunciation_coach.word_calibration import predict_word_errors

CONTINUOUS_FEATURE_NAMES = (
    "expected_mean_log_probability",
    "alternative_mean_log_probability",
    "log_posterior_margin",
    "log_frame_count",
)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
L2_STRENGTH = 0.01
RED_WORD_PRECISION_MIN = 0.80
RED_WORD_PRECISION_MARGIN = 0.02
YELLOW_PHONE_PRECISION_MIN = 0.40


@dataclass(frozen=True, slots=True)
class PhoneExample:
    speaker: str
    first_language: str
    utterance_id: str
    word_index: int
    expected: str
    perceived: str | None
    alternative: str
    error_type: str
    continuous: tuple[float, ...]

    @property
    def error(self) -> bool:
        return self.error_type != "correct"


def run_l2arctic_evidence(
    dataset_root: Path,
    manifest: L2ArcticSplitManifest,
    partition: Partition,
    inventory: CompetingPhoneInventory,
    *,
    manifest_sha256: str,
    inventory_sha256: str,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str = "cpu",
    allow_final_test: bool = False,
) -> dict[str, Any]:
    if partition == "final-test" and not allow_final_test:
        raise ValueError("L2-ARCTIC evidence runner refuses the locked final-test partition")
    recognizer = Wav2Vec2EspeakRecognizer(cache_dir=model_cache, device=device)
    competing_phones = frozenset(inventory.phones)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    excluded_additions = 0
    operation_counts: Counter[str] = Counter()

    speakers = manifest.speakers_for(partition)
    for utterance in iter_l2arctic_utterances(dataset_root, speakers, include_audio=True):
        target = tuple(phone for phone in utterance.phones if phone.error_type != "addition")
        excluded_additions += sum(phone.error_type == "addition" for phone in utterance.phones)
        try:
            expected_phones = tuple(_required_expected_ipa(phone) for phone in target)
            word_indexes = tuple(_required_word_index(phone) for phone in target)
            if utterance.audio_bytes is None:
                raise ValueError("L2-ARCTIC utterance has no audio bytes")
            recognition = recognizer.score_expected(
                load_audio(utterance.audio_bytes),
                expected_phones,
                competing_phones=competing_phones,
            )
            phone_scores: list[dict[str, Any]] = []
            for annotation, word_index, score in zip(
                target, word_indexes, recognition.alignment.phone_scores, strict=True
            ):
                operation_counts[annotation.error_type] += 1
                phone_scores.append(
                    {
                        "word_index": word_index,
                        "expected": score.phone,
                        "perceived": annotation.perceived_ipa,
                        "expected_arpabet": annotation.expected_arpabet,
                        "perceived_arpabet": annotation.perceived_arpabet,
                        "error_type": annotation.error_type,
                        "deviation": annotation.deviation,
                        "start_frame": score.start_frame,
                        "end_frame": score.end_frame,
                        "frame_count": score.frame_count,
                        "expected_mean_probability": score.expected_mean_probability,
                        "expected_mean_log_probability": score.expected_mean_log_probability,
                        "alternative": score.alternative_phone,
                        "alternative_mean_log_probability": (
                            score.alternative_mean_log_probability
                        ),
                        "log_posterior_margin": score.log_posterior_margin,
                    }
                )
            completed.append(
                {
                    "speaker": utterance.speaker.id,
                    "first_language": utterance.speaker.first_language,
                    "gender": utterance.speaker.gender,
                    "utterance_id": utterance.utterance_id,
                    "transcript": utterance.transcript,
                    "words": utterance.words,
                    "greedy_phones": recognition.greedy_phones,
                    "inference_ms": recognition.inference_ms,
                    "phone_scores": phone_scores,
                }
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append(
                {
                    "speaker": utterance.speaker.id,
                    "utterance_id": utterance.utterance_id,
                    "error": str(error),
                }
            )

    attempted = len(completed) + len(failures)
    return {
        "report_version": 1,
        "experiment": "l2-arctic-lv60-phone-evidence-v1",
        "partition": partition,
        "dataset": {"id": L2ARCTIC_DATASET_ID, "root": str(dataset_root)},
        "artifacts": {
            "manifest_sha256": manifest_sha256,
            "inventory_sha256": inventory_sha256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "competing_phones": list(inventory.phones),
        },
        "metrics": {
            "speakers": len(speakers),
            "attempted_utterances": attempted,
            "completed_utterances": len(completed),
            "failed_utterances": len(failures),
            "completion_rate": len(completed) / attempted if attempted else 0.0,
            "scored_phones": sum(len(item["phone_scores"]) for item in completed),
            "excluded_additions": excluded_additions,
            "operation_counts": dict(sorted(operation_counts.items())),
        },
        "failures": failures,
        "items": completed,
    }


def calibrate_l2arctic_phone_errors(
    training_report: dict[str, Any],
    development_report: dict[str, Any],
    *,
    training_report_sha256: str,
    development_report_sha256: str,
) -> dict[str, Any]:
    _validate_evidence_report(training_report, "train")
    _validate_evidence_report(development_report, "development")
    training = _extract_examples(training_report)
    development = _extract_examples(development_report)
    categories = tuple(sorted({example.expected for example in training} | {"<other>"}))
    alternative_categories = tuple(
        sorted({example.alternative for example in training} | {"<other>"})
    )
    means, scales, coefficients = _fit_model(training, categories, alternative_categories)
    probabilities = _predict_examples(
        development, categories, alternative_categories, means, scales, coefficients
    )
    curve = [_phone_metrics(development, probabilities, threshold) for threshold in THRESHOLDS]
    word_curve = [_word_metrics(development, probabilities, threshold) for threshold in THRESHOLDS]
    baseline = evaluate_current_word_policy(development_report)
    required_red_precision = min(
        1.0,
        max(
            RED_WORD_PRECISION_MIN,
            float(baseline["precision"]) + RED_WORD_PRECISION_MARGIN,
        ),
    )
    red = select_word_threshold(
        word_curve,
        minimum_precision=required_red_precision,
        minimum_recall=float(baseline["recall"]),
    )
    yellow = (
        _select_threshold(
            [point for point in curve if float(point["threshold"]) < float(red["threshold"])],
            YELLOW_PHONE_PRECISION_MIN,
        )
        if red is not None
        else None
    )
    candidate_model = {
        "version": 1,
        "method": "l2-logistic-regression",
        "continuous_feature_names": CONTINUOUS_FEATURE_NAMES,
        "continuous_means": means.tolist(),
        "continuous_scales": scales.tolist(),
        "expected_phone_categories": categories,
        "alternative_phone_categories": alternative_categories,
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "red_probability_threshold": float(red["threshold"]) if red is not None else None,
        "yellow_probability_threshold": (
            float(yellow["threshold"]) if yellow is not None else None
        ),
    }
    red_evaluation = (
        evaluate_l2arctic_phone_model(development_report, candidate_model)
        if red is not None
        else None
    )
    candidate_word_precision = (
        float(red_evaluation["word_metrics"]["precision"]) if red_evaluation is not None else 0.0
    )
    candidate_word_recall = (
        float(red_evaluation["word_metrics"]["recall"]) if red_evaluation is not None else 0.0
    )
    decision = (
        "development_candidate"
        if red is not None
        and red_evaluation is not None
        and candidate_word_precision >= required_red_precision
        and candidate_word_recall > float(baseline["recall"])
        else "stop"
    )
    return {
        "report_version": 1,
        "experiment": "l2-arctic-phone-error-calibration-v1",
        "decision": decision,
        "validation_required": decision == "development_candidate",
        "threshold_requirements": {
            "red_word_precision_min": RED_WORD_PRECISION_MIN,
            "red_word_precision_margin_over_current": RED_WORD_PRECISION_MARGIN,
            "required_red_word_precision": required_red_precision,
            "red_word_recall_must_exceed_current": True,
            "yellow_phone_precision_min": YELLOW_PHONE_PRECISION_MIN,
        },
        "metrics": {
            "training_phones": len(training),
            "development_phones": len(development),
            "development_error_phones": sum(example.error for example in development),
            "selected_red": red,
            "selected_yellow": yellow,
            "current_word_policy": baseline,
            "candidate_red": red_evaluation,
        },
        "threshold_curve": curve,
        "word_threshold_curve": word_curve,
        "candidate_model": candidate_model,
        "source": {
            "training_report_sha256": training_report_sha256,
            "development_report_sha256": development_report_sha256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        },
    }


def evaluate_frozen_l2arctic_candidate(
    evidence_report: dict[str, Any],
    calibration_report: dict[str, Any],
    *,
    evidence_report_sha256: str,
    calibration_report_sha256: str,
) -> dict[str, Any]:
    _validate_evidence_report(evidence_report, "final-test")
    if calibration_report["decision"] != "development_candidate":
        raise ValueError("L2-ARCTIC calibration report is not a frozen development candidate")
    candidate = calibration_report["candidate_model"]
    if not isinstance(candidate, dict):
        raise ValueError("L2-ARCTIC calibration report lacks a candidate model")
    evaluation = evaluate_l2arctic_phone_model(evidence_report, cast(dict[str, Any], candidate))
    baseline = evaluate_current_word_policy(evidence_report)
    candidate_word_metrics = evaluation["word_metrics"]
    decision = (
        "improves_precision_and_recall"
        if float(candidate_word_metrics["precision"]) > float(baseline["precision"])
        and float(candidate_word_metrics["recall"]) > float(baseline["recall"])
        else "no_joint_improvement"
    )
    return {
        "report_version": 1,
        "experiment": "l2-arctic-phone-error-final-v1",
        "decision": decision,
        "threshold_selection_performed": False,
        "metrics": {"candidate": evaluation, "current_word_policy": baseline},
        "source": {
            "evidence_report_sha256": evidence_report_sha256,
            "calibration_report_sha256": calibration_report_sha256,
        },
    }


def evaluate_l2arctic_speechocean_regression(
    evidence_report: dict[str, Any],
    calibration_report: dict[str, Any],
    *,
    warning_probability_threshold: float,
    evidence_report_sha256: str,
    calibration_report_sha256: str,
) -> dict[str, Any]:
    if evidence_report.get("experiment") != (
        "constrained-ctc-posterior-english-competitors-canonical-variants-v1"
    ):
        raise ValueError("Unexpected SpeechOcean evidence experiment")
    if calibration_report.get("decision") != "development_candidate":
        raise ValueError("L2-ARCTIC calibration report is not a development candidate")
    candidate = calibration_report.get("candidate_model")
    if not isinstance(candidate, dict):
        raise ValueError("L2-ARCTIC calibration report lacks a candidate model")
    if not 0.0 < warning_probability_threshold < 1.0:
        raise ValueError("Warning probability threshold must be between zero and one")
    model = cast(dict[str, Any], candidate).copy()
    model["red_probability_threshold"] = warning_probability_threshold
    categories, alternative_categories, means, scales, coefficients, threshold = _model_parameters(
        model
    )
    word_policy = load_word_policy().candidate_model
    totals = {name: [0, 0, 0] for name in ("candidate", "current", "union")}
    items = cast(list[dict[str, Any]], evidence_report["items"])
    for item in items:
        phone_scores = cast(list[dict[str, Any]], item["phone_scores"])
        examples = tuple(
            PhoneExample(
                speaker=str(item["speaker"]),
                first_language="",
                utterance_id=str(item["utterance_id"]),
                word_index=int(score["word_index"]),
                expected=str(score["phone"]),
                perceived=None,
                alternative=str(score["alternative_phone"]),
                error_type="correct",
                continuous=(
                    float(score["expected_mean_log_probability"]),
                    float(score["alternative_mean_log_probability"]),
                    float(score["log_posterior_margin"]),
                    float(np.log1p(int(score["frame_count"]))),
                ),
            )
            for score in phone_scores
        )
        probabilities = _predict_examples(
            examples, categories, alternative_categories, means, scales, coefficients
        )
        candidate_words = {
            example.word_index
            for example, probability in zip(examples, probabilities, strict=True)
            if probability >= threshold
        }
        current_words = {
            int(result["word_index"])
            for result in predict_word_errors(phone_scores, word_policy)
            if bool(result["review"])
        }
        truth = set(cast(list[int], item["expert_problem_word_indexes"]))
        for name, predictions in (
            ("candidate", candidate_words),
            ("current", current_words),
            ("union", candidate_words | current_words),
        ):
            _accumulate_set_metrics(totals[name], predictions, truth)
    metrics = {name: _set_metrics(values) for name, values in totals.items()}
    decision = (
        "supplemental_warning_only"
        if metrics["candidate"]["precision"] < metrics["current"]["precision"]
        else "candidate_red_eligible"
    )
    return {
        "report_version": 1,
        "experiment": "l2-arctic-speechocean-regression-v1",
        "decision": decision,
        "threshold_selection_performed": False,
        "metrics": metrics,
        "source": {
            "evidence_report_sha256": evidence_report_sha256,
            "calibration_report_sha256": calibration_report_sha256,
            "warning_probability_threshold": warning_probability_threshold,
            "recordings": len(items),
        },
    }


def evaluate_l2arctic_phone_model(
    evidence_report: dict[str, Any], candidate_model: dict[str, Any]
) -> dict[str, Any]:
    examples = _extract_examples(evidence_report)
    categories, alternative_categories, means, scales, coefficients, threshold = _model_parameters(
        candidate_model
    )
    probabilities = _predict_examples(
        examples, categories, alternative_categories, means, scales, coefficients
    )
    return {
        "phone_metrics": _phone_metrics(examples, probabilities, threshold),
        "word_metrics": _word_metrics(examples, probabilities, threshold),
        "diagnosis": _diagnosis_metrics(examples, probabilities, threshold),
        "by_first_language": {
            language: _phone_metrics(
                tuple(example for example in examples if example.first_language == language),
                np.asarray(
                    [
                        probability
                        for example, probability in zip(examples, probabilities, strict=True)
                        if example.first_language == language
                    ],
                    dtype=np.float64,
                ),
                threshold,
            )
            for language in sorted({example.first_language for example in examples})
        },
    }


def evaluate_current_word_policy(evidence_report: dict[str, Any]) -> dict[str, Any]:
    policy = load_word_policy()
    labels: list[bool] = []
    predictions: list[bool] = []
    grouped_items: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw_item in cast(list[dict[str, Any]], evidence_report["items"]):
        key = (str(raw_item["speaker"]), str(raw_item["utterance_id"]))
        grouped_items[key].append(raw_item)
    for raw_items in grouped_items.values():
        item = raw_items[0]
        phone_scores = cast(list[dict[str, Any]], item["phone_scores"])
        compatible_scores = [
            {
                "word_index": int(score["word_index"]),
                "phone": str(score["expected"]),
                "frame_count": int(score["frame_count"]),
                "expected_mean_log_probability": float(score["expected_mean_log_probability"]),
                "alternative_mean_log_probability": float(
                    score["alternative_mean_log_probability"]
                ),
                "log_posterior_margin": float(score["log_posterior_margin"]),
            }
            for score in phone_scores
        ]
        predicted = {
            int(result["word_index"]): bool(result["review"])
            for result in predict_word_errors(compatible_scores, policy.candidate_model)
        }
        word_indexes = sorted({int(score["word_index"]) for score in phone_scores})
        for word_index in word_indexes:
            labels.append(
                any(
                    int(score["word_index"]) == word_index and str(score["error_type"]) != "correct"
                    for score in phone_scores
                )
            )
            predictions.append(predicted[word_index])
    return _binary_metrics(np.asarray(labels), np.asarray(predictions))


def load_json_report(path: Path) -> dict[str, Any]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment report must be a JSON object")
    return cast(dict[str, Any], raw)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_examples(report: dict[str, Any]) -> tuple[PhoneExample, ...]:
    examples: list[PhoneExample] = []
    for item in cast(list[dict[str, Any]], report["items"]):
        for score in cast(list[dict[str, Any]], item["phone_scores"]):
            continuous = (
                float(score["expected_mean_log_probability"]),
                float(score["alternative_mean_log_probability"]),
                float(score["log_posterior_margin"]),
                float(np.log1p(int(score["frame_count"]))),
            )
            if not np.isfinite(continuous).all():
                raise ValueError("L2-ARCTIC phone features must be finite")
            examples.append(
                PhoneExample(
                    speaker=str(item["speaker"]),
                    first_language=str(item["first_language"]),
                    utterance_id=str(item["utterance_id"]),
                    word_index=int(score["word_index"]),
                    expected=str(score["expected"]),
                    perceived=(str(score["perceived"]) if score["perceived"] is not None else None),
                    alternative=str(score["alternative"]),
                    error_type=str(score["error_type"]),
                    continuous=continuous,
                )
            )
    if not examples:
        raise ValueError("L2-ARCTIC evidence contains no phone examples")
    return tuple(examples)


def _fit_model(
    examples: tuple[PhoneExample, ...],
    categories: tuple[str, ...],
    alternative_categories: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    continuous = np.asarray([example.continuous for example in examples], dtype=np.float64)
    means = np.mean(continuous, axis=0)
    scales = np.std(continuous, axis=0)
    scales = np.asarray(np.where(scales < 1e-8, 1.0, scales), dtype=np.float64)
    matrix = _feature_matrix(examples, categories, alternative_categories, means, scales)
    labels = np.asarray([example.error for example in examples], dtype=np.float64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("L2-ARCTIC calibration requires correct and incorrect phones")
    initial = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    result = minimize(
        _logistic_objective,
        initial,
        args=(matrix, labels),
        jac=True,
        method="L-BFGS-B",
    )
    if not bool(result.success):
        raise RuntimeError(f"L2-ARCTIC logistic calibration failed: {result.message}")
    return means, scales, np.asarray(result.x, dtype=np.float64)


def _feature_matrix(
    examples: tuple[PhoneExample, ...],
    categories: tuple[str, ...],
    alternative_categories: tuple[str, ...],
    means: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    continuous = np.asarray([example.continuous for example in examples], dtype=np.float64)
    standardized = (continuous - means) / scales
    expected_indexes = {phone: index for index, phone in enumerate(categories)}
    alternative_indexes = {phone: index for index, phone in enumerate(alternative_categories)}
    expected_other = expected_indexes["<other>"]
    alternative_other = alternative_indexes["<other>"]
    expected = np.zeros((len(examples), len(categories)), dtype=np.float64)
    alternative = np.zeros((len(examples), len(alternative_categories)), dtype=np.float64)
    for row, example in enumerate(examples):
        expected[row, expected_indexes.get(example.expected, expected_other)] = 1.0
        alternative[row, alternative_indexes.get(example.alternative, alternative_other)] = 1.0
    return np.concatenate((standardized, expected, alternative), axis=1)


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


def _predict_examples(
    examples: tuple[PhoneExample, ...],
    categories: tuple[str, ...],
    alternative_categories: tuple[str, ...],
    means: NDArray[np.float64],
    scales: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = _feature_matrix(examples, categories, alternative_categories, means, scales)
    return np.asarray(expit(coefficients[0] + matrix @ coefficients[1:]), dtype=np.float64)


def _phone_metrics(
    examples: tuple[PhoneExample, ...], probabilities: NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    labels = np.asarray([example.error for example in examples], dtype=np.bool_)
    flagged = probabilities >= threshold
    metrics = _binary_metrics(labels, flagged)
    metrics["threshold"] = threshold
    return metrics


def _word_metrics(
    examples: tuple[PhoneExample, ...], probabilities: NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[(example.speaker, example.utterance_id, example.word_index)].append(index)
    labels = np.asarray(
        [any(examples[index].error for index in indexes) for indexes in groups.values()],
        dtype=np.bool_,
    )
    flagged = np.asarray(
        [
            max(probabilities[index] for index in indexes) >= threshold
            for indexes in groups.values()
        ],
        dtype=np.bool_,
    )
    metrics = _binary_metrics(labels, flagged)
    metrics["threshold"] = threshold
    return metrics


def _diagnosis_metrics(
    examples: tuple[PhoneExample, ...], probabilities: NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    eligible = [
        (example, probability)
        for example, probability in zip(examples, probabilities, strict=True)
        if example.error_type == "substitution"
        and example.perceived is not None
        and probability >= threshold
    ]
    exact = sum(example.alternative == example.perceived for example, _ in eligible)
    selected_contrasts = {}
    for expected, perceived in (("θ", "f"), ("θ", "s"), ("ɹ", "l"), ("l", "ɹ")):
        relevant = [
            (example, probability)
            for example, probability in zip(examples, probabilities, strict=True)
            if example.expected == expected and example.perceived == perceived
        ]
        selected_contrasts[f"{expected}->{perceived}"] = {
            "total": len(relevant),
            "flagged": int(sum(probability >= threshold for _, probability in relevant)),
            "alternative_matched": int(
                sum(
                    probability >= threshold and example.alternative == perceived
                    for example, probability in relevant
                )
            ),
        }
    return {
        "eligible_flagged_substitutions": len(eligible),
        "exact_alternative_matches": exact,
        "exact_alternative_accuracy": exact / len(eligible) if eligible else 0.0,
        "selected_contrasts": selected_contrasts,
    }


def _binary_metrics(labels: NDArray[np.bool_], flagged: NDArray[np.bool_]) -> dict[str, Any]:
    true_positives = int(np.sum(labels & flagged))
    false_positives = int(np.sum(~labels & flagged))
    false_negatives = int(np.sum(labels & ~flagged))
    flagged_count = true_positives + false_positives
    positive_count = true_positives + false_negatives
    return {
        "examples": len(labels),
        "positive_examples": positive_count,
        "flagged": flagged_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / flagged_count if flagged_count else 0.0,
        "recall": true_positives / positive_count if positive_count else 0.0,
    }


def _accumulate_set_metrics(totals: list[int], predictions: set[int], truth: set[int]) -> None:
    totals[0] += len(predictions & truth)
    totals[1] += len(predictions - truth)
    totals[2] += len(truth - predictions)


def _set_metrics(totals: list[int]) -> dict[str, float | int]:
    true_positives, false_positives, false_negatives = totals
    predicted = true_positives + false_positives
    positive = true_positives + false_negatives
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / predicted if predicted else 0.0,
        "recall": true_positives / positive if positive else 0.0,
    }


def _select_threshold(
    curve: list[dict[str, Any]], minimum_precision: float
) -> dict[str, Any] | None:
    eligible = [point for point in curve if float(point["precision"]) >= minimum_precision]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (
            float(point["recall"]),
            float(point["precision"]),
            float(point["threshold"]),
        ),
    )


def select_word_threshold(
    curve: list[dict[str, Any]], minimum_precision: float, minimum_recall: float
) -> dict[str, Any] | None:
    eligible = [
        point
        for point in curve
        if float(point["precision"]) >= minimum_precision
        and float(point["recall"]) > minimum_recall
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (
            float(point["recall"]),
            float(point["precision"]),
            float(point["threshold"]),
        ),
    )


def _model_parameters(
    model: dict[str, Any],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    if model["version"] != 1 or model["method"] != "l2-logistic-regression":
        raise ValueError("Unsupported L2-ARCTIC phone model")
    if tuple(model["continuous_feature_names"]) != CONTINUOUS_FEATURE_NAMES:
        raise ValueError("L2-ARCTIC phone model uses different continuous features")
    categories = tuple(str(value) for value in model["expected_phone_categories"])
    alternative_categories = tuple(str(value) for value in model["alternative_phone_categories"])
    means = np.asarray(model["continuous_means"], dtype=np.float64)
    scales = np.asarray(model["continuous_scales"], dtype=np.float64)
    coefficient_values = np.asarray(model["coefficients"], dtype=np.float64)
    expected_coefficients = (
        len(CONTINUOUS_FEATURE_NAMES) + len(categories) + len(alternative_categories)
    )
    if len(means) != len(CONTINUOUS_FEATURE_NAMES) or len(scales) != len(CONTINUOUS_FEATURE_NAMES):
        raise ValueError("L2-ARCTIC phone model has invalid continuous parameters")
    if len(coefficient_values) != expected_coefficients:
        raise ValueError("L2-ARCTIC phone model has invalid coefficient count")
    threshold_value = model["red_probability_threshold"]
    if not isinstance(threshold_value, int | float):
        raise ValueError("L2-ARCTIC phone model has no red threshold")
    coefficients = np.concatenate((np.asarray([float(model["intercept"])]), coefficient_values))
    return (
        categories,
        alternative_categories,
        means,
        scales,
        coefficients,
        float(threshold_value),
    )


def _validate_evidence_report(report: dict[str, Any], partition: Partition) -> None:
    if report["experiment"] != "l2-arctic-lv60-phone-evidence-v1":
        raise ValueError("Unexpected L2-ARCTIC evidence experiment")
    if report["partition"] != partition:
        raise ValueError(f"Expected L2-ARCTIC {partition} evidence")
    artifacts = report["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("L2-ARCTIC evidence lacks artifacts")
    if artifacts["model_id"] != MODEL_ID or artifacts["model_revision"] != MODEL_REVISION:
        raise ValueError("L2-ARCTIC evidence uses a different acoustic model")


def _required_expected_ipa(phone: Any) -> str:
    expected = phone.expected_ipa
    if expected is None:
        raise ValueError(f"Unmapped expected L2-ARCTIC phone: {phone.expected_arpabet}")
    return expected


def _required_word_index(phone: Any) -> int:
    if phone.word_index is None:
        raise ValueError("L2-ARCTIC expected phone is outside every word interval")
    return int(phone.word_index)
