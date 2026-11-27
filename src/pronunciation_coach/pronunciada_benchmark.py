"""Word-level corpus benchmarks for the frozen Pronunciada UI policy."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from pronunciation_coach.audio import load_audio
from pronunciation_coach.l2arctic import (
    L2ArcticSplitManifest,
    Partition,
    iter_l2arctic_utterances,
)

if TYPE_CHECKING:
    from pronunciation_coach.benchmarking import HeldoutManifest
from pronunciation_coach.pronunciada_analysis import (
    REVIEW_ACCEPTANCE_THRESHOLD,
    STRICTNESS,
    WARNING_ACCEPTANCE_THRESHOLD,
    pin_pronunciada_download_revision,
    verify_pronunciada_assets,
)


class _WordScore(Protocol):
    text: str
    prob: float


class _SentenceScore(Protocol):
    @property
    def words(self) -> Sequence[_WordScore]: ...


class PronunciadaScorer(Protocol):
    def score(self, audio: object, text: str, strictness: float) -> _SentenceScore: ...


class _Dataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


def create_scorer(cache_dir: Path, *, precision: str, threads: int) -> PronunciadaScorer:
    """Load one scorer for a complete benchmark run."""
    os.environ["PRONUNCIADA_CACHE"] = str(cache_dir.resolve())
    pin_pronunciada_download_revision()
    scorer_module = import_module("pronunciada.scorer")
    scorer_type: Any = scorer_module.Scorer
    scorer = cast(PronunciadaScorer, scorer_type("en", precision=precision, threads=threads))
    verify_pronunciada_assets(cache_dir, precision)
    return scorer


def evaluate_speechocean_final(
    scorer: PronunciadaScorer,
    manifest: HeldoutManifest,
    *,
    dataset_cache: Path,
) -> dict[str, Any]:
    """Evaluate the frozen SpeechOcean final-test selection."""
    benchmarking: Any = import_module("pronunciation_coach.benchmarking")
    load_speechocean = cast(Callable[[str, Path], _Dataset], benchmarking.load_speechocean)

    wanted = {item.utterance_id: item for item in manifest.items}
    rows: dict[str, dict[str, Any]] = {}
    dataset = load_speechocean(manifest.split, dataset_cache)
    for index in range(len(dataset)):
        row = cast(dict[str, Any], dataset[index])
        utterance_id = _audio_id(cast(dict[str, Any], row["audio"]))
        if utterance_id in wanted:
            rows[utterance_id] = row
    missing = sorted(set(wanted) - set(rows))
    if missing:
        raise ValueError(f"SpeechOcean is missing {len(missing)} manifest recordings")

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for manifest_item in manifest.items:
        row = rows[manifest_item.utterance_id]
        try:
            raw_words = cast(list[dict[str, Any]], row["words"])
            truth = {
                index
                for index, word in enumerate(raw_words)
                if any(
                    float(cast(float | int, value)) < 1.0
                    for value in cast(list[object], word["phones-accuracy"])
                )
            }
            audio = cast(dict[str, Any], row["audio"])
            item = _score_item(
                scorer,
                _audio_source(audio),
                str(row["text"]),
                [str(word["text"]) for word in raw_words],
                truth,
            )
            completed.append(
                {
                    "utterance_id": manifest_item.utterance_id,
                    "speaker": manifest_item.speaker,
                    "age_group": manifest_item.age_group,
                    "gender": manifest_item.gender,
                    "accuracy": int(row["accuracy"]),
                    **item,
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append({"utterance_id": manifest_item.utterance_id, "error": str(error)})

    return build_word_benchmark_report(
        "speechocean762-final-test",
        completed,
        failures,
        experiment="pronunciada-word-error-benchmark-v1",
        policy={
            "strictness": STRICTNESS,
            "warning_acceptance_below": WARNING_ACCEPTANCE_THRESHOLD,
            "red_acceptance_below": REVIEW_ACCEPTANCE_THRESHOLD,
        },
        truth_definition="word has at least one expert phone-accuracy score below 1.0",
        subgroups=("age_group", "gender"),
        expected_recordings=len(manifest.items),
    )


def evaluate_l2arctic(
    scorer: PronunciadaScorer,
    dataset_root: Path,
    manifest: L2ArcticSplitManifest,
) -> dict[str, Any]:
    """Evaluate every L2-ARCTIC recording while retaining speaker partitions."""
    all_items: list[dict[str, Any]] = []
    all_failures: list[dict[str, str]] = []
    partition_reports: dict[str, dict[str, Any]] = {}
    partitions: tuple[Partition, ...] = ("train", "development", "final-test")

    for partition in partitions:
        completed: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        speakers = manifest.speakers_for(partition)
        for utterance in iter_l2arctic_utterances(dataset_root, speakers, include_audio=True):
            try:
                if utterance.audio_bytes is None:
                    raise ValueError("L2-ARCTIC audio was not loaded")
                truth = {
                    phone.word_index
                    for phone in utterance.phones
                    if phone.error_type != "correct" and phone.word_index is not None
                }
                item = _score_item(
                    scorer,
                    utterance.audio_bytes,
                    utterance.transcript,
                    utterance.words,
                    truth,
                )
                completed.append(
                    {
                        "utterance_id": utterance.utterance_id,
                        "speaker": utterance.speaker.id,
                        "gender": utterance.speaker.gender,
                        "first_language": utterance.speaker.first_language,
                        "partition": partition,
                        **item,
                    }
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                failures.append(
                    {
                        "utterance_id": f"{utterance.speaker.id}/{utterance.utterance_id}",
                        "error": str(error),
                    }
                )
        all_items.extend(completed)
        all_failures.extend(failures)
        partition_reports[partition] = _metrics(completed, failures)

    report = build_word_benchmark_report(
        "l2-arctic-v5.0-all-partitions",
        all_items,
        all_failures,
        experiment="pronunciada-word-error-benchmark-v1",
        policy={
            "strictness": STRICTNESS,
            "warning_acceptance_below": WARNING_ACCEPTANCE_THRESHOLD,
            "red_acceptance_below": REVIEW_ACCEPTANCE_THRESHOLD,
        },
        truth_definition="word has at least one annotated substitution, deletion, or addition",
        subgroups=("partition", "first_language", "gender"),
        expected_recordings=len(all_items) + len(all_failures),
    )
    report["metrics"]["partitions"] = partition_reports
    return report


def classification_metrics(
    items: Iterable[Mapping[str, Any]], prediction_key: str
) -> dict[str, int | float]:
    """Return word-level binary classification metrics."""
    true_positive = false_positive = false_negative = true_negative = 0
    for item in items:
        truth = set(cast(Sequence[int], item["truth_word_indexes"]))
        predicted = set(cast(Sequence[int], item[prediction_key]))
        word_count = len(cast(Sequence[object], item["words"]))
        true_positive += len(truth & predicted)
        false_positive += len(predicted - truth)
        false_negative += len(truth - predicted)
        true_negative += word_count - len(truth | predicted)
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    return {
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": _rate(2.0 * precision * recall, precision + recall),
        "false_positive_rate": _rate(false_positive, false_positive + true_negative),
    }


def _score_item(
    scorer: PronunciadaScorer,
    audio_source: bytes | str,
    text: str,
    expected_words: Sequence[str],
    truth: set[int],
) -> dict[str, Any]:
    audio = load_audio(audio_source)
    started = perf_counter()
    result = scorer.score(cast(Any, audio).samples, text, strictness=STRICTNESS)
    latency_ms = (perf_counter() - started) * 1000.0
    actual_words = [word.text for word in result.words]
    mapping, alignment_cost = word_index_mapping(expected_words, actual_words)
    if alignment_cost > max(2, round(len(expected_words) * 0.4)):
        raise ValueError(
            f"Pronunciada words differ too much from labels: {actual_words!r} != "
            f"{list(expected_words)!r}"
        )
    if any(index < 0 or index >= len(expected_words) for index in truth):
        raise ValueError("Truth label contains an invalid word index")
    mapped_truth = {mapping[index] for index in truth if mapping[index] is not None}
    unmapped_truth = sorted(index for index in truth if mapping[index] is None)

    probabilities = [float(word.prob) for word in result.words]
    red = [
        index
        for index, probability in enumerate(probabilities)
        if probability < REVIEW_ACCEPTANCE_THRESHOLD
    ]
    warning = [
        index
        for index, probability in enumerate(probabilities)
        if probability < WARNING_ACCEPTANCE_THRESHOLD
    ]
    return {
        "text": text,
        "words": actual_words,
        "label_words": list(expected_words),
        "word_alignment_cost": alignment_cost,
        "acceptance_probabilities": probabilities,
        "truth_word_indexes": sorted(cast(set[int], mapped_truth)),
        "unmapped_truth_word_indexes": unmapped_truth,
        "red_word_indexes": red,
        "all_highlight_word_indexes": warning,
        "inference_ms": latency_ms,
    }


def build_word_benchmark_report(
    dataset: str,
    items: list[dict[str, Any]],
    failures: list[dict[str, str]],
    *,
    experiment: str,
    policy: dict[str, Any],
    truth_definition: str,
    subgroups: tuple[str, ...],
    expected_recordings: int,
) -> dict[str, Any]:
    metrics = _metrics(items, failures)
    metrics["expected_recordings"] = expected_recordings
    metrics["completion_rate"] = _rate(len(items), expected_recordings)
    metrics["subgroups"] = {
        field: {
            value: _metrics([item for item in items if str(item[field]) == value], [])
            for value in sorted({str(item[field]) for item in items})
        }
        for field in subgroups
    }
    return {
        "report_version": 1,
        "experiment": experiment,
        "dataset": dataset,
        "truth_definition": truth_definition,
        "policy": policy,
        "metrics": metrics,
        "failures": failures,
        "items": items,
    }


def _metrics(items: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    latencies = sorted(float(item["inference_ms"]) for item in items)
    return {
        "completed_recordings": len(items),
        "failed_recordings": len(failures),
        "words": sum(len(cast(list[object], item["words"])) for item in items),
        "expert_problem_words": sum(
            len(cast(list[int], item["truth_word_indexes"])) for item in items
        ),
        "excluded_unmapped_problem_words": sum(
            len(cast(list[int], item["unmapped_truth_word_indexes"])) for item in items
        ),
        "all_highlights": classification_metrics(items, "all_highlight_word_indexes"),
        "red_highlights": classification_metrics(items, "red_word_indexes"),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def _audio_id(audio: dict[str, Any]) -> str:
    path = audio.get("path")
    if not isinstance(path, str):
        raise ValueError("SpeechOcean audio path is missing")
    return Path(path).stem


def _audio_source(audio: dict[str, Any]) -> bytes | str:
    audio_bytes = audio.get("bytes")
    if isinstance(audio_bytes, bytes):
        return audio_bytes
    path = audio.get("path")
    if isinstance(path, str):
        return path
    raise ValueError("SpeechOcean audio has neither bytes nor path")


def _normalize_word(word: str) -> str:
    normalized = word.casefold().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    return re.sub(r"(^[^a-z0-9']+|[^a-z0-9']+$)", "", normalized)


def word_index_mapping(
    expected_words: Sequence[str], actual_words: Sequence[str]
) -> tuple[list[int | None], int]:
    """Map labelled word indexes to scorer indexes with Levenshtein alignment."""
    expected = [_normalize_word(word) for word in expected_words]
    actual = [_normalize_word(word) for word in actual_words]
    rows = len(expected) + 1
    columns = len(actual) + 1
    costs = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        moves[row][0] = "delete"
    for column in range(1, columns):
        costs[0][column] = column
        moves[0][column] = "insert"

    for row in range(1, rows):
        for column in range(1, columns):
            substitution = costs[row - 1][column - 1] + (expected[row - 1] != actual[column - 1])
            deletion = costs[row - 1][column] + 1
            insertion = costs[row][column - 1] + 1
            candidates = (
                (substitution, 0, "match"),
                (deletion, 1, "delete"),
                (insertion, 2, "insert"),
            )
            cost, _, move = min(candidates)
            costs[row][column] = cost
            moves[row][column] = move

    mapping: list[int | None] = [None] * len(expected)
    row = len(expected)
    column = len(actual)
    while row or column:
        move = moves[row][column]
        if move == "match":
            mapping[row - 1] = column - 1
            row -= 1
            column -= 1
        elif move == "delete":
            row -= 1
        elif move == "insert":
            column -= 1
        else:
            raise RuntimeError("Word alignment backtrace is incomplete")
    return mapping, costs[-1][-1]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    index = round((len(values) - 1) * percentile)
    return values[index]


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0
