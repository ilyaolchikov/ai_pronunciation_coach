# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Frozen-manifest preparation and Stage 1 feasibility metrics."""

import hashlib
import json
import platform
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset
from pydantic import Field, TypeAdapter
from scipy.stats import spearmanr

from pronunciation_coach.alignment import align_prompt
from pronunciation_coach.audio import load_audio
from pronunciation_coach.config import (
    DEFAULT_DATASET_CACHE,
    DEFAULT_MODEL_CACHE,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOTS,
    SPEECHOCEAN_DATASET_ID,
    SPEECHOCEAN_REVISION,
)
from pronunciation_coach.phone_mapping import map_arpabet_sequence
from pronunciation_coach.phonemization import phonemize_words
from pronunciation_coach.recognizers.wav2vec2_espeak import Wav2Vec2EspeakRecognizer
from pronunciation_coach.schemas import ContractModel, Operation, Prompt, PromptWord
from pronunciation_coach.word_calibration import evaluate_word_model

POSTERIOR_MARGIN_THRESHOLDS = (-20.0, -15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0)


class ManifestItem(ContractModel):
    utterance_id: str
    speaker: str
    age_group: str
    accuracy_band: str
    accuracy: int = Field(ge=0, le=10)


class BenchmarkManifest(ContractModel):
    version: int = 1
    dataset_id: str
    dataset_revision: str
    split: str
    selection_seed: int
    unique_speakers: bool
    items: tuple[ManifestItem, ...]


class CompetingPhoneInventory(ContractModel):
    version: int = Field(ge=1)
    dataset_id: str
    dataset_revision: str
    split: Literal["train"]
    locale: Literal["en-us"]
    source: Literal["unique-word-types-in-pinned-training-transcripts"]
    phonemizer: str
    espeak_ng: str
    phones: tuple[str, ...] = Field(min_length=2)


class DecisionPolicy(ContractModel):
    version: int
    frozen_before_results: bool
    completion_rate_min: float
    spearman_point_min_exclusive: float
    high_rating_accuracy_min: int
    high_rating_false_review_rate_max: float
    interpretable_problem_min: int
    latency_p95_ms_max: int
    peak_rss_mb_max: int
    mapping_coverage_min: float
    bootstrap_samples: int
    bootstrap_seed: int


class HeldoutManifestItem(ContractModel):
    utterance_id: str
    speaker: str
    age_group: Literal["child", "adult"]
    gender: Literal["f", "m"]


class HeldoutManifest(ContractModel):
    version: int = 1
    dataset_id: str
    dataset_revision: str
    split: Literal["test"]
    partition: Literal["heldout-validation", "final-test"]
    selection_seed: int
    selection_algorithm: str
    speaker_count: int = Field(gt=0)
    items: tuple[HeldoutManifestItem, ...] = Field(min_length=1)


class PosteriorEvaluationPolicy(ContractModel):
    version: int
    frozen_before_heldout_results: bool
    scorer: Literal["constrained-ctc-viterbi-v1"]
    model_id: str
    model_revision: str
    phone_margin_threshold: float
    word_review_rule: Literal["any-phone-margin-below-threshold"]
    recording_review_rule: Literal["any-word-reviewed"]
    completion_rate_min: float = Field(ge=0.0, le=1.0)
    high_rating_accuracy_min: int = Field(ge=0, le=10)
    high_rating_false_review_rate_max: float = Field(ge=0.0, le=1.0)
    word_precision_min: float = Field(ge=0.0, le=1.0)
    word_recall_min: float = Field(ge=0.0, le=1.0)
    mean_margin_spearman_min_exclusive: float
    p10_margin_spearman_min_exclusive: float
    spearman_bootstrap_lower_min_exclusive: float
    latency_p95_ms_max: int = Field(gt=0)
    peak_rss_mb_max: int = Field(gt=0)
    mapping_coverage_min: float = Field(ge=0.0, le=1.0)
    bootstrap_samples: int = Field(gt=0)
    bootstrap_seed: int
    competing_phone_inventory_sha256: str | None = None
    include_speechocean_variant: bool = False
    development_report_sha256: str | None = None


def load_speechocean(split: str, cache_dir: Path = DEFAULT_DATASET_CACHE) -> Dataset:
    dataset = load_dataset(
        SPEECHOCEAN_DATASET_ID,
        revision=SPEECHOCEAN_REVISION,
        split=split,
        cache_dir=str(cache_dir),
    )
    return dataset.cast_column("audio", Audio(decode=False))


def prepare_manifest(dataset: Dataset, *, size: int = 50, seed: int = 762) -> BenchmarkManifest:
    if size != 50:
        raise ValueError("Stage 1 manifest size is frozen at 50")

    strata: dict[tuple[str, str], list[ManifestItem]] = defaultdict(list)
    for raw in dataset:
        row = cast(dict[str, Any], raw)
        item = ManifestItem(
            utterance_id=_utterance_id(cast(dict[str, Any], row["audio"])),
            speaker=str(row["speaker"]),
            age_group="child" if int(row["age"]) < 18 else "adult",
            accuracy_band=_accuracy_band(int(row["accuracy"])),
            accuracy=int(row["accuracy"]),
        )
        strata[(item.age_group, item.accuracy_band)].append(item)

    stratum_keys = (
        ("child", "low"),
        ("child", "mid"),
        ("child", "high"),
        ("adult", "low"),
        ("adult", "mid"),
        ("adult", "high"),
    )
    quotas = (9, 8, 8, 9, 8, 8)
    selected: list[ManifestItem] = []
    used_speakers: set[str] = set()
    for key, quota in zip(stratum_keys, quotas, strict=True):
        candidates = sorted(
            strata[key],
            key=lambda item: _selection_hash(item.utterance_id, seed),
        )
        for candidate in candidates:
            if candidate.speaker in used_speakers:
                continue
            selected.append(candidate)
            used_speakers.add(candidate.speaker)
            count = sum(
                item.age_group == key[0] and item.accuracy_band == key[1] for item in selected
            )
            if count == quota:
                break
        actual = sum(item.age_group == key[0] and item.accuracy_band == key[1] for item in selected)
        if actual != quota:
            message = f"Could not fill frozen stratum {key}: expected {quota}, got {actual}"
            raise RuntimeError(message)

    return BenchmarkManifest(
        dataset_id=SPEECHOCEAN_DATASET_ID,
        dataset_revision=SPEECHOCEAN_REVISION,
        split="train",
        selection_seed=seed,
        unique_speakers=True,
        items=tuple(sorted(selected, key=lambda item: item.utterance_id)),
    )


def prepare_calibration_manifest(dataset: Dataset, *, seed: int = 762) -> BenchmarkManifest:
    """Select one label-blind recording from every training speaker."""
    rows_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in dataset:
        row = cast(dict[str, Any], raw)
        rows_by_speaker[str(row["speaker"])].append(row)
    if len(rows_by_speaker) < 5:
        raise ValueError("Calibration requires at least five speakers")

    selected: list[ManifestItem] = []
    for speaker, rows in rows_by_speaker.items():
        row = min(
            rows,
            key=lambda candidate: _selection_hash(
                _utterance_id(cast(dict[str, Any], candidate["audio"])), seed
            ),
        )
        accuracy = int(row["accuracy"])
        selected.append(
            ManifestItem(
                utterance_id=_utterance_id(cast(dict[str, Any], row["audio"])),
                speaker=speaker,
                age_group="child" if int(row["age"]) < 18 else "adult",
                accuracy_band=_accuracy_band(accuracy),
                accuracy=accuracy,
            )
        )

    return BenchmarkManifest(
        dataset_id=SPEECHOCEAN_DATASET_ID,
        dataset_revision=SPEECHOCEAN_REVISION,
        split="train",
        selection_seed=seed,
        unique_speakers=True,
        items=tuple(sorted(selected, key=lambda item: item.utterance_id)),
    )


def prepare_heldout_manifests(
    dataset: Dataset, *, seed: int = 762
) -> tuple[HeldoutManifest, HeldoutManifest]:
    """Partition official test speakers without consulting scores or model output."""
    rows_by_speaker: dict[str, list[HeldoutManifestItem]] = defaultdict(list)
    metadata_by_speaker: dict[str, tuple[str, str]] = {}
    for raw in dataset:
        row = cast(dict[str, Any], raw)
        speaker = str(row["speaker"])
        age_group = "child" if int(row["age"]) < 18 else "adult"
        gender = str(row["gender"])
        if gender not in {"f", "m"}:
            raise ValueError(f"Unsupported gender value for speaker partition: {gender!r}")
        metadata = (age_group, gender)
        previous = metadata_by_speaker.setdefault(speaker, metadata)
        if previous != metadata:
            raise ValueError(f"Inconsistent metadata for speaker {speaker}")
        rows_by_speaker[speaker].append(
            HeldoutManifestItem(
                utterance_id=_utterance_id(cast(dict[str, Any], row["audio"])),
                speaker=speaker,
                age_group=age_group,
                gender=cast(Literal["f", "m"], gender),
            )
        )
    if len(rows_by_speaker) < 2:
        raise ValueError("Held-out partition requires at least two speakers")

    speakers_by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for speaker, metadata in metadata_by_speaker.items():
        speakers_by_stratum[metadata].append(speaker)
    validation_target = len(rows_by_speaker) // 2
    quotas = _largest_remainder_quotas(speakers_by_stratum, validation_target)
    validation_speakers: set[str] = set()
    for stratum, speakers in speakers_by_stratum.items():
        ordered = sorted(speakers, key=lambda speaker: _selection_hash(speaker, seed))
        validation_speakers.update(ordered[: quotas[stratum]])
    final_speakers = set(rows_by_speaker) - validation_speakers
    if len(validation_speakers) != validation_target:
        raise RuntimeError("Held-out speaker quota calculation produced the wrong size")

    algorithm = "speaker-age-gender-largest-remainder-sha256-v1"
    validation = _heldout_manifest_for_speakers(
        rows_by_speaker,
        validation_speakers,
        partition="heldout-validation",
        seed=seed,
        algorithm=algorithm,
    )
    final = _heldout_manifest_for_speakers(
        rows_by_speaker,
        final_speakers,
        partition="final-test",
        seed=seed,
        algorithm=algorithm,
    )
    return validation, final


def _largest_remainder_quotas(
    speakers_by_stratum: dict[tuple[str, str], list[str]], target: int
) -> dict[tuple[str, str], int]:
    total = sum(len(speakers) for speakers in speakers_by_stratum.values())
    exact = {
        stratum: len(speakers) * target / total for stratum, speakers in speakers_by_stratum.items()
    }
    quotas = {stratum: int(value) for stratum, value in exact.items()}
    remaining = target - sum(quotas.values())
    ranked = sorted(exact, key=lambda stratum: (-(exact[stratum] % 1), stratum))
    for stratum in ranked[:remaining]:
        quotas[stratum] += 1
    return quotas


def _heldout_manifest_for_speakers(
    rows_by_speaker: dict[str, list[HeldoutManifestItem]],
    speakers: set[str],
    *,
    partition: Literal["heldout-validation", "final-test"],
    seed: int,
    algorithm: str,
) -> HeldoutManifest:
    items = tuple(
        sorted(
            (item for speaker in speakers for item in rows_by_speaker[speaker]),
            key=lambda item: item.utterance_id,
        )
    )
    return HeldoutManifest(
        dataset_id=SPEECHOCEAN_DATASET_ID,
        dataset_revision=SPEECHOCEAN_REVISION,
        split="test",
        partition=partition,
        selection_seed=seed,
        selection_algorithm=algorithm,
        speaker_count=len(speakers),
        items=items,
    )


def run_benchmark(
    manifest: BenchmarkManifest,
    policy: DecisionPolicy,
    *,
    dataset_cache: Path = DEFAULT_DATASET_CACHE,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str = "cpu",
) -> dict[str, Any]:
    dataset = load_speechocean(manifest.split, dataset_cache)
    wanted = {item.utterance_id for item in manifest.items}
    rows = {
        _utterance_id(cast(dict[str, Any], cast(dict[str, Any], row)["audio"])): cast(
            dict[str, Any], row
        )
        for row in dataset
        if _utterance_id(cast(dict[str, Any], cast(dict[str, Any], row)["audio"])) in wanted
    }
    if set(rows) != wanted:
        raise RuntimeError("Dataset does not contain every frozen manifest item")

    recognizer = Wav2Vec2EspeakRecognizer(cache_dir=model_cache, device=device)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    mapped_count = 0
    source_phone_count = 0

    for manifest_item in manifest.items:
        row = rows[manifest_item.utterance_id]
        try:
            prompt, mapped, total = _prompt_from_row(manifest_item.utterance_id, row)
            mapped_count += mapped
            source_phone_count += total
            audio = _audio_from_row(cast(dict[str, Any], row["audio"]))
            recognition = recognizer.recognize(audio)
            alignment = align_prompt(prompt, recognition.phones)
            expert_problem_words = _expert_problem_word_indexes(row)
            detected_problem_words = {
                item.word_index
                for item in alignment.evidence
                if item.word_index is not None and item.operation is not Operation.MATCH
            }
            completed.append(
                {
                    "utterance_id": manifest_item.utterance_id,
                    "accuracy": int(row["accuracy"]),
                    "similarity": 1
                    - alignment.cost
                    / max(len(alignment.expected_phones), len(recognition.phones), 1),
                    "alignment_cost": alignment.cost,
                    "review": alignment.cost > 0,
                    "inference_ms": recognition.inference_ms,
                    "expert_problem_words": len(expert_problem_words),
                    "detected_expert_problem_words": len(
                        expert_problem_words & detected_problem_words
                    ),
                    "expected_phones": alignment.expected_phones,
                    "observed_phones": recognition.phones,
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append({"utterance_id": manifest_item.utterance_id, "error": str(error)})

    return _build_report(
        manifest=manifest,
        policy=policy,
        completed=completed,
        failures=failures,
        mapped_count=mapped_count,
        source_phone_count=source_phone_count,
        device=device,
    )


def run_posterior_benchmark(
    manifest: BenchmarkManifest,
    policy: DecisionPolicy,
    *,
    dataset_cache: Path = DEFAULT_DATASET_CACHE,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str = "cpu",
    competing_phone_inventory: CompetingPhoneInventory | None = None,
    competing_phone_inventory_sha256: str | None = None,
    include_speechocean_variant: bool = False,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    """Evaluate constrained CTC phone margins without claiming held-out validation."""
    if (competing_phone_inventory is None) != (competing_phone_inventory_sha256 is None):
        raise ValueError("competing phone inventory and SHA-256 must be provided together")
    completed, failures, mapped_count, source_phone_count = _collect_posterior_items(
        tuple(item.utterance_id for item in manifest.items),
        split=manifest.split,
        dataset_cache=dataset_cache,
        model_cache=model_cache,
        device=device,
        competing_phones=(
            frozenset(competing_phone_inventory.phones)
            if competing_phone_inventory is not None
            else None
        ),
        include_speechocean_variant=include_speechocean_variant,
        model_id=model_id,
        model_revision=model_revision,
    )

    return _build_posterior_report(
        manifest=manifest,
        policy=policy,
        completed=completed,
        failures=failures,
        mapped_count=mapped_count,
        source_phone_count=source_phone_count,
        model_cache=model_cache,
        device=device,
        competing_phone_inventory=competing_phone_inventory,
        competing_phone_inventory_sha256=competing_phone_inventory_sha256,
        include_speechocean_variant=include_speechocean_variant,
        model_id=model_id,
        model_revision=model_revision,
    )


def run_posterior_heldout_benchmark(
    manifest: HeldoutManifest,
    policy: PosteriorEvaluationPolicy,
    *,
    manifest_sha256: str,
    policy_sha256: str,
    dataset_cache: Path = DEFAULT_DATASET_CACHE,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str = "cpu",
    competing_phone_inventory: CompetingPhoneInventory | None = None,
    competing_phone_inventory_sha256: str | None = None,
    include_speechocean_variant: bool = False,
) -> dict[str, Any]:
    """Run one fixed policy without threshold selection or alternative curves."""
    _validate_heldout_manifest(manifest)
    if manifest.partition != "heldout-validation":
        raise ValueError("The held-out runner refuses the locked final-test partition")
    if (
        manifest.dataset_id != SPEECHOCEAN_DATASET_ID
        or manifest.dataset_revision != SPEECHOCEAN_REVISION
    ):
        raise ValueError("Held-out manifest does not match the pinned dataset")
    if not policy.frozen_before_heldout_results:
        raise ValueError("Posterior policy must be frozen before held-out evaluation")
    if (policy.model_id, policy.model_revision) not in MODEL_SNAPSHOTS.values():
        raise ValueError("Posterior policy does not match the pinned acoustic model")
    if policy.competing_phone_inventory_sha256 != competing_phone_inventory_sha256:
        raise ValueError("Competing phone inventory does not match the frozen policy")
    if (competing_phone_inventory is None) != (competing_phone_inventory_sha256 is None):
        raise ValueError("competing phone inventory and SHA-256 must be provided together")
    if policy.include_speechocean_variant != include_speechocean_variant:
        raise ValueError("Canonical pronunciation variants do not match the frozen policy")

    completed, failures, mapped_count, source_phone_count = _collect_posterior_items(
        tuple(item.utterance_id for item in manifest.items),
        split=manifest.split,
        dataset_cache=dataset_cache,
        model_cache=model_cache,
        device=device,
        competing_phones=(
            frozenset(competing_phone_inventory.phones)
            if competing_phone_inventory is not None
            else None
        ),
        include_speechocean_variant=include_speechocean_variant,
        model_id=policy.model_id,
        model_revision=policy.model_revision,
    )
    return _build_posterior_heldout_report(
        manifest=manifest,
        policy=policy,
        manifest_sha256=manifest_sha256,
        policy_sha256=policy_sha256,
        completed=completed,
        failures=failures,
        mapped_count=mapped_count,
        source_phone_count=source_phone_count,
        model_cache=model_cache,
        device=device,
        competing_phone_inventory=competing_phone_inventory,
    )


def run_calibrated_word_heldout_benchmark(
    manifest: HeldoutManifest,
    calibration_report: dict[str, Any],
    competing_phone_inventory: CompetingPhoneInventory,
    *,
    manifest_sha256: str,
    calibration_report_sha256: str,
    competing_phone_inventory_sha256: str,
    dataset_cache: Path = DEFAULT_DATASET_CACHE,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate one frozen word classifier without threshold selection."""
    _validate_heldout_manifest(manifest)
    if manifest.partition != "heldout-validation":
        raise ValueError("The calibrated runner refuses the locked final-test partition")
    if (
        manifest.dataset_id != SPEECHOCEAN_DATASET_ID
        or manifest.dataset_revision != SPEECHOCEAN_REVISION
    ):
        raise ValueError("Held-out manifest does not match the pinned dataset")
    if calibration_report["decision"] != "development_candidate":
        raise ValueError("Calibration report is not a frozen development candidate")
    source = cast(dict[str, Any], calibration_report["source"])
    artifacts = cast(dict[str, Any], source["artifacts"])
    model_id = str(artifacts["model_id"])
    model_revision = str(artifacts["model_revision"])
    if (model_id, model_revision) not in MODEL_SNAPSHOTS.values():
        raise ValueError("Calibration report does not use a pinned acoustic model")
    if artifacts["competing_phone_inventory_sha256"] != competing_phone_inventory_sha256:
        raise ValueError("Competing phone inventory does not match the calibration report")
    if artifacts["canonical_pronunciations"] != "espeak-lowercase-plus-speechocean-v1":
        raise ValueError("Calibration report does not use the frozen canonical variants")
    candidate_model = calibration_report["candidate_model"]
    if not isinstance(candidate_model, dict):
        raise ValueError("Calibration report has no candidate model")

    completed, failures, mapped_count, source_phone_count = _collect_posterior_items(
        tuple(item.utterance_id for item in manifest.items),
        split=manifest.split,
        dataset_cache=dataset_cache,
        model_cache=model_cache,
        device=device,
        competing_phones=frozenset(competing_phone_inventory.phones),
        include_speechocean_variant=True,
        model_id=model_id,
        model_revision=model_revision,
    )
    evaluation = evaluate_word_model(completed, cast(dict[str, Any], candidate_model))
    word_metrics = cast(dict[str, Any], evaluation["metrics"])
    completion_rate = len(completed) / len(manifest.items)
    latencies = [int(item["inference_ms"]) for item in completed]
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else float("inf")
    peak_rss_mb = _peak_rss_mb()
    mapping_coverage = mapped_count / source_phone_count if source_phone_count else 0.0
    checks = {
        "completion_rate": completion_rate >= 0.95,
        "word_precision": float(word_metrics["word_precision"]) >= 0.4,
        "word_recall": float(word_metrics["word_recall"]) >= 0.2,
        "high_rating_false_review_rate": float(word_metrics["high_rating_false_review_rate"])
        <= 0.5,
        "latency_p95_ms": latency_p95 <= 5_000,
        "peak_rss_mb": peak_rss_mb <= 6_144,
        "mapping_coverage": mapping_coverage >= 0.99,
    }
    return {
        "report_version": 1,
        "experiment": "calibrated-word-error-heldout-v1",
        "decision": "proceed_stage2" if all(checks.values()) else "stop",
        "threshold_selection_performed": False,
        "final_test_remains_locked": True,
        "checks": checks,
        "metrics": {
            "completion_rate": completion_rate,
            "completed": len(completed),
            "failed": len(failures),
            "word_classification": word_metrics,
            "calibration_bins": evaluation["calibration_bins"],
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": latency_p95 if latencies else None,
            "peak_rss_mb": peak_rss_mb,
            "phone_mapping_coverage": mapping_coverage,
            "model_cache_bytes": _directory_size_bytes(model_cache),
        },
        "artifacts": {
            "model_id": model_id,
            "model_revision": model_revision,
            "manifest_sha256": manifest_sha256,
            "calibration_report_sha256": calibration_report_sha256,
            "competing_phone_inventory_sha256": competing_phone_inventory_sha256,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": device,
        },
        "failures": failures,
        "words": evaluation["words"],
    }


def _collect_posterior_items(
    utterance_ids: tuple[str, ...],
    *,
    split: str,
    dataset_cache: Path,
    model_cache: Path,
    device: str,
    competing_phones: frozenset[str] | None = None,
    include_speechocean_variant: bool = False,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int, int]:
    dataset = load_speechocean(split, dataset_cache)
    wanted = set(utterance_ids)
    rows = {
        _utterance_id(cast(dict[str, Any], cast(dict[str, Any], row)["audio"])): cast(
            dict[str, Any], row
        )
        for row in dataset
        if _utterance_id(cast(dict[str, Any], cast(dict[str, Any], row)["audio"])) in wanted
    }
    if set(rows) != wanted:
        raise RuntimeError("Dataset does not contain every frozen manifest item")

    recognizer = Wav2Vec2EspeakRecognizer(
        cache_dir=model_cache,
        device=device,
        model_id=model_id,
        model_revision=model_revision,
    )
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    mapped_count = 0
    source_phone_count = 0
    for utterance_id in utterance_ids:
        row = rows[utterance_id]
        try:
            prompt, mapped, total = _prompt_from_row(utterance_id, row)
            mapped_count += mapped
            source_phone_count += total
            expected_phones, word_indexes = _flatten_single_variant(prompt)
            expected_variants = [expected_phones]
            word_index_variants = [word_indexes]
            variant_names = ["espeak"]
            if include_speechocean_variant:
                mapped_phones, mapped_word_indexes = _flatten_speechocean_variant(row)
                if mapped_phones != expected_phones:
                    expected_variants.append(mapped_phones)
                    word_index_variants.append(mapped_word_indexes)
                    variant_names.append("speechocean")
            audio = _audio_from_row(cast(dict[str, Any], row["audio"]))
            recognitions = recognizer.score_expected_variants(
                audio,
                tuple(expected_variants),
                competing_phones=competing_phones,
            )
            selected_variant = max(
                range(len(recognitions)),
                key=lambda index: recognitions[index].alignment.path_mean_log_probability,
            )
            recognition = recognitions[selected_variant]
            expected_phones = expected_variants[selected_variant]
            word_indexes = word_index_variants[selected_variant]
            expert_problem_words = _expert_problem_word_indexes(row)
            phone_scores = [
                {
                    "target_index": score.target_index,
                    "word_index": word_indexes[score.target_index],
                    "phone": score.phone,
                    "start_frame": score.start_frame,
                    "end_frame": score.end_frame,
                    "frame_count": score.frame_count,
                    "expected_mean_probability": score.expected_mean_probability,
                    "expected_mean_log_probability": score.expected_mean_log_probability,
                    "alternative_phone": score.alternative_phone,
                    "alternative_mean_log_probability": score.alternative_mean_log_probability,
                    "log_posterior_margin": score.log_posterior_margin,
                }
                for score in recognition.alignment.phone_scores
            ]
            completed.append(
                {
                    "utterance_id": utterance_id,
                    "speaker": str(row["speaker"]),
                    "accuracy": int(row["accuracy"]),
                    "mean_phone_margin": recognition.alignment.mean_phone_margin,
                    "p10_phone_margin": recognition.alignment.p10_phone_margin,
                    "path_mean_log_probability": recognition.alignment.path_mean_log_probability,
                    "selected_canonical_variant": variant_names[selected_variant],
                    "canonical_variant_path_mean_log_probabilities": {
                        name: candidate.alignment.path_mean_log_probability
                        for name, candidate in zip(variant_names, recognitions, strict=True)
                    },
                    "inference_ms": recognition.inference_ms,
                    "expert_problem_word_indexes": sorted(expert_problem_words),
                    "expected_phones": expected_phones,
                    "greedy_phones": recognition.greedy_phones,
                    "phone_scores": phone_scores,
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures.append({"utterance_id": utterance_id, "error": str(error)})
    return completed, failures, mapped_count, source_phone_count


def load_manifest(path: Path) -> BenchmarkManifest:
    return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_competing_phone_inventory(path: Path) -> CompetingPhoneInventory:
    inventory = CompetingPhoneInventory.model_validate_json(path.read_text(encoding="utf-8"))
    if (
        inventory.dataset_id != SPEECHOCEAN_DATASET_ID
        or inventory.dataset_revision != SPEECHOCEAN_REVISION
    ):
        raise ValueError("Competing phone inventory does not match the pinned dataset")
    if len(set(inventory.phones)) != len(inventory.phones):
        raise ValueError("Competing phone inventory contains duplicate phones")
    if any(not phone.strip() for phone in inventory.phones):
        raise ValueError("Competing phone inventory contains a blank phone")
    return inventory


def load_policy(path: Path) -> DecisionPolicy:
    return DecisionPolicy.model_validate_json(path.read_text(encoding="utf-8"))


def load_heldout_manifest(path: Path) -> HeldoutManifest:
    manifest = HeldoutManifest.model_validate_json(path.read_text(encoding="utf-8"))
    _validate_heldout_manifest(manifest)
    return manifest


def load_posterior_evaluation_policy(path: Path) -> PosteriorEvaluationPolicy:
    return PosteriorEvaluationPolicy.model_validate_json(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_heldout_pair(validation: HeldoutManifest, final: HeldoutManifest) -> None:
    _validate_heldout_manifest(validation)
    _validate_heldout_manifest(final)
    if validation.partition != "heldout-validation" or final.partition != "final-test":
        raise ValueError("Held-out manifests have incorrect partition labels")
    validation_speakers = {item.speaker for item in validation.items}
    final_speakers = {item.speaker for item in final.items}
    if validation_speakers & final_speakers:
        raise ValueError("Held-out validation and final test speakers overlap")
    validation_items = {item.utterance_id for item in validation.items}
    final_items = {item.utterance_id for item in final.items}
    if validation_items & final_items:
        raise ValueError("Held-out validation and final test utterances overlap")


def _validate_heldout_manifest(manifest: HeldoutManifest) -> None:
    speakers = {item.speaker for item in manifest.items}
    utterance_ids = {item.utterance_id for item in manifest.items}
    if len(speakers) != manifest.speaker_count:
        raise ValueError("Held-out manifest speaker_count does not match its items")
    if len(utterance_ids) != len(manifest.items):
        raise ValueError("Held-out manifest contains duplicate utterance IDs")
    metadata: dict[str, tuple[str, str]] = {}
    for item in manifest.items:
        value = (item.age_group, item.gender)
        previous = metadata.setdefault(item.speaker, value)
        if previous != value:
            raise ValueError(f"Held-out manifest has inconsistent metadata for {item.speaker}")


def write_json(path: Path, value: ContractModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, ContractModel):
        content = value.model_dump_json(indent=2)
    else:
        content = json.dumps(value, ensure_ascii=False, indent=2)
    path.write_text(f"{content}\n", encoding="utf-8")


def _prompt_from_row(utterance_id: str, row: dict[str, Any]) -> tuple[Prompt, int, int]:
    raw_words = cast(list[dict[str, Any]], row["words"])
    canonical_words = phonemize_words(tuple(str(word["text"]).lower() for word in raw_words))
    prompt_words: list[PromptWord] = []
    mapped_count = 0
    total_count = 0
    for raw_word, canonical in zip(raw_words, canonical_words, strict=True):
        source = tuple(str(phone) for phone in cast(list[object], raw_word["phones"]))
        mapped, unmapped = map_arpabet_sequence(source)
        total_count += len(source)
        mapped_count += len(mapped)
        if unmapped:
            raise ValueError(f"Unmapped ARPAbet phones: {unmapped}")
        prompt_words.append(PromptWord(text=str(raw_word["text"]), phone_variants=(canonical,)))
    prompt = Prompt(
        id=f"speechocean-{utterance_id.lower()}",
        text=str(row["text"]),
        locale="en-us",
        words=tuple(prompt_words),
        catalogue_version=1,
    )
    return prompt, mapped_count, total_count


def _audio_from_row(audio: dict[str, Any]):  # type: ignore[no-untyped-def]
    audio_bytes = audio.get("bytes")
    if isinstance(audio_bytes, bytes):
        return load_audio(audio_bytes)
    path = audio.get("path")
    if isinstance(path, str):
        return load_audio(path)
    raise ValueError("Dataset audio has neither bytes nor path")


def _expert_problem_word_indexes(row: dict[str, Any]) -> set[int]:
    problems: set[int] = set()
    for word_index, raw_word in enumerate(cast(list[dict[str, Any]], row["words"])):
        accuracies = cast(list[float], raw_word["phones-accuracy"])
        if any(score < 1.0 for score in accuracies):
            problems.add(word_index)
    return problems


def _flatten_single_variant(prompt: Prompt) -> tuple[tuple[str, ...], tuple[int, ...]]:
    phones: list[str] = []
    word_indexes: list[int] = []
    for word_index, word in enumerate(prompt.words):
        variant = word.phone_variants[0]
        phones.extend(variant)
        word_indexes.extend([word_index] * len(variant))
    return tuple(phones), tuple(word_indexes)


def _flatten_speechocean_variant(row: dict[str, Any]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    phones: list[str] = []
    word_indexes: list[int] = []
    for word_index, raw_word in enumerate(cast(list[dict[str, Any]], row["words"])):
        source = tuple(str(phone) for phone in cast(list[object], raw_word["phones"]))
        mapped, unmapped = map_arpabet_sequence(source)
        if unmapped:
            raise ValueError(f"Unmapped ARPAbet phones: {unmapped}")
        normalized = _normalize_speechocean_variant(mapped)
        phones.extend(normalized)
        word_indexes.extend([word_index] * len(normalized))
    return tuple(phones), tuple(word_indexes)


def _normalize_speechocean_variant(phones: tuple[str, ...]) -> tuple[str, ...]:
    # shortcut: only bridge token shapes present in the pinned English corpus/model pair.
    replacements = {"ɝː": "ɜː"}
    compounds = {
        ("ɑː", "ɹ"): "ɑːɹ",  # noqa: RUF001
        ("ɔː", "ɹ"): "ɔːɹ",
        ("ɛ", "ɹ"): "ɛɹ",
        ("ɪ", "ɹ"): "ɪɹ",
        ("ʊ", "ɹ"): "ʊɹ",
        ("ə", "l"): "əl",
    }
    normalized = tuple(replacements.get(phone, phone) for phone in phones)
    result: list[str] = []
    index = 0
    while index < len(normalized):
        compound = (
            compounds.get((normalized[index], normalized[index + 1]))
            if index + 1 < len(normalized)
            else None
        )
        if compound is not None:
            result.append(compound)
            index += 2
            continue
        result.append(normalized[index])
        index += 1
    return tuple(result)


def _flagged_word_indexes(item: dict[str, Any], threshold: float) -> set[int]:
    phone_scores = cast(list[dict[str, Any]], item["phone_scores"])
    return {
        int(score["word_index"])
        for score in phone_scores
        if float(score["log_posterior_margin"]) < threshold
    }


def build_posterior_threshold_curve(
    completed: list[dict[str, Any]], policy: DecisionPolicy
) -> list[dict[str, Any]]:
    return [
        posterior_metrics_at_threshold(
            completed,
            threshold=threshold,
            high_rating_accuracy_min=policy.high_rating_accuracy_min,
        )
        for threshold in POSTERIOR_MARGIN_THRESHOLDS
    ]


def posterior_metrics_at_threshold(
    completed: list[dict[str, Any]], *, threshold: float, high_rating_accuracy_min: int
) -> dict[str, Any]:
    high_rated = [item for item in completed if int(item["accuracy"]) >= high_rating_accuracy_min]
    total_expert_problems = sum(
        len(cast(list[int], item["expert_problem_word_indexes"])) for item in completed
    )
    reviewed = 0
    high_rated_reviews = 0
    flagged_words = 0
    detected_expert_problems = 0
    for item in completed:
        flagged = _flagged_word_indexes(item, threshold)
        expert = set(cast(list[int], item["expert_problem_word_indexes"]))
        reviewed += bool(flagged)
        flagged_words += len(flagged)
        detected_expert_problems += len(flagged & expert)
        if int(item["accuracy"]) >= high_rating_accuracy_min:
            high_rated_reviews += bool(flagged)
    precision = detected_expert_problems / flagged_words if flagged_words else 0.0
    recall = detected_expert_problems / total_expert_problems if total_expert_problems else 0.0
    return {
        "threshold": threshold,
        "reviewed_recordings": reviewed,
        "high_rating_count": len(high_rated),
        "high_rating_reviews": high_rated_reviews,
        "high_rating_false_review_rate": (
            high_rated_reviews / len(high_rated) if high_rated else 1.0
        ),
        "flagged_words": flagged_words,
        "total_expert_problem_words": total_expert_problems,
        "detected_expert_problem_words": detected_expert_problems,
        "word_precision": precision,
        "word_recall": recall,
    }


def select_development_threshold(
    curve: list[dict[str, Any]], policy: DecisionPolicy
) -> dict[str, Any] | None:
    eligible = [
        point
        for point in curve
        if float(point["high_rating_false_review_rate"]) <= policy.high_rating_false_review_rate_max
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (
            int(point["detected_expert_problem_words"]),
            float(point["word_precision"]),
            -float(point["high_rating_false_review_rate"]),
            -float(point["threshold"]),
        ),
    )


def _build_posterior_report(
    *,
    manifest: BenchmarkManifest,
    policy: DecisionPolicy,
    completed: list[dict[str, Any]],
    failures: list[dict[str, str]],
    mapped_count: int,
    source_phone_count: int,
    model_cache: Path,
    device: str,
    competing_phone_inventory: CompetingPhoneInventory | None,
    competing_phone_inventory_sha256: str | None,
    include_speechocean_variant: bool,
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    completion_rate = len(completed) / len(manifest.items)
    accuracies = [float(item["accuracy"]) for item in completed]
    mean_margins = [float(item["mean_phone_margin"]) for item in completed]
    p10_margins = [float(item["p10_phone_margin"]) for item in completed]
    mean_margin_spearman = _spearman(accuracies, mean_margins)
    curve = build_posterior_threshold_curve(completed, policy)
    selected = select_development_threshold(curve, policy)
    latencies = [int(item["inference_ms"]) for item in completed]
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else float("inf")
    peak_rss_mb = _peak_rss_mb()
    mapping_coverage = mapped_count / source_phone_count if source_phone_count else 0.0
    selected_false_review = (
        float(selected["high_rating_false_review_rate"]) if selected is not None else 1.0
    )
    selected_detected = (
        int(selected["detected_expert_problem_words"]) if selected is not None else 0
    )
    checks = {
        "completion_rate": completion_rate >= policy.completion_rate_min,
        "positive_spearman": mean_margin_spearman > policy.spearman_point_min_exclusive,
        "high_rating_false_review_rate": selected_false_review
        <= policy.high_rating_false_review_rate_max,
        "interpretable_problems": selected_detected >= policy.interpretable_problem_min,
        "latency_p95_ms": latency_p95 <= policy.latency_p95_ms_max,
        "peak_rss_mb": peak_rss_mb <= policy.peak_rss_mb_max,
        "mapping_coverage": mapping_coverage >= policy.mapping_coverage_min,
    }
    selected_threshold = float(selected["threshold"]) if selected is not None else None
    high_rated_reviews = [
        item
        for item in completed
        if selected_threshold is not None
        and int(item["accuracy"]) >= policy.high_rating_accuracy_min
        and _flagged_word_indexes(item, selected_threshold)
    ]
    return {
        "report_version": 1,
        "experiment": _posterior_experiment_name(
            competing_phone_inventory is not None, include_speechocean_variant
        ),
        "decision": "development_candidate" if all(checks.values()) else "stop",
        "validation_required": True,
        "checks": checks,
        "metrics": {
            "completion_rate": completion_rate,
            "completed": len(completed),
            "failed": len(failures),
            "mean_margin_spearman": mean_margin_spearman,
            "mean_margin_spearman_bootstrap_95_percent": _bootstrap_spearman(
                accuracies,
                mean_margins,
                samples=policy.bootstrap_samples,
                seed=policy.bootstrap_seed,
            ),
            "p10_margin_spearman": _spearman(accuracies, p10_margins),
            "selected_development_threshold": selected_threshold,
            "selected_threshold_metrics": selected,
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": latency_p95 if latencies else None,
            "peak_rss_mb": peak_rss_mb,
            "phone_mapping_coverage": mapping_coverage,
            "model_cache_bytes": _directory_size_bytes(model_cache),
        },
        "threshold_curve": curve,
        "artifacts": {
            "model_id": model_id,
            "model_revision": model_revision,
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "manifest_version": manifest.version,
            "policy_version": policy.version,
            "canonical_pronunciations": (
                "espeak-lowercase-plus-speechocean-v1"
                if include_speechocean_variant
                else "espeak-lowercase-words-v1"
            ),
            "competing_phone_inventory": (
                competing_phone_inventory.model_dump(mode="json")
                if competing_phone_inventory is not None
                else None
            ),
            "competing_phone_inventory_sha256": competing_phone_inventory_sha256,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": device,
        },
        "failures": failures,
        "qualitative_failures": {"high_rated_reviews": high_rated_reviews[:5]},
        "items": completed,
    }


def _posterior_experiment_name(
    has_competing_phone_inventory: bool, include_speechocean_variant: bool
) -> str:
    if has_competing_phone_inventory and include_speechocean_variant:
        return "constrained-ctc-posterior-english-competitors-canonical-variants-v1"
    if has_competing_phone_inventory:
        return "constrained-ctc-posterior-english-competitors-v1"
    if include_speechocean_variant:
        return "constrained-ctc-posterior-canonical-variants-v1"
    return "constrained-ctc-posterior-v1"


def _build_posterior_heldout_report(
    *,
    manifest: HeldoutManifest,
    policy: PosteriorEvaluationPolicy,
    manifest_sha256: str,
    policy_sha256: str,
    completed: list[dict[str, Any]],
    failures: list[dict[str, str]],
    mapped_count: int,
    source_phone_count: int,
    model_cache: Path,
    device: str,
    competing_phone_inventory: CompetingPhoneInventory | None,
) -> dict[str, Any]:
    completion_rate = len(completed) / len(manifest.items)
    accuracies = [float(item["accuracy"]) for item in completed]
    mean_margins = [float(item["mean_phone_margin"]) for item in completed]
    p10_margins = [float(item["p10_phone_margin"]) for item in completed]
    mean_spearman = _spearman(accuracies, mean_margins)
    p10_spearman = _spearman(accuracies, p10_margins)
    mean_interval = _cluster_bootstrap_spearman(
        completed,
        score_key="mean_phone_margin",
        samples=policy.bootstrap_samples,
        seed=policy.bootstrap_seed,
    )
    p10_interval = _cluster_bootstrap_spearman(
        completed,
        score_key="p10_phone_margin",
        samples=policy.bootstrap_samples,
        seed=policy.bootstrap_seed,
    )
    threshold_metrics = posterior_metrics_at_threshold(
        completed,
        threshold=policy.phone_margin_threshold,
        high_rating_accuracy_min=policy.high_rating_accuracy_min,
    )
    latencies = [int(item["inference_ms"]) for item in completed]
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else float("inf")
    peak_rss_mb = _peak_rss_mb()
    mapping_coverage = mapped_count / source_phone_count if source_phone_count else 0.0
    checks = {
        "completion_rate": completion_rate >= policy.completion_rate_min,
        "high_rating_false_review_rate": float(threshold_metrics["high_rating_false_review_rate"])
        <= policy.high_rating_false_review_rate_max,
        "word_precision": float(threshold_metrics["word_precision"]) >= policy.word_precision_min,
        "word_recall": float(threshold_metrics["word_recall"]) >= policy.word_recall_min,
        "mean_margin_spearman": mean_spearman > policy.mean_margin_spearman_min_exclusive,
        "p10_margin_spearman": p10_spearman > policy.p10_margin_spearman_min_exclusive,
        "mean_spearman_cluster_bootstrap_lower": mean_interval[0]
        > policy.spearman_bootstrap_lower_min_exclusive,
        "p10_spearman_cluster_bootstrap_lower": p10_interval[0]
        > policy.spearman_bootstrap_lower_min_exclusive,
        "latency_p95_ms": latency_p95 <= policy.latency_p95_ms_max,
        "peak_rss_mb": peak_rss_mb <= policy.peak_rss_mb_max,
        "mapping_coverage": mapping_coverage >= policy.mapping_coverage_min,
    }
    high_rated_reviews = [
        _compact_review_item(item, policy.phone_margin_threshold)
        for item in completed
        if int(item["accuracy"]) >= policy.high_rating_accuracy_min
        and _flagged_word_indexes(item, policy.phone_margin_threshold)
    ]
    return {
        "report_version": 1,
        "experiment": _posterior_experiment_name(
            competing_phone_inventory is not None, policy.include_speechocean_variant
        ),
        "decision": "proceed_stage2" if all(checks.values()) else "stop",
        "threshold_selection_performed": False,
        "final_test_remains_locked": True,
        "checks": checks,
        "metrics": {
            "completion_rate": completion_rate,
            "completed": len(completed),
            "failed": len(failures),
            "mean_margin_spearman": mean_spearman,
            "mean_margin_spearman_cluster_bootstrap_95_percent": mean_interval,
            "p10_margin_spearman": p10_spearman,
            "p10_margin_spearman_cluster_bootstrap_95_percent": p10_interval,
            "fixed_threshold_metrics": threshold_metrics,
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": latency_p95 if latencies else None,
            "peak_rss_mb": peak_rss_mb,
            "phone_mapping_coverage": mapping_coverage,
            "model_cache_bytes": _directory_size_bytes(model_cache),
        },
        "frozen_policy": policy.model_dump(mode="json"),
        "artifacts": {
            "model_id": policy.model_id,
            "model_revision": policy.model_revision,
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "manifest_partition": manifest.partition,
            "manifest_version": manifest.version,
            "manifest_sha256": manifest_sha256,
            "policy_version": policy.version,
            "policy_sha256": policy_sha256,
            "competing_phone_inventory": (
                competing_phone_inventory.model_dump(mode="json")
                if competing_phone_inventory is not None
                else None
            ),
            "canonical_pronunciations": (
                "espeak-lowercase-plus-speechocean-v1"
                if policy.include_speechocean_variant
                else "espeak-lowercase-words-v1"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": device,
        },
        "failures": failures,
        "qualitative_failures": {"high_rated_reviews": high_rated_reviews[:20]},
        "items": [_compact_review_item(item, policy.phone_margin_threshold) for item in completed],
    }


def _compact_review_item(item: dict[str, Any], threshold: float) -> dict[str, Any]:
    phone_scores = cast(list[dict[str, Any]], item["phone_scores"])
    flagged_phones = [
        {
            "word_index": int(score["word_index"]),
            "phone": str(score["phone"]),
            "alternative_phone": str(score["alternative_phone"]),
            "log_posterior_margin": float(score["log_posterior_margin"]),
        }
        for score in phone_scores
        if float(score["log_posterior_margin"]) < threshold
    ]
    return {
        "utterance_id": str(item["utterance_id"]),
        "speaker": str(item["speaker"]),
        "accuracy": int(item["accuracy"]),
        "mean_phone_margin": float(item["mean_phone_margin"]),
        "p10_phone_margin": float(item["p10_phone_margin"]),
        "inference_ms": int(item["inference_ms"]),
        "expert_problem_word_indexes": cast(list[int], item["expert_problem_word_indexes"]),
        "flagged_phones": flagged_phones,
    }


def _build_report(
    *,
    manifest: BenchmarkManifest,
    policy: DecisionPolicy,
    completed: list[dict[str, Any]],
    failures: list[dict[str, str]],
    mapped_count: int,
    source_phone_count: int,
    device: str,
) -> dict[str, Any]:
    completion_rate = len(completed) / len(manifest.items)
    accuracies = [float(item["accuracy"]) for item in completed]
    similarities = [float(item["similarity"]) for item in completed]
    correlation = _spearman(accuracies, similarities)
    interval = _bootstrap_spearman(
        accuracies,
        similarities,
        samples=policy.bootstrap_samples,
        seed=policy.bootstrap_seed,
    )
    high_rated = [
        item for item in completed if int(item["accuracy"]) >= policy.high_rating_accuracy_min
    ]
    false_review_rate = (
        sum(bool(item["review"]) for item in high_rated) / len(high_rated) if high_rated else 1.0
    )
    interpretable = sum(int(item["detected_expert_problem_words"]) for item in completed)
    latencies = [int(item["inference_ms"]) for item in completed]
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else float("inf")
    peak_rss_mb = _peak_rss_mb()
    mapping_coverage = mapped_count / source_phone_count if source_phone_count else 0.0

    checks = {
        "completion_rate": completion_rate >= policy.completion_rate_min,
        "positive_spearman": correlation > policy.spearman_point_min_exclusive,
        "high_rating_false_review_rate": false_review_rate
        <= policy.high_rating_false_review_rate_max,
        "interpretable_problems": interpretable >= policy.interpretable_problem_min,
        "latency_p95_ms": latency_p95 <= policy.latency_p95_ms_max,
        "peak_rss_mb": peak_rss_mb <= policy.peak_rss_mb_max,
        "mapping_coverage": mapping_coverage >= policy.mapping_coverage_min,
    }
    return {
        "report_version": 1,
        "decision": "proceed" if all(checks.values()) else "stop",
        "checks": checks,
        "metrics": {
            "completion_rate": completion_rate,
            "completed": len(completed),
            "failed": len(failures),
            "spearman": correlation,
            "spearman_bootstrap_95_percent": interval,
            "high_rating_count": len(high_rated),
            "high_rating_false_review_rate": false_review_rate,
            "detected_expert_problem_words": interpretable,
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": latency_p95 if latencies else None,
            "peak_rss_mb": peak_rss_mb,
            "phone_mapping_coverage": mapping_coverage,
            "model_cache_bytes": _directory_size_bytes(DEFAULT_MODEL_CACHE),
        },
        "artifacts": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "manifest_version": manifest.version,
            "policy_version": policy.version,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": device,
        },
        "failures": failures,
        "qualitative_failures": {
            "high_rated_reviews": [item for item in high_rated if bool(item["review"])][:5],
            "missed_expert_problems": [
                item
                for item in completed
                if int(item["expert_problem_words"]) > 0
                and int(item["detected_expert_problem_words"]) == 0
            ][:5],
        },
        "items": completed,
    }


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return 0.0
    statistic = cast(float, spearmanr(left, right)[0])
    return float(statistic)


def _bootstrap_spearman(
    left: list[float], right: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not left:
        return (0.0, 0.0)
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indexes = generator.integers(0, len(left), size=len(left))
        sampled_left = [left[int(index)] for index in indexes]
        sampled_right = [right[int(index)] for index in indexes]
        values.append(_spearman(sampled_left, sampled_right))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def _cluster_bootstrap_spearman(
    completed: list[dict[str, Any]], *, score_key: str, samples: int, seed: int
) -> tuple[float, float]:
    if not completed:
        return (0.0, 0.0)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in completed:
        by_speaker[str(item["speaker"])].append(item)
    speakers = tuple(sorted(by_speaker))
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled_indexes = generator.integers(0, len(speakers), size=len(speakers))
        sampled_items = [
            item for index in sampled_indexes for item in by_speaker[speakers[int(index)]]
        ]
        accuracies = [float(item["accuracy"]) for item in sampled_items]
        scores = [float(item[score_key]) for item in sampled_items]
        values.append(_spearman(accuracies, scores))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def _utterance_id(audio: dict[str, Any]) -> str:
    path = audio.get("path")
    if not isinstance(path, str):
        raise ValueError("Dataset audio path is missing")
    return Path(path).stem


def _accuracy_band(accuracy: int) -> str:
    if accuracy <= 5:
        return "low"
    if accuracy <= 8:
        return "mid"
    return "high"


def _selection_hash(utterance_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{utterance_id}".encode()).hexdigest()


def _directory_size_bytes(path: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        stat = item.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += stat.st_size
    return total


MANIFEST_ADAPTER = TypeAdapter(BenchmarkManifest)
POLICY_ADAPTER = TypeAdapter(DecisionPolicy)
HELDOUT_MANIFEST_ADAPTER = TypeAdapter(HeldoutManifest)
POSTERIOR_EVALUATION_POLICY_ADAPTER = TypeAdapter(PosteriorEvaluationPolicy)
