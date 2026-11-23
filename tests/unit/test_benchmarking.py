# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from pronunciation_coach.benchmarking import (
    DecisionPolicy,
    HeldoutManifest,
    HeldoutManifestItem,
    PosteriorEvaluationPolicy,
    _flatten_speechocean_variant,
    _prompt_from_row,
    build_posterior_threshold_curve,
    file_sha256,
    load_competing_phone_inventory,
    load_heldout_manifest,
    load_posterior_evaluation_policy,
    posterior_metrics_at_threshold,
    prepare_calibration_manifest,
    prepare_heldout_manifests,
    prepare_manifest,
    run_calibrated_word_heldout_benchmark,
    run_posterior_heldout_benchmark,
    select_development_threshold,
    validate_heldout_pair,
)
from pronunciation_coach.config import MODEL_ID, MODEL_REVISION, SPEECHOCEAN_DATASET_ID

ENGLISH_INVENTORY = Path("benchmarks/inventories/english-espeak-train-v1.json")
ENGLISH_POLICY = Path("benchmarks/policies/ctc-posterior-english-variants-v1.json")
XLSR_POLICY = Path("benchmarks/policies/ctc-posterior-xlsr53-english-variants-v1.json")
HELDOUT_MANIFEST = Path("benchmarks/manifests/heldout-validation-v1.json")


def test_manifest_selection_is_balanced_unique_and_deterministic() -> None:
    rows: list[dict[str, Any]] = []
    counter = 0
    for age_group, age in (("child", 12), ("adult", 24)):
        for accuracy_band, accuracy in (("low", 4), ("mid", 7), ("high", 9)):
            for _ in range(10):
                counter += 1
                rows.append(
                    {
                        "audio": {"path": f"{counter:09}.wav", "bytes": None},
                        "speaker": f"speaker-{counter}",
                        "age": age,
                        "accuracy": accuracy,
                        "stratum": f"{age_group}-{accuracy_band}",
                    }
                )
    dataset = Dataset.from_list(rows)

    first = prepare_manifest(dataset)
    second = prepare_manifest(dataset)

    assert first == second
    assert len(first.items) == 50
    assert len({item.speaker for item in first.items}) == 50
    counts = {
        (age_group, accuracy_band): sum(
            item.age_group == age_group and item.accuracy_band == accuracy_band
            for item in first.items
        )
        for age_group in ("child", "adult")
        for accuracy_band in ("low", "mid", "high")
    }
    assert sorted(counts.values()) == [8, 8, 8, 8, 9, 9]


def test_calibration_manifest_selects_one_recording_per_speaker_without_labels() -> None:
    rows: list[dict[str, Any]] = []
    for speaker in range(5):
        for recording in range(3):
            rows.append(
                {
                    "audio": {"path": f"{speaker:03}{recording:06}.wav", "bytes": None},
                    "speaker": f"speaker-{speaker}",
                    "age": 20 + speaker,
                    "accuracy": recording + 3,
                }
            )
    dataset = Dataset.from_list(rows)
    relabeled = Dataset.from_list([{**row, "accuracy": 10 - int(row["accuracy"])} for row in rows])

    first = prepare_calibration_manifest(dataset)
    second = prepare_calibration_manifest(dataset)
    after_relabeling = prepare_calibration_manifest(relabeled)

    assert first == second
    assert len(first.items) == 5
    assert len({item.speaker for item in first.items}) == 5
    assert [item.utterance_id for item in first.items] == [
        item.utterance_id for item in after_relabeling.items
    ]


def test_posterior_threshold_selection_respects_false_review_gate() -> None:
    policy = DecisionPolicy(
        version=1,
        frozen_before_results=True,
        completion_rate_min=0.95,
        spearman_point_min_exclusive=0.0,
        high_rating_accuracy_min=9,
        high_rating_false_review_rate_max=0.5,
        interpretable_problem_min=1,
        latency_p95_ms_max=5_000,
        peak_rss_mb_max=6_144,
        mapping_coverage_min=0.99,
        bootstrap_samples=10,
        bootstrap_seed=762,
    )
    completed = [
        {
            "accuracy": 9,
            "expert_problem_word_indexes": [],
            "phone_scores": [{"word_index": 0, "log_posterior_margin": -9.0}],
        },
        {
            "accuracy": 10,
            "expert_problem_word_indexes": [],
            "phone_scores": [{"word_index": 0, "log_posterior_margin": 2.0}],
        },
        {
            "accuracy": 4,
            "expert_problem_word_indexes": [0],
            "phone_scores": [{"word_index": 0, "log_posterior_margin": -9.0}],
        },
    ]

    curve = build_posterior_threshold_curve(completed, policy)
    selected = select_development_threshold(curve, policy)

    assert selected is not None
    assert selected["threshold"] == -8.0
    assert selected["detected_expert_problem_words"] == 1
    assert selected["high_rating_false_review_rate"] == 0.5


def test_english_competing_phone_inventory_is_pinned_and_unique(tmp_path: Path) -> None:
    inventory = load_competing_phone_inventory(ENGLISH_INVENTORY)

    assert len(inventory.phones) == 57
    assert {"aɪ", "ə", "θ", "ʒ"} <= set(inventory.phones)

    raw = json.loads(ENGLISH_INVENTORY.read_text(encoding="utf-8"))
    raw["phones"].append(raw["phones"][0])
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate phones"):
        load_competing_phone_inventory(duplicate_path)


def test_dataset_phonemization_lowercases_uppercase_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, ...]] = []

    def fake_phonemize(words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        received.append(words)
        return (("ɪ", "t"),)

    monkeypatch.setattr("pronunciation_coach.benchmarking.phonemize_words", fake_phonemize)
    prompt, mapped, total = _prompt_from_row(
        "one",
        {
            "text": "IT",
            "words": [{"text": "IT", "phones": ["IH1", "T"]}],
        },
    )

    assert received == [("it",)]
    assert prompt.words[0].phone_variants == (("ɪ", "t"),)
    assert (mapped, total) == (2, 2)


def test_speechocean_variant_uses_model_compatible_english_tokens() -> None:
    phones, word_indexes = _flatten_speechocean_variant(
        {
            "words": [
                {"text": "CAR", "phones": ["K", "AA1", "R"]},
                {"text": "A", "phones": ["AH0", "L"]},
                {"text": "HER", "phones": ["HH", "ER1"]},
            ]
        }
    )

    assert phones == ("k", "ɑːɹ", "əl", "h", "ɜː")
    assert word_indexes == (0, 0, 1, 2, 2)


def test_heldout_partition_is_deterministic_balanced_and_label_blind() -> None:
    rows: list[dict[str, Any]] = []
    speaker_number = 0
    for age, gender in ((10, "f"), (10, "m"), (30, "f"), (30, "m")):
        for _ in range(2):
            speaker_number += 1
            for utterance in range(2):
                rows.append(
                    {
                        "audio": {
                            "path": f"{speaker_number:03}-{utterance:02}.wav",
                            "bytes": None,
                        },
                        "speaker": f"speaker-{speaker_number}",
                        "age": age,
                        "gender": gender,
                    }
                )
    dataset = Dataset.from_list(rows)

    first_validation, first_final = prepare_heldout_manifests(dataset)
    second_validation, second_final = prepare_heldout_manifests(dataset)

    assert (first_validation, first_final) == (second_validation, second_final)
    validate_heldout_pair(first_validation, first_final)
    assert first_validation.speaker_count == 4
    assert first_final.speaker_count == 4
    assert len(first_validation.items) == 8
    assert len(first_final.items) == 8
    assert {item.age_group for item in first_validation.items} == {"child", "adult"}
    assert {item.gender for item in first_validation.items} == {"f", "m"}


def test_fixed_threshold_metrics_do_not_select_an_alternative() -> None:
    completed = [
        {
            "accuracy": 9,
            "expert_problem_word_indexes": [],
            "phone_scores": [{"word_index": 0, "log_posterior_margin": -9.0}],
        },
        {
            "accuracy": 4,
            "expert_problem_word_indexes": [0],
            "phone_scores": [{"word_index": 0, "log_posterior_margin": -7.0}],
        },
    ]

    metrics = posterior_metrics_at_threshold(completed, threshold=-8.0, high_rating_accuracy_min=9)

    assert metrics["threshold"] == -8.0
    assert metrics["high_rating_false_review_rate"] == 1.0
    assert metrics["detected_expert_problem_words"] == 0


def test_heldout_runner_refuses_final_test_partition_before_loading_model() -> None:
    manifest = HeldoutManifest(
        dataset_id=SPEECHOCEAN_DATASET_ID,
        dataset_revision="revision",
        split="test",
        partition="final-test",
        selection_seed=762,
        selection_algorithm="test",
        speaker_count=1,
        items=(
            HeldoutManifestItem(
                utterance_id="one",
                speaker="speaker-one",
                age_group="adult",
                gender="f",
            ),
        ),
    )
    policy = PosteriorEvaluationPolicy(
        version=1,
        frozen_before_heldout_results=True,
        scorer="constrained-ctc-viterbi-v1",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        phone_margin_threshold=-8.0,
        word_review_rule="any-phone-margin-below-threshold",
        recording_review_rule="any-word-reviewed",
        completion_rate_min=0.95,
        high_rating_accuracy_min=9,
        high_rating_false_review_rate_max=0.5,
        word_precision_min=0.4,
        word_recall_min=0.2,
        mean_margin_spearman_min_exclusive=0.0,
        p10_margin_spearman_min_exclusive=0.0,
        spearman_bootstrap_lower_min_exclusive=0.0,
        latency_p95_ms_max=5_000,
        peak_rss_mb_max=6_144,
        mapping_coverage_min=0.99,
        bootstrap_samples=10,
        bootstrap_seed=762,
    )

    with pytest.raises(ValueError, match="refuses the locked final-test"):
        run_posterior_heldout_benchmark(
            manifest,
            policy,
            manifest_sha256="manifest",
            policy_sha256="policy",
        )


def test_heldout_runner_requires_the_frozen_competing_inventory() -> None:
    manifest = load_heldout_manifest(HELDOUT_MANIFEST)
    for policy_path in (ENGLISH_POLICY, XLSR_POLICY):
        policy = load_posterior_evaluation_policy(policy_path)
        with pytest.raises(ValueError, match="inventory does not match"):
            run_posterior_heldout_benchmark(
                manifest,
                policy,
                manifest_sha256=file_sha256(HELDOUT_MANIFEST),
                policy_sha256=file_sha256(policy_path),
            )


def test_calibrated_runner_refuses_final_test_partition_before_loading_model() -> None:
    manifest = HeldoutManifest(
        dataset_id=SPEECHOCEAN_DATASET_ID,
        dataset_revision="revision",
        split="test",
        partition="final-test",
        selection_seed=762,
        selection_algorithm="test",
        speaker_count=1,
        items=(
            HeldoutManifestItem(
                utterance_id="one",
                speaker="speaker-one",
                age_group="adult",
                gender="f",
            ),
        ),
    )

    with pytest.raises(ValueError, match="refuses the locked final-test"):
        run_calibrated_word_heldout_benchmark(
            manifest,
            {},
            load_competing_phone_inventory(ENGLISH_INVENTORY),
            manifest_sha256="manifest",
            calibration_report_sha256="calibration",
            competing_phone_inventory_sha256=file_sha256(ENGLISH_INVENTORY),
        )
