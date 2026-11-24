from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest

from pronunciation_coach.benchmarking import CompetingPhoneInventory
from pronunciation_coach.l2arctic import (
    SPEAKER_BY_ID,
    iter_l2arctic_utterances,
    parse_textgrid,
    prepare_l2arctic_manifest,
)
from pronunciation_coach.l2arctic_experiment import (
    evaluate_l2arctic_phone_model,
    evaluate_l2arctic_speechocean_regression,
    run_l2arctic_evidence,
    select_word_threshold,
)

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

item [1]:
    class = "IntervalTier"
    name = "words"
    intervals [1]:
        xmin = 0
        xmax = 0.5
        text = "three"
item [2]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.2
        text = "TH, F, S"
    intervals [2]:
        xmin = 0.2
        xmax = 0.3
        text = "R"
    intervals [3]:
        xmin = 0.3
        xmax = 0.5
        text = "IY1"
"""


def test_manifest_is_deterministic_balanced_and_disjoint() -> None:
    first = prepare_l2arctic_manifest()
    second = prepare_l2arctic_manifest()

    assert first == second
    assert len(first.train_speakers) == 12
    assert len(first.development_speakers) == 6
    assert len(first.final_test_speakers) == 6
    assert set(first.train_speakers).isdisjoint(first.development_speakers)
    assert set(first.train_speakers).isdisjoint(first.final_test_speakers)
    assert set(first.development_speakers).isdisjoint(first.final_test_speakers)
    all_speakers = first.train_speakers + first.development_speakers + first.final_test_speakers
    assert set(all_speakers) == set(SPEAKER_BY_ID)


def test_textgrid_parser_retains_only_requested_tiers() -> None:
    tiers = parse_textgrid(TEXTGRID)

    assert tuple(tiers) == ("words", "phones")
    assert tiers["words"][0].text == "three"
    assert tiers["phones"][0].text == "TH, F, S"


def test_zip_loader_parses_explicit_substitution(tmp_path: Path) -> None:
    archive_path = tmp_path / "ABA.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("ABA/annotation/arctic_a0001.TextGrid", TEXTGRID)
        archive.writestr("ABA/transcript/arctic_a0001.txt", "Three")
        archive.writestr("ABA/wav/arctic_a0001.wav", b"not-read-in-this-test")

    utterance = next(iter_l2arctic_utterances(tmp_path, ("ABA",), include_audio=False))

    assert utterance.words == ("three",)
    assert utterance.audio_bytes is None
    assert utterance.phones[0].expected_arpabet == "TH"
    assert utterance.phones[0].perceived_arpabet == "F"
    assert utterance.phones[0].expected_ipa == "θ"
    assert utterance.phones[0].perceived_ipa == "f"
    assert utterance.phones[0].error_type == "substitution"
    assert utterance.phones[0].word_index == 0


def test_evidence_runner_refuses_locked_final_before_loading_model(tmp_path: Path) -> None:
    manifest = prepare_l2arctic_manifest()
    inventory = CompetingPhoneInventory(
        version=1,
        dataset_id="mispeech/speechocean762",
        dataset_revision="06385584fad212b26134c656fdd3ccf9f093f33e",
        split="train",
        locale="en-us",
        source="unique-word-types-in-pinned-training-transcripts",
        phonemizer="3.6.0",
        espeak_ng="1.52.0",
        phones=("f", "θ"),
    )

    with pytest.raises(ValueError, match="refuses the locked final-test"):
        run_l2arctic_evidence(
            tmp_path,
            manifest,
            "final-test",
            inventory,
            manifest_sha256="manifest",
            inventory_sha256="inventory",
        )


def test_phone_candidate_localizes_and_diagnoses_substitution() -> None:
    report = {
        "items": [
            {
                "speaker": "ABA",
                "first_language": "Arabic",
                "utterance_id": "one",
                "phone_scores": [
                    {
                        "word_index": 0,
                        "expected": "θ",
                        "perceived": "f",
                        "alternative": "f",
                        "error_type": "substitution",
                        "expected_mean_log_probability": -4.0,
                        "alternative_mean_log_probability": -1.0,
                        "log_posterior_margin": -3.0,
                        "frame_count": 4,
                    },
                    {
                        "word_index": 0,
                        "expected": "iː",
                        "perceived": "iː",
                        "alternative": "ɪ",
                        "error_type": "correct",
                        "expected_mean_log_probability": -1.0,
                        "alternative_mean_log_probability": -4.0,
                        "log_posterior_margin": 3.0,
                        "frame_count": 4,
                    },
                ],
            }
        ]
    }
    model = {
        "version": 1,
        "method": "l2-logistic-regression",
        "continuous_feature_names": (
            "expected_mean_log_probability",
            "alternative_mean_log_probability",
            "log_posterior_margin",
            "log_frame_count",
        ),
        "continuous_means": [0.0, 0.0, 0.0, 0.0],
        "continuous_scales": [1.0, 1.0, 1.0, 1.0],
        "expected_phone_categories": ["<other>"],
        "alternative_phone_categories": ["<other>"],
        "intercept": 0.0,
        "coefficients": [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
        "red_probability_threshold": 0.5,
        "yellow_probability_threshold": 0.2,
    }

    evaluation = evaluate_l2arctic_phone_model(report, model)

    assert evaluation["phone_metrics"]["precision"] == 1.0
    assert evaluation["phone_metrics"]["recall"] == 1.0
    assert evaluation["word_metrics"]["precision"] == 1.0
    assert evaluation["diagnosis"]["exact_alternative_accuracy"] == 1.0


def test_red_threshold_requires_word_precision_margin_and_recall_gain() -> None:
    curve = [
        {"threshold": 0.4, "precision": 0.81, "recall": 0.2},
        {"threshold": 0.5, "precision": 0.85, "recall": 0.1},
        {"threshold": 0.6, "precision": 0.9, "recall": 0.04},
    ]

    selected = select_word_threshold(curve, minimum_precision=0.82, minimum_recall=0.05)

    assert selected is curve[1]


def test_speechocean_regression_rejects_invalid_warning_threshold() -> None:
    evidence = {"experiment": "constrained-ctc-posterior-english-competitors-canonical-variants-v1"}
    calibration: dict[str, Any] = {
        "decision": "development_candidate",
        "candidate_model": {},
    }

    with pytest.raises(ValueError, match="between zero and one"):
        evaluate_l2arctic_speechocean_regression(
            evidence,
            calibration,
            warning_probability_threshold=1.0,
            evidence_report_sha256="evidence",
            calibration_report_sha256="calibration",
        )
