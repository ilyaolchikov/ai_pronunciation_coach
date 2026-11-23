from typing import Any

import pytest

from pronunciation_coach.word_calibration import calibrate_word_errors, evaluate_word_model


def _posterior_report() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for index in range(10):
        severe = index < 5
        margin = -8.0 if severe else 2.0
        items.append(
            {
                "utterance_id": f"utterance-{index}",
                "speaker": f"speaker-{index}",
                "accuracy": 4 if severe else 9,
                "selected_canonical_variant": "espeak",
                "expert_problem_word_indexes": [0] if severe else [],
                "phone_scores": [
                    {
                        "word_index": 0,
                        "phone": "a",
                        "frame_count": 4,
                        "expected_mean_log_probability": margin - 1.0,
                        "alternative_mean_log_probability": -1.0,
                        "log_posterior_margin": margin,
                    }
                ],
            }
        )
    return {
        "experiment": "test-posterior",
        "artifacts": {
            "competing_phone_inventory": {"phones": ["a", "b"]},
        },
        "items": items,
    }


def test_word_calibration_is_speaker_disjoint_deterministic_and_predictive() -> None:
    first = calibrate_word_errors(_posterior_report(), source_report_sha256="source")
    second = calibrate_word_errors(_posterior_report(), source_report_sha256="source")

    assert first == second
    assert first["decision"] == "development_candidate"
    assert first["metrics"]["speakers"] == 10
    assert first["metrics"]["words"] == 10
    assert first["metrics"]["out_of_fold_brier_score"] < 0.25
    selected = first["metrics"]["selected_threshold_metrics"]
    assert selected["word_precision"] == 1.0
    assert selected["word_recall"] == 1.0
    assert all(
        fold["training_speakers"] + fold["validation_speakers"] == 10 for fold in first["folds"]
    )

    evaluation = evaluate_word_model(_posterior_report()["items"], first["candidate_model"])
    assert evaluation["metrics"]["word_precision"] == 1.0
    assert evaluation["metrics"]["word_recall"] == 1.0

    invalid_model = {**first["candidate_model"], "continuous_feature_names": ["different"]}
    with pytest.raises(ValueError, match="different continuous features"):
        evaluate_word_model(_posterior_report()["items"], invalid_model)
