import numpy as np
import pytest

from pronunciation_coach.ctc_scoring import constrained_ctc_score


def _logs(rows: list[list[float]]) -> np.ndarray:
    return np.log(np.asarray(rows, dtype=np.float64))


def test_constrained_alignment_preserves_close_alternative_evidence() -> None:
    vocabulary = {"<blank>": 0, "a": 1, "b": 2, "x": 3}
    probabilities = _logs(
        [
            [0.90, 0.04, 0.03, 0.03],
            [0.05, 0.40, 0.10, 0.45],
            [0.05, 0.42, 0.10, 0.43],
            [0.90, 0.04, 0.03, 0.03],
            [0.05, 0.05, 0.85, 0.05],
            [0.90, 0.04, 0.03, 0.03],
        ]
    )

    result = constrained_ctc_score(
        probabilities,
        ("a", "b"),
        vocabulary,
        blank_id=0,
        excluded_alternative_ids=frozenset({0}),
    )

    first, second = result.phone_scores
    assert (first.start_frame, first.end_frame) == (1, 3)
    assert first.alternative_phone == "x"
    assert first.log_posterior_margin < 0
    assert second.alternative_phone in {"a", "x"}
    assert second.log_posterior_margin > 2
    assert result.mean_phone_margin > first.log_posterior_margin


def test_repeated_phone_requires_an_intervening_blank() -> None:
    vocabulary = {"<blank>": 0, "a": 1, "x": 2}
    probabilities = _logs(
        [
            [0.05, 0.90, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
        ]
    )

    result = constrained_ctc_score(probabilities, ("a", "a"), vocabulary, blank_id=0)

    assert [(score.start_frame, score.end_frame) for score in result.phone_scores] == [
        (0, 1),
        (2, 3),
    ]


def test_alignment_rejects_too_few_frames() -> None:
    vocabulary = {"<blank>": 0, "a": 1, "x": 2}
    probabilities = _logs([[0.05, 0.90, 0.05], [0.05, 0.90, 0.05]])

    with pytest.raises(ValueError, match="too few CTC frames"):
        constrained_ctc_score(probabilities, ("a", "a"), vocabulary, blank_id=0)


def test_alignment_rejects_unknown_target_phone() -> None:
    vocabulary = {"<blank>": 0, "a": 1, "x": 2}
    probabilities = _logs([[0.05, 0.90, 0.05]])

    with pytest.raises(ValueError, match="absent from the model vocabulary"):
        constrained_ctc_score(probabilities, ("b",), vocabulary, blank_id=0)
