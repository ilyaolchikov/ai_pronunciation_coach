import pytest
from pydantic import ValidationError

from pronunciation_coach.schemas import PhoneResult, WordResult, WordStatus


def test_pronunciada_word_contract_round_trips() -> None:
    word = WordResult(
        text="thought",
        status=WordStatus.REVIEW,
        acceptance_probability=0.3,
        phones=(PhoneResult(expected="θ", heard="f", acceptance_probability=0.2),),
    )

    restored = WordResult.model_validate_json(word.model_dump_json())

    assert restored == word
    assert word.model_dump(mode="json") == {
        "text": "thought",
        "status": "review",
        "acceptance_probability": 0.3,
        "phones": [{"expected": "θ", "heard": "f", "acceptance_probability": 0.2}],
    }


def test_contracts_reject_legacy_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WordResult(
            text="word",
            status=WordStatus.REVIEW,
            acceptance_probability=0.3,
            phones=(),
            review_probability=0.7,  # type: ignore[call-arg]
        )
