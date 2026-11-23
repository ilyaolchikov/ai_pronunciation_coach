import pytest
from pydantic import ValidationError

from pronunciation_coach.analysis import analyze
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.recognizers.fake import FakePhoneRecognizer
from pronunciation_coach.schemas import AnalysisResult, Operation, PhoneEvidence


def test_result_json_round_trip_and_contract_snapshot() -> None:
    prompt = get_prompt("thought-about-it")
    result = analyze(
        prompt,
        FakePhoneRecognizer(("aɪ", "s", "ɔː", "t", "ɐ", "b", "aʊ", "t", "ɪ", "t")),
    )

    restored = AnalysisResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert result.model_dump(mode="json").keys() == {
        "schema_version",
        "prompt_id",
        "prompt_catalogue_version",
        "expected_text",
        "selected_variant_indices",
        "alignment_cost",
        "expected_phones",
        "observed_phones",
        "audio",
        "model",
        "policy",
        "words",
        "limitations",
        "warnings",
    }
    assert result.model_dump(mode="json")["model"] == {
        "backend": "fake",
        "revision": "stage0",
        "inference_ms": 0,
        "device": None,
    }


def test_phone_evidence_rejects_invalid_operation_shape() -> None:
    with pytest.raises(ValidationError, match="substitution requires two different phones"):
        PhoneEvidence(expected="s", observed="s", operation=Operation.SUBSTITUTION)


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PhoneEvidence(
            expected="a",
            observed="a",
            operation=Operation.MATCH,
            unknown=True,  # type: ignore[call-arg]
        )
