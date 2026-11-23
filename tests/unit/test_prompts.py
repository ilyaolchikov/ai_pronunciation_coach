import pytest

from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.prompts import get_prompt, load_prompts


def test_loads_three_versioned_prompts() -> None:
    prompts = load_prompts()

    assert len(prompts) == 3
    assert {prompt.catalogue_version for prompt in prompts} == {1}
    assert {prompt.id for prompt in prompts} == {
        "thought-about-it",
        "she-sells-seashells",
        "three-free-throws",
    }


def test_unknown_prompt_has_stable_error() -> None:
    with pytest.raises(CoachError) as captured:
        get_prompt("missing")

    assert captured.value.code is ErrorCode.PROMPT_UNKNOWN
    assert str(captured.value) == "Unknown prompt: missing"
