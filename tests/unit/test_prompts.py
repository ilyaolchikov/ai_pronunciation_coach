import pytest

from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.prompts import get_prompt, load_prompts


def test_loads_twenty_five_versioned_prompts() -> None:
    prompts = load_prompts()

    assert len(prompts) == 25
    assert {prompt.catalogue_version for prompt in prompts} == {2}
    assert {prompt.id for prompt in prompts} == {
        "thought-about-it",
        "she-sells-seashells",
        "three-free-throws",
        "think-about-it",
        "turn-right-here",
        "the-weather-is-warm-today",
        "please-open-the-blue-door",
        "fresh-fish-swim-fast",
        "the-red-light-is-bright",
        "she-chose-cheap-chairs",
        "very-brave-people-vote",
        "peter-packed-paper-bags",
        "good-kids-can-cook",
        "jane-enjoys-orange-juice",
        "those-three-brothers-think",
        "the-quick-brown-fox",
        "quiet-walk-in-the-park",
        "new-yellow-jacket",
        "birds-near-the-river",
        "bring-three-clean-glasses",
        "measure-the-blue-boxes",
        "child-found-five-coins",
        "open-the-window-gently",
        "train-came-early",
        "short-letter-yesterday",
    }


def test_unknown_prompt_has_stable_error() -> None:
    with pytest.raises(CoachError) as captured:
        get_prompt("missing")

    assert captured.value.code is ErrorCode.PROMPT_UNKNOWN
    assert str(captured.value) == "Unknown prompt: missing"
