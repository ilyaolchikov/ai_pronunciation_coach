import pytest

gradio = pytest.importorskip("gradio")

from pronunciation_coach.prompts import load_prompts  # noqa: E402
from pronunciation_coach.ui import build_app, navigate_prompt  # noqa: E402


def test_builds_local_ui_without_loading_the_model() -> None:
    app = build_app()

    assert isinstance(app, gradio.Blocks)


def test_navigation_wraps_and_clears_the_previous_recording_and_results() -> None:
    prompts = load_prompts()

    forward = navigate_prompt(0, 1, prompts)
    backward = navigate_prompt(0, -1, prompts)

    assert forward[0] == 1
    assert prompts[1].text in forward[1]
    assert forward[2] is None
    assert forward[3] == []
    assert forward[6] is None
    assert backward[0] == len(prompts) - 1
    assert prompts[-1].text in backward[1]
