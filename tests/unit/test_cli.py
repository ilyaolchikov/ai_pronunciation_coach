import json
from typing import Any

from pytest import CaptureFixture

from pronunciation_coach.cli import main


def parsed_output(captured: CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(captured.readouterr().out)


def test_cli_emits_result_json(capsys: CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "analyze",
            "--prompt",
            "thought-about-it",
            "--fake-observed",
            "aɪ s ɔː t ɐ b aʊ t ɪ t",
        ]
    )
    output = parsed_output(capsys)

    assert exit_code == 0
    assert output["schema_version"] == "1.2"
    assert output["words"][1]["status"] == "review"
    assert output["words"][1]["tips"] == ["theta-vs-s"]


def test_cli_reports_unknown_prompt_as_json(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["analyze", "--prompt", "missing", "--fake-observed", "a"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"code": "prompt_unknown", "message": "Unknown prompt: missing"}
    }


def test_cli_accepts_an_empty_observed_sequence(capsys: CaptureFixture[str]) -> None:
    exit_code = main(["analyze", "--prompt", "thought-about-it", "--fake-observed", ""])
    output = parsed_output(capsys)

    assert exit_code == 0
    assert {word["status"] for word in output["words"]} == {"not_detected"}
