from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest import MonkeyPatch

from pronunciation_coach.audio import NormalizedAudio
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.schemas import (
    AnalysisResult,
    ModelMetadata,
    Prompt,
    ScoringSettings,
    WordResult,
    WordStatus,
)
from pronunciation_coach.ui_adapter import (
    NO_PROBLEM_LABEL,
    REVIEW_LABEL,
    WARNING_LABEL,
    Analyzer,
    analyze_for_ui,
    prompt_text,
)


class FakeAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, prompt: Prompt, audio: NormalizedAudio) -> AnalysisResult:
        self.calls += 1
        assert audio.metadata.speech_region is not None
        probabilities = (0.2, 0.7, 0.9, 0.9)
        statuses = (
            WordStatus.REVIEW,
            WordStatus.WARNING,
            WordStatus.NO_PROBLEM_DETECTED,
            WordStatus.NO_PROBLEM_DETECTED,
        )
        words = tuple(
            WordResult(
                text=word,
                status=statuses[index],
                acceptance_probability=probabilities[index],
                phones=(),
            )
            for index, word in enumerate(prompt.words)
        )
        return AnalysisResult(
            prompt_id=prompt.id,
            prompt_catalogue_version=prompt.catalogue_version,
            expected_text=prompt.text,
            audio=audio.metadata,
            model=ModelMetadata(
                version="test",
                acoustic_model="test",
                precision="int8",
                inference_ms=12,
            ),
            scoring=ScoringSettings(strictness=0.75, review_below=0.5, warning_below=0.8),
            words=words,
            limitations=("This is not proof of an error.",),
            warnings=("Test warning.",),
        )


def _speech() -> tuple[int, NDArray[np.int16]]:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    samples = (16_000 * np.sin(2 * np.pi * 220 * time)).astype(np.int16)
    return 16_000, samples


def test_adapter_returns_only_pronunciada_scores_and_metadata() -> None:
    analyzer = FakeAnalyzer()

    result = analyze_for_ui("thought-about-it", _speech(), lambda: analyzer)

    assert analyzer.calls == 1
    assert result.highlights == [
        ("I", REVIEW_LABEL),
        ("thought", WARNING_LABEL),
        ("about", NO_PROBLEM_LABEL),
        ("it", NO_PROBLEM_LABEL),
    ]
    assert "1 to review, 1 to check again" in result.status
    assert "inference: 12 ms" in result.status
    assert "not proof" in result.notes
    assert "Test warning" in result.notes
    assert set(result.details) == {"model_load_or_cache_lookup_ms", "analysis"}
    analysis = result.details["analysis"]
    assert analysis["schema_version"] == "2.0"
    assert analysis["words"][1] == {
        "text": "thought",
        "status": "warning",
        "acceptance_probability": 0.7,
        "phones": [],
    }
    assert "observed_phones" not in analysis
    assert "review_probability" not in analysis["words"][0]


def test_adapter_rejects_silence_before_loading_the_model() -> None:
    provider_calls = 0

    def provider() -> Analyzer:
        nonlocal provider_calls
        provider_calls += 1
        return FakeAnalyzer()

    result = analyze_for_ui(
        "thought-about-it",
        (16_000, np.zeros(8_000, dtype=np.int16)),
        provider,
    )

    assert provider_calls == 0
    assert result.highlights == []
    assert result.details["error"]["code"] == "audio_silent"
    assert "microphone" in result.notes


def test_adapter_requires_audio_without_loading_the_model() -> None:
    result = analyze_for_ui("thought-about-it", None, lambda: FakeAnalyzer())

    assert result.details["error"]["code"] == "audio_invalid"
    assert "Record or upload" in result.status


def test_adapter_does_not_persist_browser_audio(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = analyze_for_ui("thought-about-it", _speech(), lambda: FakeAnalyzer())

    assert "analysis" in result.details
    assert not tuple(tmp_path.iterdir())


def test_prompt_text_uses_the_catalogue_sentence() -> None:
    assert get_prompt("thought-about-it").text in prompt_text("thought-about-it")
