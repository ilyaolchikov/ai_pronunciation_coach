from pathlib import Path

import numpy as np
import soundfile as sf
from pytest import MonkeyPatch

from pronunciation_coach.audio import NormalizedAudio
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.schemas import (
    AnalysisResult,
    ModelMetadata,
    Prompt,
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
        words = tuple(
            WordResult(
                text=word.text,
                status=WordStatus.REVIEW if index == 0 else WordStatus.NO_PROBLEM_DETECTED,
                evidence=(),
                review_probability=(0.8, 0.2, 0.1, 0.1)[index],
            )
            for index, word in enumerate(prompt.words)
        )
        expected = tuple(phone for word in prompt.words for phone in word.phone_variants[0])
        return AnalysisResult(
            prompt_id=prompt.id,
            prompt_catalogue_version=prompt.catalogue_version,
            expected_text=prompt.text,
            selected_variant_indices=tuple(0 for _ in prompt.words),
            alignment_cost=0,
            expected_phones=expected,
            observed_phones=expected,
            audio=audio.metadata,
            model=ModelMetadata(
                backend="fake",
                revision="test",
                inference_ms=12,
                device="cpu",
            ),
            words=words,
            limitations=("This is not proof of an error.",),
            warnings=("Test warning.",),
        )


def _speech() -> tuple[int, np.ndarray]:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    samples = (16_000 * np.sin(2 * np.pi * 220 * time)).astype(np.int16)
    return 16_000, samples


def test_adapter_returns_word_states_notes_and_technical_details(tmp_path: Path) -> None:
    analyzer = FakeAnalyzer()

    result = analyze_for_ui(
        "thought-about-it", _speech(), lambda: analyzer, recording_directory=tmp_path
    )

    assert analyzer.calls == 1
    assert result.highlights[0] == ("I", REVIEW_LABEL)
    assert result.highlights[1] == ("thought", WARNING_LABEL)
    assert {category for _, category in result.highlights[2:]} == {NO_PROBLEM_LABEL}
    assert "1 to review, 1 to check again" in result.status
    assert "inference: 12 ms" in result.status
    assert "not proof" in result.notes
    assert "Test warning" in result.notes
    assert result.details["analysis"]["schema_version"] == "1.3"
    saved_recording = Path(result.details["saved_recording"])
    assert saved_recording.parent == tmp_path
    assert saved_recording.suffix == ".wav"
    assert sf.info(saved_recording).samplerate == 16_000


def test_adapter_rejects_silence_before_loading_the_model(tmp_path: Path) -> None:
    provider_calls = 0

    def provider() -> Analyzer:
        nonlocal provider_calls
        provider_calls += 1
        return FakeAnalyzer()

    result = analyze_for_ui(
        "thought-about-it",
        (16_000, np.zeros(8_000, dtype=np.int16)),
        provider,
        recording_directory=tmp_path,
    )

    assert provider_calls == 0
    assert result.highlights == []
    assert result.details["error"]["code"] == "audio_silent"
    assert Path(result.details["saved_recording"]).is_file()
    assert "microphone" in result.notes


def test_adapter_requires_audio_without_loading_the_model(tmp_path: Path) -> None:
    def provider() -> Analyzer:
        return FakeAnalyzer()

    result = analyze_for_ui("thought-about-it", None, provider, recording_directory=tmp_path)

    assert result.details["error"]["code"] == "audio_invalid"
    assert "Record or upload" in result.status
    assert not tuple(tmp_path.iterdir())


def test_save_failure_prevents_analysis(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    analyzer = FakeAnalyzer()

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr("pronunciation_coach.ui_adapter.sf.write", fail_write)

    result = analyze_for_ui(
        "thought-about-it", _speech(), lambda: analyzer, recording_directory=tmp_path
    )

    assert analyzer.calls == 0
    assert result.details["error"]["code"] == "audio_save_failed"
    assert not tuple(tmp_path.glob("*"))


def test_prompt_text_uses_the_catalogue_sentence() -> None:
    assert get_prompt("thought-about-it").text in prompt_text("thought-about-it")
