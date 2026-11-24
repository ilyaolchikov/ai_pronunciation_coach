from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pronunciation_coach.audio import NormalizedAudio
from pronunciation_coach.intentional_benchmark import (
    Analyzer,
    IntentionalBenchmarkItem,
    IntentionalBenchmarkManifest,
    load_intentional_manifest,
    run_intentional_benchmark,
)
from pronunciation_coach.schemas import (
    AnalysisResult,
    ModelMetadata,
    Prompt,
    WordResult,
    WordStatus,
)

MANIFEST = Path("benchmarks/intentional-errors/manifest.v1.json")


class FakeAnalyzer:
    def __init__(self) -> None:
        self.calls = 0
        self.probabilities = (
            (0.2, 0.1, 0.1),
            (0.1, 0.4, 0.1),
            (0.1, 0.1, 0.1),
        )

    def analyze(self, prompt: Prompt, audio: NormalizedAudio) -> AnalysisResult:
        probabilities = self.probabilities[self.calls]
        self.calls += 1
        words = tuple(
            WordResult(
                text=word.text,
                status=(
                    WordStatus.REVIEW
                    if probabilities[index] >= 0.3
                    else WordStatus.NO_PROBLEM_DETECTED
                ),
                evidence=(),
                review_probability=probabilities[index],
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
            model=ModelMetadata(backend="fake", revision="test", inference_ms=1),
            words=words,
            limitations=(),
        )


def _item(item_id: str, audio_path: str, *, target: int | None) -> IntentionalBenchmarkItem:
    return IntentionalBenchmarkItem(
        id=item_id,
        speaker="test-speaker",
        prompt_id="three-free-throws",
        audio_path=Path(audio_path),
        spoken_text="Three free throws." if target is None else "Free free throws.",
        target_word_index=target,
        expected_phone=None if target is None else "θ",
        substituted_phone=None if target is None else "f",
    )


def _manifest() -> IntentionalBenchmarkManifest:
    return IntentionalBenchmarkManifest(
        purpose="development-only-intentional-single-phone-localization",
        items=(
            _item("localized", "localized.wav", target=0),
            _item("wrong-word", "wrong-word.wav", target=0),
            _item("control", "control.wav", target=None),
        ),
    )


def test_bundled_manifest_has_three_substitutions_and_controls() -> None:
    manifest = load_intentional_manifest(MANIFEST)

    assert len(manifest.items) == 6
    assert sum(not item.is_control for item in manifest.items) == 3
    assert {item.prompt_id for item in manifest.items} == {
        "three-free-throws",
        "think-about-it",
        "turn-right-here",
    }


def test_measures_target_localization_separately_from_any_warning(tmp_path: Path) -> None:
    samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(8_000) / 16_000)
    for item in _manifest().items:
        sf.write(tmp_path / item.audio_path, samples, 16_000)

    report = run_intentional_benchmark(
        _manifest(),
        tmp_path,
        FakeAnalyzer,
        manifest_sha256="test",
    )

    metrics = report["metrics"]
    assert metrics["any_warning_rate"] == 1.0
    assert metrics["target_localization_rate"] == 0.5
    assert metrics["exact_localization_rate"] == 0.5
    assert metrics["wrong_word_only_rate"] == 0.5
    assert metrics["control_false_warning_rate"] == 0.0
    first_word = report["items"][0]["word_results"][0]
    assert first_word["status"] == "no_problem_detected"
    assert first_word["phone_error_probability"] is None


def test_missing_recordings_do_not_load_the_model(tmp_path: Path) -> None:
    provider_calls = 0

    def provider() -> Analyzer:
        nonlocal provider_calls
        provider_calls += 1
        return FakeAnalyzer()

    with pytest.raises(ValueError, match="Missing intentional benchmark audio"):
        run_intentional_benchmark(_manifest(), tmp_path, provider, manifest_sha256="test")

    assert provider_calls == 0
