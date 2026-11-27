import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pronunciation_coach.runtime as runtime
from pronunciation_coach.audio import normalize_audio
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.pronunciada_analysis import (
    ZIPA_INT8_CACHE_SHA256,
    ZIPA_INT8_SOURCE_SHA256,
    ZIPA_TOKENS_SHA256,
    PronunciadaAnalyzer,
    pin_pronunciada_download_revision,
    verify_pronunciada_assets,
)
from pronunciation_coach.schemas import WordStatus


@dataclass
class FakePhone:
    ipa: str
    prob: float
    heard: str | None = None


@dataclass
class FakeWord:
    text: str
    prob: float
    phones: list[FakePhone]


@dataclass
class FakeResult:
    words: list[FakeWord]
    timing_ms: dict[str, float] = field(default_factory=lambda: {"total": 12.0})
    warnings: list[str] = field(default_factory=list[str])
    audio: dict[str, object] | None = None


class FakeScorer:
    def score(self, audio: object, text: str, strictness: float) -> FakeResult:
        del audio, text
        assert strictness == 0.75
        return FakeResult(
            words=[
                FakeWord("I", 0.95, [FakePhone("aɪ", 0.98)]),
                FakeWord("thought", 0.30, [FakePhone("θ", 0.20, "f")]),
                FakeWord("about", 0.70, [FakePhone("ə", 0.65)]),
                FakeWord("it", 0.85, [FakePhone("ɪ", 0.90)]),
            ]
        )


class NonFiniteScorer:
    def score(self, audio: object, text: str, strictness: float) -> FakeResult:
        del audio, text, strictness
        return FakeResult(
            words=[
                FakeWord("I", float("nan"), [FakePhone("aɪ", 0.98)]),
                FakeWord("thought", 0.9, [FakePhone("θ", 0.9)]),
                FakeWord("about", 0.9, [FakePhone("ə", 0.9)]),
                FakeWord("it", 0.9, [FakePhone("ɪ", 0.9)]),
            ]
        )


def test_maps_acceptance_probabilities_to_existing_word_states() -> None:
    analyzer = PronunciadaAnalyzer(FakeScorer(), "test")
    samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000)

    result = analyzer.analyze(
        get_prompt("thought-about-it"), normalize_audio(samples.astype(np.float32), 16_000)
    )

    assert [word.status for word in result.words] == [
        WordStatus.NO_PROBLEM_DETECTED,
        WordStatus.REVIEW,
        WordStatus.WARNING,
        WordStatus.NO_PROBLEM_DETECTED,
    ]
    assert result.words[1].acceptance_probability == 0.3
    assert result.words[1].phones[0].acceptance_probability == 0.2
    assert result.words[1].phones[0].heard == "f"
    assert result.model.backend == "pronunciada"
    assert result.model.device == "cpu"
    assert result.scoring.policy == "release-pronunciada-v1"
    assert result.scoring.warning_below == 0.8


def test_rejects_non_finite_model_scores_as_analysis_failure() -> None:
    samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000)

    with pytest.raises(CoachError) as caught:
        PronunciadaAnalyzer(NonFiniteScorer(), "test").analyze(
            get_prompt("thought-about-it"),
            normalize_audio(samples.astype(np.float32), 16_000),
        )

    assert caught.value.code is ErrorCode.ANALYSIS_FAILED


def test_runtime_reports_invalid_cached_assets_as_model_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime.get_analyzer.cache_clear()

    def fake_create(*, precision: str, threads: int) -> PronunciadaAnalyzer:
        del precision, threads
        return PronunciadaAnalyzer(FakeScorer(), "test")

    monkeypatch.setattr(runtime.PronunciadaAnalyzer, "create", fake_create)

    def reject_assets(_cache_dir: Path, _precision: str) -> None:
        raise ValueError("invalid cache")

    monkeypatch.setattr(runtime, "verify_pronunciada_assets", reject_assets)

    with pytest.raises(CoachError) as caught:
        runtime.get_analyzer(tmp_path)

    assert caught.value.code is ErrorCode.MODEL_UNAVAILABLE
    runtime.get_analyzer.cache_clear()


def test_verifies_the_pinned_pronunciada_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models" / "zipa-small-crctc-500k"
    model_dir.mkdir(parents=True)
    model = model_dir / "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx"
    model.write_bytes(b"model")
    (model_dir / "tokens.txt").write_bytes(b"tokens")
    model.with_suffix(model.suffix + ".json").write_text(
        json.dumps({"source_sha256": ZIPA_INT8_SOURCE_SHA256}), encoding="utf-8"
    )
    hashes = {model: ZIPA_INT8_CACHE_SHA256, model_dir / "tokens.txt": ZIPA_TOKENS_SHA256}

    def fake_hash(path: Path) -> str:
        return hashes[path]

    monkeypatch.setattr("pronunciation_coach.pronunciada_analysis._file_sha256", fake_hash)

    verify_pronunciada_assets(tmp_path, "int8")


def test_rejects_an_unpinned_pronunciada_source(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "zipa-small-crctc-500k"
    model_dir.mkdir(parents=True)
    model = model_dir / "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx"
    model.with_suffix(model.suffix + ".json").write_text(
        json.dumps({"source_sha256": "0" * 64}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match"):
        verify_pronunciada_assets(tmp_path, "int8")


def test_pins_pronunciada_model_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    acoustic = SimpleNamespace(HF_URL="mutable")

    def fake_import(_name: str) -> SimpleNamespace:
        return acoustic

    monkeypatch.setattr("pronunciation_coach.pronunciada_analysis.import_module", fake_import)

    pin_pronunciada_download_revision()

    assert "/resolve/dfe87c97662606fa6db90ad06f627d4363314744/" in acoustic.HF_URL
