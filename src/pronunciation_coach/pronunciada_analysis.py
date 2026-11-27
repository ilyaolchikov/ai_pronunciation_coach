"""Pronunciada-backed pronunciation analysis."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

from pronunciation_coach.audio import NormalizedAudio, detect_speech_region
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.schemas import (
    AnalysisResult,
    ModelMetadata,
    PhoneResult,
    Prompt,
    ScoringSettings,
    WordResult,
    WordStatus,
)

STRICTNESS = 0.75
WARNING_ACCEPTANCE_THRESHOLD = 0.8
REVIEW_ACCEPTANCE_THRESHOLD = 0.5
PRECISION = "int8"
ZIPA_MODEL = "zipa-small-crctc-500k"
ZIPA_REVISION = "dfe87c97662606fa6db90ad06f627d4363314744"
ZIPA_INT8_SOURCE_SHA256 = "d0e28b68164e8b1fbd6105100c01798828aa0855000ce9bbbd1a2cec233adf13"
ZIPA_INT8_CACHE_SHA256 = "36a7cdc229e08c57b6c568fc0a4b6dc697c01f435b09183699b9a1f5b19c034c"
ZIPA_TOKENS_SHA256 = "f8e042a0c9130532b22d03ec7cae2f75a23fbec70c450c31a8efb51787b2b8fe"

LIMITATIONS = (
    "A highlighted word is a calibrated model estimate, not proof of an error.",
    "No problem detected does not mean perfect pronunciation.",
    "Strictness 0.75 is an experimental UI setting and still needs validation on human "
    "recordings from more speakers.",
)


class _PhoneScore(Protocol):
    ipa: str
    prob: float
    heard: str | None


class _WordScore(Protocol):
    text: str
    prob: float

    @property
    def phones(self) -> Sequence[_PhoneScore]: ...


class _SentenceScore(Protocol):
    @property
    def words(self) -> Sequence[_WordScore]: ...

    @property
    def warnings(self) -> Sequence[str]: ...

    @property
    def audio(self) -> Mapping[str, object] | None: ...


class _Scorer(Protocol):
    def score(self, audio: object, text: str, strictness: float) -> _SentenceScore: ...


class PronunciadaAnalyzer:
    def __init__(self, scorer: _Scorer, version: str, precision: str = PRECISION) -> None:
        self.scorer = scorer
        self.version = version
        self.precision = precision

    @classmethod
    def create(cls, *, precision: str = PRECISION, threads: int = 4) -> PronunciadaAnalyzer:
        try:
            module = import_module("pronunciada")
        except ImportError as error:
            raise CoachError(ErrorCode.MODEL_UNAVAILABLE, "Pronunciada is not installed") from error
        pin_pronunciada_download_revision()
        scorer_module = import_module("pronunciada.scorer")
        scorer_type = cast(Callable[..., _Scorer], scorer_module.__dict__["Scorer"])
        try:
            scorer = scorer_type("en", precision=precision, threads=threads)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise CoachError(ErrorCode.MODEL_UNAVAILABLE, "Pronunciada could not load") from error
        return cls(scorer, str(module.__dict__.get("__version__", "unknown")), precision)

    def analyze(self, prompt: Prompt, audio: NormalizedAudio) -> AnalysisResult:
        if audio.metadata.speech_region is None:
            audio = detect_speech_region(audio)
        started = perf_counter()
        try:
            scored = self.scorer.score(audio.samples, prompt.text, strictness=STRICTNESS)
            self._validate_words(prompt, scored)
            warnings = list(scored.warnings)
            if scored.audio is not None:
                issues = scored.audio.get("issues")
                if isinstance(issues, list):
                    warnings.extend(
                        issue
                        for issue in cast(list[object], issues)
                        if isinstance(issue, str) and issue
                    )
            result = AnalysisResult(
                prompt_id=prompt.id,
                prompt_catalogue_version=prompt.catalogue_version,
                expected_text=prompt.text,
                audio=audio.metadata,
                model=ModelMetadata(
                    version=self.version,
                    acoustic_model=f"{ZIPA_MODEL}@{ZIPA_REVISION}",
                    precision=self.precision,
                    inference_ms=round((perf_counter() - started) * 1000),
                ),
                scoring=ScoringSettings(
                    strictness=STRICTNESS,
                    review_below=REVIEW_ACCEPTANCE_THRESHOLD,
                    warning_below=WARNING_ACCEPTANCE_THRESHOLD,
                ),
                words=tuple(self._word_result(word) for word in scored.words),
                limitations=LIMITATIONS,
                warnings=tuple(dict.fromkeys(warnings)),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise CoachError(ErrorCode.ANALYSIS_FAILED, "Audio could not be analyzed") from error
        return result

    @staticmethod
    def _validate_words(prompt: Prompt, scored: _SentenceScore) -> None:
        expected = [word.casefold() for word in prompt.words]
        actual = [word.text.casefold() for word in scored.words]
        if actual != expected:
            raise ValueError("Pronunciada result words do not match the prompt")

    @staticmethod
    def _word_result(word: _WordScore) -> WordResult:
        if word.prob < REVIEW_ACCEPTANCE_THRESHOLD:
            status = WordStatus.REVIEW
        elif word.prob < WARNING_ACCEPTANCE_THRESHOLD:
            status = WordStatus.WARNING
        else:
            status = WordStatus.NO_PROBLEM_DETECTED
        return WordResult(
            text=word.text,
            status=status,
            acceptance_probability=word.prob,
            phones=tuple(
                PhoneResult(
                    expected=phone.ipa,
                    heard=phone.heard,
                    acceptance_probability=phone.prob,
                )
                for phone in word.phones
            ),
        )


def verify_pronunciada_assets(cache_dir: Path, precision: str) -> None:
    """Refuse a mutable upstream ZIPA download that differs from the benchmarked int8 model."""
    if precision != PRECISION:
        return
    model_dir = cache_dir / "models" / ZIPA_MODEL
    model = model_dir / "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx"
    sidecar = model.with_suffix(model.suffix + ".json")
    tokens = model_dir / "tokens.txt"
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Pronunciada ZIPA cache metadata is missing or invalid") from error
    if metadata.get("source_sha256") != ZIPA_INT8_SOURCE_SHA256:
        raise ValueError("Pronunciada ZIPA source does not match the pinned release model")
    if _file_sha256(model) != ZIPA_INT8_CACHE_SHA256 or _file_sha256(tokens) != ZIPA_TOKENS_SHA256:
        raise ValueError("Pronunciada ZIPA cache does not match the pinned release assets")


def pin_pronunciada_download_revision() -> None:
    """Replace Pronunciada 1.1.0's mutable model URL with the benchmarked ZIPA revision."""
    acoustic = import_module("pronunciada.acoustic")
    acoustic.__dict__["HF_URL"] = (
        f"https://huggingface.co/anyspeech/{{name}}/resolve/{ZIPA_REVISION}/{{file}}"
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Pronunciada asset cannot be read: {path}") from error
    return digest.hexdigest()
