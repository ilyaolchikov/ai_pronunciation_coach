"""Framework-independent adapter between browser audio and Pronunciada."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pronunciation_coach.audio import (
    NormalizedAudio,
    detect_speech_region,
    normalize_audio,
)
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.schemas import AnalysisResult, Prompt, WordResult, WordStatus

BrowserAudio: TypeAlias = tuple[int, NDArray[np.generic]]
WordHighlights: TypeAlias = list[tuple[str, str]]

REVIEW_LABEL = "Review"
WARNING_LABEL = "Check again"
NO_PROBLEM_LABEL = "No problem detected"


class Analyzer(Protocol):
    def analyze(self, prompt: Prompt, audio: NormalizedAudio) -> AnalysisResult: ...


@dataclass(frozen=True, slots=True)
class UIResult:
    highlights: WordHighlights
    status: str
    notes: str
    details: dict[str, Any]


def analyze_for_ui(
    prompt_id: str,
    browser_audio: BrowserAudio | None,
    analyzer_provider: Callable[[], Analyzer],
) -> UIResult:
    try:
        if browser_audio is None:
            raise CoachError(ErrorCode.AUDIO_INVALID, "Record or upload audio first")
        sample_rate, samples = browser_audio
        prompt = get_prompt(prompt_id)
        audio = detect_speech_region(normalize_audio(np.asarray(samples), sample_rate))

        load_started = perf_counter()
        analyzer = analyzer_provider()
        model_load_ms = round((perf_counter() - load_started) * 1000)
        result = analyzer.analyze(prompt, audio)
    except CoachError as error:
        return _error_result(error)

    review_count = sum(word.status is WordStatus.REVIEW for word in result.words)
    warning_count = sum(word.status is WordStatus.WARNING for word in result.words)
    status = (
        f"Analysis complete: **{review_count} to review, {warning_count} to check again** "
        f"out of {len(result.words)} words. Model load/warm-up: {model_load_ms} ms; "
        f"inference: {result.model.inference_ms} ms."
    )
    return UIResult(
        highlights=[(word.text, word_highlight_label(word)) for word in result.words],
        status=status,
        notes=_format_notes(result),
        details={
            "model_load_or_cache_lookup_ms": model_load_ms,
            "analysis": result.model_dump(mode="json"),
        },
    )


def prompt_text(prompt_id: str) -> str:
    try:
        return f"### Read this aloud\n\n{get_prompt(prompt_id).text}"
    except CoachError as error:
        return f"Could not load the prompt: {error}"


def word_highlight_label(word: WordResult) -> str:
    if word.status is WordStatus.REVIEW:
        return REVIEW_LABEL
    if word.status is WordStatus.WARNING:
        return WARNING_LABEL
    return NO_PROBLEM_LABEL


def _format_notes(result: AnalysisResult) -> str:
    lines = ["### How to read this result", *(f"- {item}" for item in result.limitations)]
    if result.warnings:
        lines.append("\n### Recording warnings")
        lines.extend(f"- {item}" for item in result.warnings)
    return "\n".join(lines)


def _error_result(error: CoachError) -> UIResult:
    guidance = {
        ErrorCode.AUDIO_INVALID: "Record or upload a browser-supported audio file.",
        ErrorCode.AUDIO_TOO_SHORT: "Record at least 0.25 seconds and read the whole sentence.",
        ErrorCode.AUDIO_TOO_LONG: "Keep the recording under 15 seconds.",
        ErrorCode.AUDIO_SILENT: "Check the microphone and speak clearly before trying again.",
        ErrorCode.MODEL_UNAVAILABLE: "Install Pronunciada and ensure its model is available.",
    }.get(error.code, "Try another recording. If the problem continues, restart the app.")
    return UIResult(
        highlights=[],
        status=f"Analysis could not be completed: **{error}**",
        notes=guidance,
        details={"error": {"code": error.code.value, "message": str(error)}},
    )
