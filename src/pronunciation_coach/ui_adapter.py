"""Framework-independent adapter between browser audio and the analysis contract."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, TypeAlias
from uuid import uuid4

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from pronunciation_coach.audio import (
    NormalizedAudio,
    detect_speech_region,
    normalize_audio,
    to_float_audio,
)
from pronunciation_coach.config import DEFAULT_UI_RECORDING_DIR
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.schemas import AnalysisResult, Prompt, WordResult, WordStatus

BrowserAudio: TypeAlias = tuple[int, NDArray[np.generic]]
WordHighlights: TypeAlias = list[tuple[str, str]]

REVIEW_LABEL = "Review"
WARNING_LABEL = "Check again"
NO_PROBLEM_LABEL = "No problem detected"
YELLOW_PROBABILITY_THRESHOLD = 0.2


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
    *,
    recording_directory: Path = DEFAULT_UI_RECORDING_DIR,
) -> UIResult:
    saved_recording: Path | None = None
    try:
        if browser_audio is None:
            raise CoachError(ErrorCode.AUDIO_INVALID, "Record or upload audio first")
        sample_rate, samples = browser_audio
        prompt = get_prompt(prompt_id)
        saved_recording = _save_recording(
            prompt_id, sample_rate, np.asarray(samples), recording_directory
        )
        audio = detect_speech_region(normalize_audio(np.asarray(samples), sample_rate))

        load_started = perf_counter()
        analyzer = analyzer_provider()
        model_load_ms = round((perf_counter() - load_started) * 1000)
        result = analyzer.analyze(prompt, audio)
    except CoachError as error:
        return _error_result(error, saved_recording)

    review_count = sum(word.status is WordStatus.REVIEW for word in result.words)
    highlights = [(word.text, word_highlight_label(word)) for word in result.words]
    warning_count = sum(label == WARNING_LABEL for _, label in highlights)
    status = (
        f"Analysis complete: **{review_count} to review, {warning_count} to check again** "
        f"out of {len(result.words)} words. "
        f"Model load/warm-up: {model_load_ms} ms; inference: {result.model.inference_ms} ms. "
        f"Saved recording: `{saved_recording}`."
    )
    notes = _format_notes(result)
    details = {
        "timing": {
            "model_load_or_cache_lookup_ms": model_load_ms,
            "inference_ms": result.model.inference_ms,
        },
        "saved_recording": str(saved_recording),
        "analysis": result.model_dump(mode="json"),
    }
    return UIResult(highlights=highlights, status=status, notes=notes, details=details)


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
    if (
        word.review_probability is not None
        and word.review_probability >= YELLOW_PROBABILITY_THRESHOLD
    ):
        return WARNING_LABEL
    return NO_PROBLEM_LABEL


def _save_recording(
    prompt_id: str,
    sample_rate: int,
    samples: NDArray[np.generic],
    recording_directory: Path,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = recording_directory / f"{timestamp}-{prompt_id}-{uuid4().hex[:8]}.wav"
    temporary = destination.with_suffix(".tmp.wav")
    try:
        recording_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        recording_directory.chmod(0o700)
        sf.write(
            temporary,
            to_float_audio(samples),
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        temporary.chmod(0o600)
        temporary.replace(destination)
    except (OSError, sf.LibsndfileError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise CoachError(ErrorCode.AUDIO_SAVE_FAILED, "Recording could not be saved") from error
    return destination


def _format_notes(result: AnalysisResult) -> str:
    lines = ["### How to read this result"]
    lines.extend(f"- {item}" for item in result.limitations)
    if result.warnings:
        lines.append("\n### Recording warnings")
        lines.extend(f"- {item}" for item in result.warnings)
    return "\n".join(lines)


def _error_result(error: CoachError, saved_recording: Path | None) -> UIResult:
    guidance = {
        ErrorCode.AUDIO_INVALID: "Record or upload a WAV, FLAC, or browser-supported audio file.",
        ErrorCode.AUDIO_TOO_SHORT: "Record at least 0.25 seconds and read the whole sentence.",
        ErrorCode.AUDIO_TOO_LONG: "Keep the recording under 15 seconds, then try again.",
        ErrorCode.AUDIO_SILENT: "Move closer to the microphone and make sure it is not muted.",
        ErrorCode.AUDIO_SAVE_FAILED: "Check that the recordings directory is writable.",
        ErrorCode.MODEL_UNAVAILABLE: "Prefetch the model, then restart the app.",
        ErrorCode.PROMPT_UNKNOWN: "Select one of the available prompts and try again.",
    }.get(error.code, "Try the recording again. If it keeps failing, restart the app.")
    details: dict[str, Any] = {"error": {"code": error.code.value, "message": str(error)}}
    if saved_recording is not None:
        guidance += f"\n\nRecording saved to `{saved_recording}`."
        details["saved_recording"] = str(saved_recording)
    return UIResult(
        highlights=[],
        status=f"Could not analyze the recording: **{error}**",
        notes=guidance,
        details=details,
    )
