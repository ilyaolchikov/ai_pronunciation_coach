"""Expected domain errors exposed by the CLI and future UI adapters."""

from enum import StrEnum


class ErrorCode(StrEnum):
    PROMPT_UNKNOWN = "prompt_unknown"
    CATALOGUE_INVALID = "catalogue_invalid"
    RECOGNITION_INVALID = "recognition_invalid"
    ANALYSIS_FAILED = "analysis_failed"
    AUDIO_DECODE_FAILED = "audio_decode_failed"
    AUDIO_FILE_TOO_LARGE = "audio_file_too_large"
    AUDIO_INVALID = "audio_invalid"
    AUDIO_SILENT = "audio_silent"
    AUDIO_TOO_LONG = "audio_too_long"
    AUDIO_TOO_SHORT = "audio_too_short"
    MODEL_UNAVAILABLE = "model_unavailable"


class CoachError(RuntimeError):
    """An expected failure with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
