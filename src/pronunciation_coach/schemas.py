"""Validated contracts for the Pronunciada-backed application."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WordStatus(StrEnum):
    NO_PROBLEM_DETECTED = "no_problem_detected"
    WARNING = "warning"
    REVIEW = "review"


class Prompt(ContractModel):
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    text: str = Field(min_length=1)
    locale: Literal["en-us"]
    words: tuple[str, ...] = Field(min_length=1, max_length=60)
    catalogue_version: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_words(self) -> Self:
        if any(not word.strip() for word in self.words):
            raise ValueError("prompt words must not be blank")
        return self


class SpeechRegion(ContractModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    duration_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.end_ms <= self.start_ms or self.duration_ms != self.end_ms - self.start_ms:
            raise ValueError("speech region bounds are inconsistent")
        return self


class AudioMetadata(ContractModel):
    duration_ms: int = Field(ge=0)
    input_sample_rate_hz: int = Field(gt=0)
    analysis_sample_rate_hz: int = Field(gt=0)
    input_channels: int = Field(ge=1)
    clipped_sample_ratio: float = Field(ge=0.0, le=1.0)
    speech_region: SpeechRegion | None = None


class ModelMetadata(ContractModel):
    backend: Literal["pronunciada"] = "pronunciada"
    version: str
    acoustic_model: str
    precision: str
    device: Literal["cpu"] = "cpu"
    inference_ms: int = Field(ge=0)


class ScoringSettings(ContractModel):
    policy: Literal["release-pronunciada-v1"] = "release-pronunciada-v1"
    strictness: float = Field(ge=0.0)
    review_below: float = Field(ge=0.0, le=1.0)
    warning_below: float = Field(ge=0.0, le=1.0)


class PhoneResult(ContractModel):
    expected: str
    heard: str | None = None
    acceptance_probability: float = Field(ge=0.0, le=1.0)


class WordResult(ContractModel):
    text: str
    status: WordStatus
    acceptance_probability: float = Field(ge=0.0, le=1.0)
    phones: tuple[PhoneResult, ...]


class AnalysisResult(ContractModel):
    schema_version: Literal["2.0"] = "2.0"
    prompt_id: str
    prompt_catalogue_version: int = Field(ge=1)
    expected_text: str
    audio: AudioMetadata
    model: ModelMetadata
    scoring: ScoringSettings
    words: tuple[WordResult, ...]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...] = ()
