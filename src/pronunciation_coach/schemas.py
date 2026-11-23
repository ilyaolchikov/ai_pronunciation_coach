"""Versioned contracts shared across the deterministic analysis pipeline."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Operation(StrEnum):
    MATCH = "match"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


class WordStatus(StrEnum):
    MATCH = "match"
    NO_PROBLEM_DETECTED = "no_problem_detected"
    REVIEW = "review"
    NOT_DETECTED = "not_detected"
    UNASSESSABLE = "unassessable"


class PromptWord(ContractModel):
    text: str = Field(min_length=1)
    phone_variants: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_phone_variants(self) -> Self:
        if any(not variant for variant in self.phone_variants):
            raise ValueError("phone variants must not be empty")
        if any(not phone.strip() for variant in self.phone_variants for phone in variant):
            raise ValueError("phones must not be blank")
        if len(set(self.phone_variants)) != len(self.phone_variants):
            raise ValueError("phone variants must be unique")
        return self


class GenerationMetadata(ContractModel):
    phonemizer: str
    espeak_ng: str


class Prompt(ContractModel):
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    text: str = Field(min_length=1)
    locale: str = Field(pattern=r"^[a-z]{2}-[a-z]{2}$")
    words: tuple[PromptWord, ...] = Field(min_length=1, max_length=20)
    catalogue_version: int = Field(ge=1)
    generated_with: GenerationMetadata | None = None

    @model_validator(mode="after")
    def bound_variant_combinations(self) -> Self:
        combinations = 1
        for word in self.words:
            combinations *= len(word.phone_variants)
        if combinations > 256:
            raise ValueError("prompt has more than 256 pronunciation combinations")
        return self


class RecognitionResult(ContractModel):
    phones: tuple[str, ...]
    backend: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    inference_ms: int = Field(ge=0)
    device: str | None = None
    diagnostics: tuple[str, ...] = ()


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


class PhoneEvidence(ContractModel):
    expected: str | None
    observed: str | None
    operation: Operation
    word_index: int | None = Field(default=None, ge=0)
    phone_index: int | None = Field(default=None, ge=0)
    boundary_ambiguous: bool = False
    displayable: bool = False

    @model_validator(mode="after")
    def validate_operation_shape(self) -> Self:
        if self.operation is Operation.MATCH:
            if self.expected is None or self.expected != self.observed:
                raise ValueError("a match requires equal expected and observed phones")
        elif self.operation is Operation.SUBSTITUTION:
            if self.expected is None or self.observed is None or self.expected == self.observed:
                raise ValueError("a substitution requires two different phones")
        elif self.operation is Operation.DELETION:
            if self.expected is None or self.observed is not None:
                raise ValueError("a deletion requires only an expected phone")
        elif self.operation is Operation.INSERTION and (
            self.expected is not None or self.observed is None
        ):
            raise ValueError("an insertion requires only an observed phone")
        return self


class AlignmentResult(ContractModel):
    cost: int = Field(ge=0)
    selected_variant_indices: tuple[int, ...]
    expected_phones: tuple[str, ...]
    evidence: tuple[PhoneEvidence, ...]


class ModelMetadata(ContractModel):
    backend: str
    revision: str
    inference_ms: int = Field(ge=0)
    device: str | None = None


class PolicyMetadata(ContractModel):
    id: str
    version: int = Field(ge=1)
    word_probability_threshold: float = Field(gt=0.0, lt=1.0)
    calibration_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WordResult(ContractModel):
    text: str
    status: WordStatus
    evidence: tuple[PhoneEvidence, ...]
    tips: tuple[str, ...] = ()
    review_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalysisResult(ContractModel):
    schema_version: str = "1.2"
    prompt_id: str
    prompt_catalogue_version: int = Field(ge=1)
    expected_text: str
    selected_variant_indices: tuple[int, ...]
    alignment_cost: int = Field(ge=0)
    expected_phones: tuple[str, ...]
    observed_phones: tuple[str, ...]
    audio: AudioMetadata | None = None
    model: ModelMetadata
    policy: PolicyMetadata | None = None
    words: tuple[WordResult, ...]
    limitations: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class FeedbackTip(ContractModel):
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    expected: str
    observed: str
    message: str = Field(min_length=1)
    examples: tuple[str, ...] = Field(min_length=1)
