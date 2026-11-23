# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
"""Stage 2 calibrated word-level analysis."""

import json
from dataclasses import dataclass
from importlib.resources import files
from itertools import product
from typing import Any, Protocol, cast

from pronunciation_coach.alignment import align_prompt
from pronunciation_coach.audio import NormalizedAudio, detect_speech_region
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.schemas import (
    AnalysisResult,
    ModelMetadata,
    PolicyMetadata,
    Prompt,
    WordResult,
    WordStatus,
)
from pronunciation_coach.word_calibration import predict_word_errors, validate_word_model

CALIBRATED_LIMITATIONS = (
    "A highlighted word is a calibrated model estimate, not proof of an error.",
    "No problem detected does not mean perfect pronunciation.",
    "The current evidence localizes words, not a reliably incorrect individual sound.",
)


class PhoneScore(Protocol):
    @property
    def target_index(self) -> int: ...

    @property
    def phone(self) -> str: ...

    @property
    def frame_count(self) -> int: ...

    @property
    def expected_mean_log_probability(self) -> float: ...

    @property
    def alternative_mean_log_probability(self) -> float: ...

    @property
    def log_posterior_margin(self) -> float: ...


class Alignment(Protocol):
    @property
    def path_mean_log_probability(self) -> float: ...

    @property
    def phone_scores(self) -> tuple[PhoneScore, ...]: ...


class PosteriorResult(Protocol):
    @property
    def alignment(self) -> Alignment: ...

    @property
    def greedy_phones(self) -> tuple[str, ...]: ...

    @property
    def inference_ms(self) -> int: ...

    @property
    def device(self) -> str: ...


class PosteriorRecognizer(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def score_expected_variants(
        self,
        audio: NormalizedAudio,
        expected_variants: tuple[tuple[str, ...], ...],
        *,
        competing_phones: frozenset[str] | None = None,
    ) -> tuple[PosteriorResult, ...]: ...


@dataclass(frozen=True, slots=True)
class WordPolicy:
    id: str
    version: int
    model_id: str
    model_revision: str
    calibration_report_sha256: str
    candidate_model: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.candidate_model["word_probability_threshold"])

    @property
    def competing_phones(self) -> frozenset[str]:
        categories = cast(list[str], self.candidate_model["phone_categories"])
        return frozenset(phone for phone in categories if phone != "<other>")


def load_word_policy() -> WordPolicy:
    resource = files("pronunciation_coach.data").joinpath("word_policy.v1.json")
    try:
        raw: object = json.loads(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("word policy must be an object")
        value = cast(dict[str, Any], raw)
        candidate = value["candidate_model"]
        if not isinstance(candidate, dict):
            raise ValueError("candidate model must be an object")
        candidate_model = cast(dict[str, Any], candidate)
        validate_word_model(candidate_model)
        policy = WordPolicy(
            id=str(value["id"]),
            version=int(value["version"]),
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            calibration_report_sha256=str(value["calibration_report_sha256"]),
            candidate_model=candidate_model,
        )
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        raise CoachError(ErrorCode.CATALOGUE_INVALID, "Word policy is invalid") from error
    if len(policy.calibration_report_sha256) != 64 or not set(
        policy.calibration_report_sha256
    ) <= set("0123456789abcdef"):
        raise CoachError(ErrorCode.CATALOGUE_INVALID, "Word policy provenance is invalid")
    return policy


class CalibratedAnalyzer:
    def __init__(self, recognizer: PosteriorRecognizer, policy: WordPolicy | None = None) -> None:
        self.recognizer = recognizer
        self.policy = load_word_policy() if policy is None else policy
        if (
            recognizer.model_id != self.policy.model_id
            or recognizer.model_revision != self.policy.model_revision
        ):
            raise CoachError(ErrorCode.MODEL_UNAVAILABLE, "Model does not match the word policy")

    def analyze(self, prompt: Prompt, audio: NormalizedAudio) -> AnalysisResult:
        if audio.metadata.speech_region is None:
            audio = detect_speech_region(audio)
        variants = _prompt_variants(prompt)
        try:
            recognitions = self.recognizer.score_expected_variants(
                audio,
                tuple(item[1] for item in variants),
                competing_phones=self.policy.competing_phones,
            )
            if len(recognitions) != len(variants):
                raise ValueError("recognizer returned the wrong number of variants")
            selected_index = max(
                range(len(recognitions)),
                key=lambda index: recognitions[index].alignment.path_mean_log_probability,
            )
            selected = recognitions[selected_index]
            variant_indices, expected_phones, word_indexes = variants[selected_index]
            phone_scores = [
                {
                    "word_index": word_indexes[score.target_index],
                    "phone": score.phone,
                    "frame_count": score.frame_count,
                    "expected_mean_log_probability": score.expected_mean_log_probability,
                    "alternative_mean_log_probability": score.alternative_mean_log_probability,
                    "log_posterior_margin": score.log_posterior_margin,
                }
                for score in selected.alignment.phone_scores
            ]
            predictions = predict_word_errors(phone_scores, self.policy.candidate_model)
            by_word = {int(item["word_index"]): item for item in predictions}
            if set(by_word) != set(range(len(prompt.words))):
                raise ValueError("word classifier did not score every prompt word")
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise CoachError(ErrorCode.ANALYSIS_FAILED, "Audio could not be analyzed") from error

        diagnostic_alignment = align_prompt(prompt, selected.greedy_phones)
        words = tuple(
            WordResult(
                text=word.text,
                status=(
                    WordStatus.REVIEW
                    if bool(by_word[index]["review"])
                    else WordStatus.NO_PROBLEM_DETECTED
                ),
                evidence=(),
                review_probability=float(by_word[index]["probability"]),
            )
            for index, word in enumerate(prompt.words)
        )
        warnings = (
            ("The recording contains substantial clipping.",)
            if audio.metadata.clipped_sample_ratio >= 0.01
            else ()
        )
        return AnalysisResult(
            prompt_id=prompt.id,
            prompt_catalogue_version=prompt.catalogue_version,
            expected_text=prompt.text,
            selected_variant_indices=variant_indices,
            alignment_cost=diagnostic_alignment.cost,
            expected_phones=expected_phones,
            observed_phones=selected.greedy_phones,
            audio=audio.metadata,
            model=ModelMetadata(
                backend="wav2vec2-espeak-ctc",
                revision=self.policy.model_revision,
                inference_ms=selected.inference_ms,
                device=selected.device,
            ),
            policy=PolicyMetadata(
                id=self.policy.id,
                version=self.policy.version,
                word_probability_threshold=self.policy.threshold,
                calibration_report_sha256=self.policy.calibration_report_sha256,
            ),
            words=words,
            limitations=CALIBRATED_LIMITATIONS,
            warnings=warnings,
        )


def _prompt_variants(
    prompt: Prompt,
) -> tuple[tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]], ...]:
    variants: list[tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]] = []
    ranges = tuple(range(len(word.phone_variants)) for word in prompt.words)
    for variant_indices in product(*ranges):
        phones: list[str] = []
        word_indexes: list[int] = []
        for word_index, variant_index in enumerate(variant_indices):
            variant = prompt.words[word_index].phone_variants[variant_index]
            phones.extend(variant)
            word_indexes.extend([word_index] * len(variant))
        variants.append((variant_indices, tuple(phones), tuple(word_indexes)))
    return tuple(variants)
