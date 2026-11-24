"""Development benchmark for intentional single-phone substitutions."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import Field, model_validator

from pronunciation_coach.audio import NormalizedAudio, load_audio
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.schemas import AnalysisResult, ContractModel, Prompt
from pronunciation_coach.ui_adapter import NO_PROBLEM_LABEL, word_highlight_label


class IntentionalBenchmarkItem(ContractModel):
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    speaker: str = Field(min_length=1)
    prompt_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    audio_path: Path
    spoken_text: str = Field(min_length=1)
    target_word_index: int | None = Field(default=None, ge=0)
    expected_phone: str | None = None
    substituted_phone: str | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.audio_path.is_absolute() or ".." in self.audio_path.parts:
            raise ValueError("audio_path must stay relative to the manifest")
        substitution = (
            self.target_word_index,
            self.expected_phone,
            self.substituted_phone,
        )
        if all(value is None for value in substitution):
            return self
        if any(value is None for value in substitution):
            raise ValueError("substitution cases require a word index and two phones")
        if self.expected_phone == self.substituted_phone:
            raise ValueError("substitution phones must differ")
        return self

    @property
    def is_control(self) -> bool:
        return self.target_word_index is None


class IntentionalBenchmarkManifest(ContractModel):
    version: Literal[1] = 1
    purpose: Literal["development-only-intentional-single-phone-localization"]
    items: tuple[IntentionalBenchmarkItem, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_items(self) -> Self:
        ids = [item.id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("intentional benchmark item IDs must be unique")
        if all(item.is_control for item in self.items):
            raise ValueError("intentional benchmark requires a substitution case")
        if all(not item.is_control for item in self.items):
            raise ValueError("intentional benchmark requires a correct control")
        return self


class Analyzer(Protocol):
    def analyze(self, prompt: Prompt, audio: NormalizedAudio) -> AnalysisResult: ...


def load_intentional_manifest(path: Path) -> IntentionalBenchmarkManifest:
    return IntentionalBenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))


def run_intentional_benchmark(
    manifest: IntentionalBenchmarkManifest,
    manifest_directory: Path,
    analyzer_provider: Callable[[], Analyzer],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    prompts = {item.id: _validated_prompt(item) for item in manifest.items}
    audio_paths = {item.id: manifest_directory / item.audio_path for item in manifest.items}
    missing = [
        str(audio_paths[item.id]) for item in manifest.items if not audio_paths[item.id].is_file()
    ]
    if missing:
        raise ValueError(f"Missing intentional benchmark audio: {', '.join(missing)}")

    analyzer = analyzer_provider()
    completed: list[dict[str, Any]] = []
    for item in manifest.items:
        result = analyzer.analyze(prompts[item.id], load_audio(audio_paths[item.id]))
        labels = tuple(word_highlight_label(word) for word in result.words)
        flagged = tuple(index for index, label in enumerate(labels) if label != NO_PROBLEM_LABEL)
        target_localized = None if item.is_control else item.target_word_index in flagged
        exact_localization = None if item.is_control else flagged == (item.target_word_index,)
        wrong_word_only = (
            None if item.is_control else bool(flagged) and item.target_word_index not in flagged
        )
        completed.append(
            {
                "id": item.id,
                "speaker": item.speaker,
                "prompt_id": item.prompt_id,
                "spoken_text": item.spoken_text,
                "target_word_index": item.target_word_index,
                "expected_phone": item.expected_phone,
                "substituted_phone": item.substituted_phone,
                "word_results": [
                    {
                        "word_index": index,
                        "text": word.text,
                        "status": word.status.value,
                        "label": labels[index],
                        "review_probability": word.review_probability,
                        "phone_error_probability": word.phone_error_probability,
                    }
                    for index, word in enumerate(result.words)
                ],
                "flagged_word_indexes": flagged,
                "target_localized": target_localized,
                "exact_localization": exact_localization,
                "wrong_word_only": wrong_word_only,
                "control_pass": None if not item.is_control else not flagged,
                "alignment_cost": result.alignment_cost,
                "expected_phones": result.expected_phones,
                "observed_phones": result.observed_phones,
            }
        )

    substitutions = [item for item in completed if item["target_word_index"] is not None]
    controls = [item for item in completed if item["target_word_index"] is None]
    target_localized_count = sum(item["target_localized"] is True for item in substitutions)
    exact_localization_count = sum(item["exact_localization"] is True for item in substitutions)
    wrong_word_only_count = sum(item["wrong_word_only"] is True for item in substitutions)
    any_warning_count = sum(bool(item["flagged_word_indexes"]) for item in substitutions)
    control_false_warning_count = sum(item["control_pass"] is False for item in controls)
    return {
        "report_version": 1,
        "experiment": "intentional-single-phone-localization-v1",
        "development_only": True,
        "manifest_sha256": manifest_sha256,
        "metrics": {
            "substitution_cases": len(substitutions),
            "target_localized_cases": target_localized_count,
            "target_localization_rate": _rate(target_localized_count, len(substitutions)),
            "exact_localization_cases": exact_localization_count,
            "exact_localization_rate": _rate(exact_localization_count, len(substitutions)),
            "wrong_word_only_cases": wrong_word_only_count,
            "wrong_word_only_rate": _rate(wrong_word_only_count, len(substitutions)),
            "any_warning_cases": any_warning_count,
            "any_warning_rate": _rate(any_warning_count, len(substitutions)),
            "control_cases": len(controls),
            "control_false_warning_cases": control_false_warning_count,
            "control_false_warning_rate": _rate(control_false_warning_count, len(controls)),
        },
        "items": completed,
    }


def _validated_prompt(item: IntentionalBenchmarkItem) -> Prompt:
    prompt = get_prompt(item.prompt_id)
    if item.is_control:
        return prompt
    if item.target_word_index is None or item.target_word_index >= len(prompt.words):
        raise ValueError(f"Target word is outside prompt for {item.id}")
    variants = prompt.words[item.target_word_index].phone_variants
    if not any(item.expected_phone in variant for variant in variants):
        raise ValueError(f"Expected phone is not in the target word for {item.id}")
    return prompt


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator
