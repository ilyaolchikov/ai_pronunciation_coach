"""Deterministic recognizer used before acoustic inference is introduced."""

from dataclasses import dataclass

from pronunciation_coach.phones import normalize_phones
from pronunciation_coach.schemas import RecognitionResult


@dataclass(frozen=True, slots=True)
class FakePhoneRecognizer:
    phones: tuple[str, ...]
    revision: str = "stage0"

    def recognize(self) -> RecognitionResult:
        return RecognitionResult(
            phones=normalize_phones(self.phones),
            backend="fake",
            revision=self.revision,
            inference_ms=0,
            device=None,
        )
