"""Backend-neutral phone recognizer contract."""

from typing import Protocol

from pronunciation_coach.audio import NormalizedAudio
from pronunciation_coach.schemas import RecognitionResult


class PhoneRecognizer(Protocol):
    def recognize(self, audio: NormalizedAudio) -> RecognitionResult: ...
