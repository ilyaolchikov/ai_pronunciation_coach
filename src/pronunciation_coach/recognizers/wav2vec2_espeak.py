# pyright: reportUnknownMemberType=false
"""Pinned Wav2Vec2 eSpeak CTC phone recognizer."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCTC, AutoProcessor

from pronunciation_coach.audio import NormalizedAudio
from pronunciation_coach.config import DEFAULT_MODEL_CACHE, MODEL_ID, MODEL_REVISION
from pronunciation_coach.ctc_scoring import ConstrainedCTCScore, constrained_ctc_score
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.phones import parse_phone_sequence
from pronunciation_coach.schemas import RecognitionResult


@dataclass(frozen=True, slots=True)
class PosteriorRecognition:
    alignment: ConstrainedCTCScore
    greedy_phones: tuple[str, ...]
    inference_ms: int
    device: str


class Wav2Vec2EspeakRecognizer:
    def __init__(
        self,
        *,
        cache_dir: Path = DEFAULT_MODEL_CACHE,
        device: str = "auto",
        local_files_only: bool = True,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
    ) -> None:
        self.cache_dir = cache_dir
        self.device = _resolve_device(device)
        self.model_id = model_id
        self.model_revision = model_revision
        try:
            self.processor: Any = AutoProcessor.from_pretrained(
                model_id,
                revision=model_revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.model: Any = AutoModelForCTC.from_pretrained(
                model_id,
                revision=model_revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except (OSError, ValueError) as error:
            raise CoachError(
                ErrorCode.MODEL_UNAVAILABLE,
                "Pinned phone model is unavailable; run pronunciation-coach-model prefetch",
            ) from error
        self.model.eval()
        self.model.to(self.device)

    def recognize(self, audio: NormalizedAudio) -> RecognitionResult:
        started = perf_counter()
        logits = self._infer_logits(audio)
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = self.processor.batch_decode(predicted_ids.cpu())[0]
        inference_ms = round((perf_counter() - started) * 1000)
        return RecognitionResult(
            phones=parse_phone_sequence(decoded),
            backend="wav2vec2-espeak-ctc",
            revision=self.model_revision,
            inference_ms=inference_ms,
            device=self.device,
            diagnostics=(f"device={self.device}",),
        )

    def score_expected(
        self,
        audio: NormalizedAudio,
        expected_phones: tuple[str, ...],
        *,
        competing_phones: frozenset[str] | None = None,
    ) -> PosteriorRecognition:
        return self.score_expected_variants(
            audio,
            (expected_phones,),
            competing_phones=competing_phones,
        )[0]

    def score_expected_variants(
        self,
        audio: NormalizedAudio,
        expected_variants: tuple[tuple[str, ...], ...],
        *,
        competing_phones: frozenset[str] | None = None,
    ) -> tuple[PosteriorRecognition, ...]:
        if not expected_variants:
            raise ValueError("expected_variants must not be empty")
        started = perf_counter()
        logits = self._infer_logits(audio)
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = self.processor.batch_decode(predicted_ids.cpu())[0]
        tokenizer = self.processor.tokenizer
        vocabulary = {
            str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()
        }
        blank_id = int(self.model.config.pad_token_id)
        excluded_ids = frozenset(int(token_id) for token_id in tokenizer.all_special_ids)
        if competing_phones is not None:
            missing = competing_phones - vocabulary.keys()
            if missing:
                missing_list = tuple(sorted(missing))
                raise ValueError(
                    f"competing phones are absent from the model vocabulary: {missing_list}"
                )
            if len(competing_phones) < 2:
                raise ValueError("competing_phones must contain at least two phones")
            excluded_ids |= frozenset(
                token_id for phone, token_id in vocabulary.items() if phone not in competing_phones
            )
        log_probabilities = torch.log_softmax(logits[0], dim=-1).cpu().numpy()
        alignments = tuple(
            constrained_ctc_score(
                log_probabilities,
                expected_phones,
                vocabulary,
                blank_id=blank_id,
                excluded_alternative_ids=excluded_ids,
            )
            for expected_phones in expected_variants
        )
        inference_ms = round((perf_counter() - started) * 1000)
        greedy_phones = parse_phone_sequence(decoded)
        return tuple(
            PosteriorRecognition(
                alignment=alignment,
                greedy_phones=greedy_phones,
                inference_ms=inference_ms,
                device=self.device,
            )
            for alignment in alignments
        )

    def _infer_logits(self, audio: NormalizedAudio) -> torch.Tensor:
        inputs = self.processor(
            audio.samples,
            sampling_rate=audio.metadata.analysis_sample_rate_hz,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        return logits


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return requested
    if requested == "mps":
        if torch.backends.mps.is_available():
            return requested
        raise CoachError(ErrorCode.MODEL_UNAVAILABLE, "Apple MPS is unavailable")
    if requested == "cuda":
        if torch.cuda.is_available():
            return requested
        raise CoachError(ErrorCode.MODEL_UNAVAILABLE, "CUDA is unavailable")
    if requested != "auto":
        raise CoachError(ErrorCode.MODEL_UNAVAILABLE, f"Unsupported device: {requested}")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
