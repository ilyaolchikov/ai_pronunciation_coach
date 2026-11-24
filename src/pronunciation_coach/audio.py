# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Bounded decoding and normalization for short speech recordings."""

from dataclasses import dataclass, replace
from io import BytesIO
from math import gcd
from pathlib import Path
from typing import TypeAlias

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from pronunciation_coach.config import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_DURATION_SECONDS,
    MAX_INPUT_CHANNELS,
    MAX_INPUT_SAMPLE_RATE_HZ,
    MIN_AUDIO_DURATION_SECONDS,
    MIN_INPUT_SAMPLE_RATE_HZ,
    MODEL_SAMPLE_RATE_HZ,
)
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.schemas import AudioMetadata, SpeechRegion

FloatAudio: TypeAlias = NDArray[np.float32]
AudioSource: TypeAlias = str | Path | bytes


@dataclass(frozen=True, slots=True)
class NormalizedAudio:
    samples: FloatAudio
    metadata: AudioMetadata


def detect_speech_region(audio: NormalizedAudio) -> NormalizedAudio:
    window = round(MODEL_SAMPLE_RATE_HZ * 0.02)
    hop = round(MODEL_SAMPLE_RATE_HZ * 0.01)
    root_mean_squares = np.asarray(
        [
            np.sqrt(np.mean(audio.samples[start : start + window] ** 2))
            for start in range(0, len(audio.samples) - window + 1, hop)
        ],
        dtype=np.float32,
    )
    peak = float(np.max(root_mean_squares)) if root_mean_squares.size else 0.0
    if peak < 0.01:
        raise CoachError(ErrorCode.AUDIO_SILENT, "Audio contains no detectable speech")

    active = np.flatnonzero(root_mean_squares >= max(0.01, peak * 0.1))
    padding = round(MODEL_SAMPLE_RATE_HZ * 0.05)
    start_sample = max(0, int(active[0]) * hop - padding)
    end_sample = min(len(audio.samples), int(active[-1]) * hop + window + padding)
    start_ms = round(start_sample / MODEL_SAMPLE_RATE_HZ * 1000)
    end_ms = round(end_sample / MODEL_SAMPLE_RATE_HZ * 1000)
    region = SpeechRegion(
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
    )
    return replace(
        audio,
        metadata=audio.metadata.model_copy(update={"speech_region": region}),
    )


def load_audio(source: AudioSource) -> NormalizedAudio:
    if isinstance(source, bytes):
        if len(source) > MAX_AUDIO_BYTES:
            raise CoachError(ErrorCode.AUDIO_FILE_TOO_LARGE, "Audio file exceeds 10 MiB")
        handle: str | BytesIO = BytesIO(source)
    else:
        path = Path(source)
        try:
            file_size = path.stat().st_size
        except OSError as error:
            raise CoachError(ErrorCode.AUDIO_DECODE_FAILED, "Audio file cannot be read") from error
        if file_size > MAX_AUDIO_BYTES:
            raise CoachError(ErrorCode.AUDIO_FILE_TOO_LARGE, "Audio file exceeds 10 MiB")
        handle = str(path)

    try:
        info = sf.info(handle)
    except (sf.LibsndfileError, TypeError, ValueError) as error:
        raise CoachError(
            ErrorCode.AUDIO_DECODE_FAILED, "Audio format could not be decoded"
        ) from error

    _validate_stream_info(info.samplerate, info.channels, info.frames)

    if isinstance(handle, BytesIO):
        handle.seek(0)
    try:
        samples, sample_rate = sf.read(handle, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, TypeError, ValueError) as error:
        raise CoachError(
            ErrorCode.AUDIO_DECODE_FAILED, "Audio format could not be decoded"
        ) from error

    return normalize_audio(samples, sample_rate)


def normalize_audio(samples: NDArray[np.generic], sample_rate: int) -> NormalizedAudio:
    array = np.asarray(samples)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise CoachError(ErrorCode.AUDIO_INVALID, "Audio must have one sample and channel axis")

    frame_count, channel_count = array.shape
    _validate_stream_info(sample_rate, channel_count, frame_count)
    float_samples = to_float_audio(array)
    if not np.all(np.isfinite(float_samples)):
        raise CoachError(ErrorCode.AUDIO_INVALID, "Audio contains NaN or infinite samples")

    clipped_ratio = float(np.mean(np.abs(float_samples) >= 0.999))
    mono = np.mean(float_samples, axis=1, dtype=np.float32)
    normalized = _resample(mono, sample_rate, MODEL_SAMPLE_RATE_HZ)
    normalized = np.ascontiguousarray(np.clip(normalized, -1.0, 1.0), dtype=np.float32)

    return NormalizedAudio(
        samples=normalized,
        metadata=AudioMetadata(
            duration_ms=round(frame_count / sample_rate * 1000),
            input_sample_rate_hz=sample_rate,
            analysis_sample_rate_hz=MODEL_SAMPLE_RATE_HZ,
            input_channels=channel_count,
            clipped_sample_ratio=clipped_ratio,
        ),
    )


def to_float_audio(samples: NDArray[np.generic]) -> FloatAudio:
    if np.issubdtype(samples.dtype, np.bool_):
        raise CoachError(ErrorCode.AUDIO_INVALID, "Boolean audio samples are unsupported")
    if np.issubdtype(samples.dtype, np.signedinteger):
        info = np.iinfo(samples.dtype.name)
        scale = float(max(abs(info.min), info.max))
        return np.asarray(samples, dtype=np.float32) / scale
    if np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype.name)
        midpoint = float(info.max + 1) / 2
        return (np.asarray(samples, dtype=np.float32) - midpoint) / midpoint
    return np.asarray(samples, dtype=np.float32)


def _validate_stream_info(sample_rate: int, channels: int, frames: int) -> None:
    if not MIN_INPUT_SAMPLE_RATE_HZ <= sample_rate <= MAX_INPUT_SAMPLE_RATE_HZ:
        raise CoachError(ErrorCode.AUDIO_INVALID, "Audio sample rate is unsupported")
    if not 1 <= channels <= MAX_INPUT_CHANNELS:
        raise CoachError(ErrorCode.AUDIO_INVALID, "Audio must be mono or stereo")

    duration = frames / sample_rate
    if duration < MIN_AUDIO_DURATION_SECONDS:
        raise CoachError(ErrorCode.AUDIO_TOO_SHORT, "Audio is shorter than 0.25 seconds")
    if duration > MAX_AUDIO_DURATION_SECONDS:
        raise CoachError(ErrorCode.AUDIO_TOO_LONG, "Audio is longer than 15 seconds")


def _resample(samples: FloatAudio, source_rate: int, target_rate: int) -> FloatAudio:
    if source_rate == target_rate:
        return samples.copy()
    divisor = gcd(source_rate, target_rate)
    result = resample_poly(samples, target_rate // divisor, source_rate // divisor)
    return np.asarray(result, dtype=np.float32)
