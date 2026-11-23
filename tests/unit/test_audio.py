# pyright: reportUnknownMemberType=false

from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from pronunciation_coach.audio import detect_speech_region, load_audio, normalize_audio
from pronunciation_coach.errors import CoachError, ErrorCode

pytestmark = pytest.mark.stage1


def test_normalizes_stereo_and_resamples_to_16khz() -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    stereo = np.column_stack((mono, mono * 0.5))

    result = normalize_audio(stereo, sample_rate)

    assert result.samples.dtype == np.float32
    assert result.samples.shape == (16_000,)
    assert result.metadata.duration_ms == 1000
    assert result.metadata.input_channels == 2
    assert result.metadata.input_sample_rate_hz == 48_000
    assert result.metadata.analysis_sample_rate_hz == 16_000


def test_reports_clipping_before_downmix() -> None:
    samples = np.ones(16_000, dtype=np.float32)

    result = normalize_audio(samples, 16_000)

    assert result.metadata.clipped_sample_ratio == 1.0


@pytest.mark.parametrize(
    ("sample_count", "expected_code"),
    [
        (3_999, ErrorCode.AUDIO_TOO_SHORT),
        (240_001, ErrorCode.AUDIO_TOO_LONG),
    ],
)
def test_rejects_duration_outside_bounds(sample_count: int, expected_code: ErrorCode) -> None:
    with pytest.raises(CoachError) as captured:
        normalize_audio(np.zeros(sample_count, dtype=np.float32), 16_000)

    assert captured.value.code is expected_code


def test_rejects_non_finite_samples() -> None:
    samples = np.zeros(16_000, dtype=np.float32)
    samples[0] = np.nan

    with pytest.raises(CoachError) as captured:
        normalize_audio(samples, 16_000)

    assert captured.value.code is ErrorCode.AUDIO_INVALID


def test_loads_wav_bytes_without_temporary_file() -> None:
    output = BytesIO()
    sf.write(output, np.zeros(8_000, dtype=np.float32), 16_000, format="WAV")

    result = load_audio(output.getvalue())

    assert result.metadata.duration_ms == 500
    assert result.samples.shape == (8_000,)


def test_rejects_undecodable_bytes() -> None:
    with pytest.raises(CoachError) as captured:
        load_audio(b"not audio")

    assert captured.value.code is ErrorCode.AUDIO_DECODE_FAILED


def test_detects_a_bounded_speech_region() -> None:
    samples = np.zeros(16_000, dtype=np.float32)
    time = np.arange(4_800, dtype=np.float32) / 16_000
    samples[3_200:8_000] = 0.2 * np.sin(2 * np.pi * 220 * time)

    result = detect_speech_region(normalize_audio(samples, 16_000))

    assert result.metadata.speech_region is not None
    assert 140 <= result.metadata.speech_region.start_ms <= 210
    assert 490 <= result.metadata.speech_region.end_ms <= 560


def test_silence_has_a_stable_error() -> None:
    with pytest.raises(CoachError) as captured:
        detect_speech_region(normalize_audio(np.zeros(8_000, dtype=np.float32), 16_000))

    assert captured.value.code is ErrorCode.AUDIO_SILENT
