# pyright: reportPrivateUsage=false

import os
from pathlib import Path

import numpy as np
import pytest

from pronunciation_coach.audio import normalize_audio
from pronunciation_coach.config import MODEL_REVISION
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.recognizers.wav2vec2_espeak import Wav2Vec2EspeakRecognizer
from pronunciation_coach.runtime import get_analyzer

pytestmark = [
    pytest.mark.stage1,
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 to use the pinned 1.2 GB model cache",
    ),
]


def test_cached_model_runs_offline_on_bounded_audio() -> None:
    audio = normalize_audio(np.zeros(8_000, dtype=np.float32), 16_000)
    recognizer = Wav2Vec2EspeakRecognizer(device="cpu", local_files_only=True)

    result = recognizer.recognize(audio)

    assert result.revision == MODEL_REVISION
    assert result.backend == "wav2vec2-espeak-ctc"
    assert result.device == "cpu"


def test_cached_model_produces_constrained_posterior_scores() -> None:
    audio = normalize_audio(np.zeros(8_000, dtype=np.float32), 16_000)
    recognizer = Wav2Vec2EspeakRecognizer(device="cpu", local_files_only=True)

    result = recognizer.score_expected(audio, ("a",))

    assert result.alignment.phones == ("a",)
    assert len(result.alignment.phone_scores) == 1
    assert result.alignment.phone_scores[0].alternative_phone != "a"
    assert result.device == "cpu"


def test_cached_model_restricts_posterior_competitors() -> None:
    audio = normalize_audio(np.zeros(8_000, dtype=np.float32), 16_000)
    recognizer = Wav2Vec2EspeakRecognizer(device="cpu", local_files_only=True)

    results = recognizer.score_expected_variants(
        audio,
        (("a",), ("b",)),
        competing_phones=frozenset({"a", "b"}),
    )

    assert len(results) == 2
    assert results[0].alignment.phone_scores[0].alternative_phone == "b"
    assert results[1].alignment.phone_scores[0].alternative_phone == "a"
    assert results[0].greedy_phones == results[1].greedy_phones


def test_cached_stage2_analyzer_runs_offline_and_is_reused() -> None:
    get_analyzer.cache_clear()
    analyzer = get_analyzer(Path(".cache/huggingface"), "cpu")
    assert analyzer is get_analyzer(Path(".cache/huggingface"), "cpu")
    time = np.arange(32_000, dtype=np.float32) / 16_000
    audio = normalize_audio(0.2 * np.sin(2 * np.pi * 220 * time), 16_000)

    result = analyzer.analyze(get_prompt("thought-about-it"), audio)

    assert result.policy is not None
    assert result.policy.word_probability_threshold == 0.3
    assert len(result.words) == 4
