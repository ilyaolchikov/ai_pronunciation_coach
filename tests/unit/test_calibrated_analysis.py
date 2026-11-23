# pyright: reportPrivateUsage=false

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from pytest import CaptureFixture, MonkeyPatch

from pronunciation_coach.audio import NormalizedAudio, normalize_audio
from pronunciation_coach.audio_cli import main
from pronunciation_coach.calibrated_analysis import (
    CalibratedAnalyzer,
    WordPolicy,
    load_word_policy,
)
from pronunciation_coach.ctc_scoring import ConstrainedCTCScore, PhonePosteriorScore
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.recognizers.wav2vec2_espeak import PosteriorRecognition, _resolve_device
from pronunciation_coach.schemas import WordStatus


def _policy() -> WordPolicy:
    return WordPolicy(
        id="test-policy",
        version=1,
        model_id="test-model",
        model_revision="test-revision",
        calibration_report_sha256="0" * 64,
        candidate_model={
            "version": 1,
            "method": "l2-logistic-regression",
            "continuous_feature_names": [
                "minimum_phone_margin",
                "mean_phone_margin",
                "p10_phone_margin",
                "mean_expected_log_probability",
                "maximum_alternative_log_probability",
                "log_phone_count",
                "log_frame_count",
            ],
            "continuous_means": [0.0] * 7,
            "continuous_scales": [1.0] * 7,
            "phone_categories": ["aɪ", "<other>"],
            "intercept": 0.0,
            "coefficients": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "word_probability_threshold": 0.5,
        },
    )


class FakePosteriorRecognizer:
    model_id = "test-model"
    model_revision = "test-revision"

    def __init__(self) -> None:
        self.calls = 0

    def score_expected_variants(
        self,
        audio: NormalizedAudio,
        expected_variants: tuple[tuple[str, ...], ...],
        *,
        competing_phones: frozenset[str] | None = None,
    ) -> tuple[PosteriorRecognition, ...]:
        self.calls += 1
        assert audio.metadata.analysis_sample_rate_hz == 16_000
        assert competing_phones == frozenset({"aɪ"})
        results: list[PosteriorRecognition] = []
        for phones in expected_variants:
            scores = tuple(
                PhonePosteriorScore(
                    target_index=index,
                    phone=phone,
                    start_frame=index,
                    end_frame=index + 1,
                    expected_mean_log_probability=-4.0 if index == 0 else 4.0,
                    expected_mean_probability=0.5,
                    alternative_phone="x",
                    alternative_mean_log_probability=0.0,
                    log_posterior_margin=-4.0 if index == 0 else 4.0,
                )
                for index, phone in enumerate(phones)
            )
            results.append(
                PosteriorRecognition(
                    alignment=ConstrainedCTCScore(
                        phones=phones,
                        phone_scores=scores,
                        path_mean_log_probability=-1.0,
                    ),
                    greedy_phones=phones,
                    inference_ms=12,
                    device="cpu",
                )
            )
        return tuple(results)


def _speech_audio() -> NormalizedAudio:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    return normalize_audio(0.2 * np.sin(2 * np.pi * 220 * time), 16_000)


def test_calibrated_analyzer_returns_conservative_word_probabilities() -> None:
    recognizer = FakePosteriorRecognizer()
    analyzer = CalibratedAnalyzer(recognizer, _policy())

    result = analyzer.analyze(get_prompt("thought-about-it"), _speech_audio())

    assert recognizer.calls == 1
    assert result.policy is not None
    assert result.policy.word_probability_threshold == 0.5
    assert result.audio is not None and result.audio.speech_region is not None
    assert result.words[0].status is WordStatus.REVIEW
    assert result.words[0].review_probability is not None
    assert result.words[0].review_probability > 0.5
    assert {word.status for word in result.words[1:]} == {WordStatus.NO_PROBLEM_DETECTED}
    assert all(not word.evidence and not word.tips for word in result.words)


def test_silent_audio_returns_no_partial_judgment() -> None:
    recognizer = FakePosteriorRecognizer()
    analyzer = CalibratedAnalyzer(recognizer, _policy())

    with pytest.raises(CoachError) as captured:
        analyzer.analyze(
            get_prompt("thought-about-it"),
            normalize_audio(np.zeros(8_000, dtype=np.float32), 16_000),
        )

    assert captured.value.code is ErrorCode.AUDIO_SILENT
    assert recognizer.calls == 0


def test_bundled_policy_matches_the_frozen_calibration() -> None:
    policy = load_word_policy()

    assert policy.threshold == 0.3
    assert policy.calibration_report_sha256 == (
        "46c78016ac62c1af5832e0a72bc344e5bb1db83e1ced6ad088c7e9c4581a4025"
    )
    assert len(policy.competing_phones) == 57
    source: dict[str, Any] = json.loads(
        Path("benchmarks/reports/stage1c-word-calibration-v1.json").read_text(encoding="utf-8")
    )
    for key, value in policy.candidate_model.items():
        assert value == source["candidate_model"][key]


def test_unavailable_explicit_device_has_a_stable_error(monkeypatch: MonkeyPatch) -> None:
    def unavailable() -> bool:
        return False

    monkeypatch.setattr("torch.cuda.is_available", unavailable)

    with pytest.raises(CoachError) as captured:
        _resolve_device("cuda")

    assert captured.value.code is ErrorCode.MODEL_UNAVAILABLE


def test_audio_cli_emits_stage2_json(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    path = tmp_path / "speech.wav"
    sf.write(path, _speech_audio().samples, 16_000)
    analyzer = CalibratedAnalyzer(FakePosteriorRecognizer(), _policy())

    def fake_get_analyzer(_cache: Path, _device: str) -> CalibratedAnalyzer:
        return analyzer

    monkeypatch.setattr("pronunciation_coach.audio_cli._get_analyzer", fake_get_analyzer)

    exit_code = main(["--prompt", "thought-about-it", "--audio", str(path)])
    output: dict[str, Any] = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["schema_version"] == "1.2"
    assert output["policy"]["id"] == "test-policy"
    assert output["words"][0]["status"] == "review"
