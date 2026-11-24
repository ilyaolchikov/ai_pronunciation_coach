"""Diagnostic real-audio CLI for the Stage 1 feasibility spike."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pronunciation_coach.audio import detect_speech_region, load_audio
from pronunciation_coach.config import DEFAULT_MODEL_CACHE
from pronunciation_coach.errors import CoachError
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.recognizers.wav2vec2_espeak import Wav2Vec2EspeakRecognizer
from pronunciation_coach.runtime import get_analyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach-audio")
    parser.add_argument("--prompt", required=True, help="prompt catalogue ID")
    parser.add_argument("--audio", required=True, type=Path, help="short WAV or FLAC file")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument(
        "--posterior",
        action="store_true",
        help="emit experimental constrained-CTC phone scores instead of the result contract",
    )
    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=-8.0,
        help="development-only log-margin threshold (default: -8.0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        prompt = get_prompt(arguments.prompt)
        audio = detect_speech_region(load_audio(arguments.audio))
        if arguments.posterior:
            recognizer = Wav2Vec2EspeakRecognizer(
                cache_dir=arguments.cache_dir,
                device=arguments.device,
                local_files_only=True,
            )
            expected_phones = tuple(
                phone for word in prompt.words for phone in word.phone_variants[0]
            )
            word_indexes = tuple(
                word_index
                for word_index, word in enumerate(prompt.words)
                for _ in word.phone_variants[0]
            )
            posterior = recognizer.score_expected(audio, expected_phones)
            phone_scores = [
                {
                    "phone": score.phone,
                    "word_index": word_indexes[score.target_index],
                    "word": prompt.words[word_indexes[score.target_index]].text,
                    "start_frame": score.start_frame,
                    "end_frame": score.end_frame,
                    "expected_mean_probability": score.expected_mean_probability,
                    "alternative_phone": score.alternative_phone,
                    "log_posterior_margin": score.log_posterior_margin,
                    "review": score.log_posterior_margin < arguments.posterior_threshold,
                }
                for score in posterior.alignment.phone_scores
            ]
            payload = {
                "experimental": True,
                "prompt_id": prompt.id,
                "threshold": arguments.posterior_threshold,
                "threshold_status": "development-selected; held-out validation required",
                "review": any(bool(score["review"]) for score in phone_scores),
                "mean_phone_margin": posterior.alignment.mean_phone_margin,
                "p10_phone_margin": posterior.alignment.p10_phone_margin,
                "greedy_phones": posterior.greedy_phones,
                "phone_scores": phone_scores,
                "limitations": [
                    "A low margin is acoustic evidence, not proof of mispronunciation.",
                    "The -8 threshold was selected on 50 development recordings.",
                ],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        result = get_analyzer(arguments.cache_dir, arguments.device).analyze(prompt, audio)
    except CoachError as error:
        payload = {"error": {"code": error.code.value, "message": str(error)}}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
