"""Analyze one recording with Pronunciada."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pronunciation_coach.audio import detect_speech_region, load_audio
from pronunciation_coach.config import DEFAULT_PRONUNCIADA_CACHE
from pronunciation_coach.errors import CoachError
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.runtime import get_analyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach-audio")
    parser.add_argument("--prompt", required=True, help="prompt catalogue ID")
    parser.add_argument("--audio", required=True, type=Path, help="WAV, FLAC, or AIFF file")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_PRONUNCIADA_CACHE)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        prompt = get_prompt(arguments.prompt)
        audio = detect_speech_region(load_audio(arguments.audio))
        result = get_analyzer(arguments.cache_dir, threads=arguments.threads).analyze(prompt, audio)
    except CoachError as error:
        payload = {"error": {"code": error.code.value, "message": str(error)}}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
