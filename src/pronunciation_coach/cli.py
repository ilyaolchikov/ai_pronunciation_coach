"""Command-line adapter for the deterministic Stage 0 pipeline."""

import argparse
import json
import sys
from collections.abc import Sequence

from pronunciation_coach.analysis import analyze
from pronunciation_coach.errors import CoachError
from pronunciation_coach.phones import parse_phone_sequence
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.recognizers.fake import FakePhoneRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="analyze a configured fake sequence")
    analyze_parser.add_argument("--prompt", required=True, help="prompt catalogue ID")
    analyze_parser.add_argument(
        "--fake-observed",
        required=True,
        help="space-separated observed phones; use an empty string for no phones",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        prompt = get_prompt(arguments.prompt)
        observed = parse_phone_sequence(arguments.fake_observed)
        result = analyze(prompt, FakePhoneRecognizer(observed))
    except CoachError as error:
        payload = {"error": {"code": error.code.value, "message": str(error)}}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2
    except ValueError as error:
        payload = {"error": {"code": "recognition_invalid", "message": str(error)}}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
