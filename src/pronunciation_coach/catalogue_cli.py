"""Catalogue validation against pinned eSpeak-generated phones."""

import argparse
import json
from collections.abc import Sequence

from pronunciation_coach.phonemization import generation_metadata, validate_prompt_phones
from pronunciation_coach.prompts import load_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach-catalogue")
    parser.add_subparsers(dest="command", required=True).add_parser(
        "validate", help="compare checked-in phones with eSpeak"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    metadata = generation_metadata()
    results: list[dict[str, object]] = []
    mismatch_count = 0
    for prompt in load_prompts():
        mismatches = validate_prompt_phones(prompt)
        mismatch_count += len(mismatches)
        results.append(
            {
                "prompt_id": prompt.id,
                "mismatches": [
                    {"word_index": word_index, "generated": generated}
                    for word_index, generated in mismatches
                ],
            }
        )
    payload = {
        "generated_with": metadata.model_dump(mode="json"),
        "mismatch_count": mismatch_count,
        "prompts": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if mismatch_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
