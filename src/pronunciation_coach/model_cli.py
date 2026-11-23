# pyright: reportUnknownVariableType=false
"""One-time download command for the immutable Stage 1 model snapshot."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download

from pronunciation_coach.config import (
    DEFAULT_MODEL_CACHE,
    MODEL_SNAPSHOTS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach-model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prefetch = subparsers.add_parser("prefetch", help="download the pinned model snapshot")
    prefetch.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    prefetch.add_argument("--model", choices=tuple(MODEL_SNAPSHOTS), default="lv60")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    model_id, model_revision = MODEL_SNAPSHOTS[arguments.model]
    snapshot_path = snapshot_download(
        repo_id=model_id,
        revision=model_revision,
        cache_dir=arguments.cache_dir,
    )
    payload = {
        "model_id": model_id,
        "revision": model_revision,
        "snapshot_path": snapshot_path,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
