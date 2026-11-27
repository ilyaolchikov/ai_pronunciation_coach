"""Pronunciada benchmark commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pronunciation_coach.benchmarking import file_sha256, load_heldout_manifest
from pronunciation_coach.config import DEFAULT_DATASET_CACHE
from pronunciation_coach.l2arctic import load_l2arctic_manifest
from pronunciation_coach.performance_benchmark import measure_release_performance
from pronunciation_coach.pronunciada_analysis import (
    PRECISION,
    ZIPA_MODEL,
    ZIPA_REVISION,
)
from pronunciation_coach.pronunciada_benchmark import (
    create_scorer,
    evaluate_l2arctic,
    evaluate_speechocean_final,
)
from pronunciation_coach.runtime import get_analyzer

DEFAULT_FINAL_MANIFEST = Path("benchmarks/manifests/final-test-v1.json")
DEFAULT_L2ARCTIC_ROOT = Path("l2arctic_release_v5.0")
DEFAULT_L2ARCTIC_MANIFEST = Path("benchmarks/manifests/l2arctic-speakers-v1.json")
DEFAULT_CORPORA_REPORT = Path("benchmarks/reports/pronunciada-corpora-v1.json")
DEFAULT_SPEECHOCEAN_REPORT = Path("benchmark-data/pronunciada-speechocean-final-v1.json")
DEFAULT_L2ARCTIC_REPORT = Path("benchmark-data/pronunciada-l2arctic-v1.json")
DEFAULT_PERFORMANCE_REPORT = Path("benchmarks/reports/performance-v1.json")


def _add_scorer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pronunciada-cache", type=Path, default=Path(".cache/pronunciada"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpora = subparsers.add_parser("evaluate-corpora")
    corpora.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)
    corpora.add_argument("--speechocean-manifest", type=Path, default=DEFAULT_FINAL_MANIFEST)
    corpora.add_argument("--l2arctic-root", type=Path, default=DEFAULT_L2ARCTIC_ROOT)
    corpora.add_argument("--l2arctic-manifest", type=Path, default=DEFAULT_L2ARCTIC_MANIFEST)
    corpora.add_argument("--output", type=Path, default=DEFAULT_CORPORA_REPORT)
    corpora.add_argument("--speechocean-output", type=Path, default=DEFAULT_SPEECHOCEAN_REPORT)
    corpora.add_argument("--l2arctic-output", type=Path, default=DEFAULT_L2ARCTIC_REPORT)
    _add_scorer_arguments(corpora)

    performance = subparsers.add_parser("evaluate-performance")
    performance.add_argument("--audio", required=True, type=Path)
    performance.add_argument("--prompt", required=True)
    performance.add_argument("--pronunciada-cache", type=Path, default=Path(".cache/pronunciada"))
    performance.add_argument("--threads", type=int, default=4)
    performance.add_argument("--warm-runs", type=int, default=30)
    performance.add_argument("--output", type=Path, default=DEFAULT_PERFORMANCE_REPORT)
    performance.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "evaluate-performance":
        get_analyzer.cache_clear()
        report = measure_release_performance(
            lambda: get_analyzer(arguments.pronunciada_cache, threads=arguments.threads),
            audio_path=arguments.audio,
            prompt_id=arguments.prompt,
            cache_dir=arguments.pronunciada_cache,
            warm_runs=arguments.warm_runs,
        )
        report["runtime"] = _runtime_provenance()
        _write_json(arguments.output, report, arguments.overwrite)
        print(json.dumps({"metrics": report["metrics"], "report": str(arguments.output)}, indent=2))
        return 0
    scorer = create_scorer(
        arguments.pronunciada_cache,
        precision=PRECISION,
        threads=arguments.threads,
    )
    speechocean = evaluate_speechocean_final(
        scorer,
        load_heldout_manifest(arguments.speechocean_manifest),
        dataset_cache=arguments.dataset_cache,
    )
    l2arctic = evaluate_l2arctic(
        scorer,
        arguments.l2arctic_root,
        load_l2arctic_manifest(arguments.l2arctic_manifest),
    )
    _write_json(arguments.speechocean_output, speechocean, arguments.overwrite)
    _write_json(arguments.l2arctic_output, l2arctic, arguments.overwrite)
    report = {
        "report_version": 1,
        "experiment": "pronunciada-all-human-corpora-v1",
        "runtime": _runtime_provenance(),
        "dataset_manifests": {
            "speechocean": {
                "path": str(arguments.speechocean_manifest),
                "sha256": file_sha256(arguments.speechocean_manifest),
            },
            "l2arctic": {
                "path": str(arguments.l2arctic_manifest),
                "sha256": file_sha256(arguments.l2arctic_manifest),
            },
        },
        "speechocean": {
            "metrics": speechocean["metrics"],
            "detail_report": str(arguments.speechocean_output),
        },
        "l2arctic": {
            "metrics": l2arctic["metrics"],
            "detail_report": str(arguments.l2arctic_output),
        },
    }
    _write_json(arguments.output, report, arguments.overwrite)
    print(
        json.dumps(
            {"metrics": report.get("metrics", report), "report": str(arguments.output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite experiment artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runtime_provenance() -> dict[str, str]:
    return {
        "package": f"pronunciada=={version('pronunciada')}",
        "language": "en",
        "acoustic_model": ZIPA_MODEL,
        "revision": ZIPA_REVISION,
        "precision": PRECISION,
    }


if __name__ == "__main__":
    raise SystemExit(main())
