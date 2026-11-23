"""Commands for freezing and running the Stage 1 feasibility benchmark."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pronunciation_coach.benchmarking import (
    file_sha256,
    load_competing_phone_inventory,
    load_heldout_manifest,
    load_manifest,
    load_policy,
    load_posterior_evaluation_policy,
    load_speechocean,
    prepare_calibration_manifest,
    prepare_heldout_manifests,
    prepare_manifest,
    run_benchmark,
    run_calibrated_word_heldout_benchmark,
    run_posterior_benchmark,
    run_posterior_heldout_benchmark,
    validate_heldout_pair,
    write_json,
)
from pronunciation_coach.config import (
    DEFAULT_DATASET_CACHE,
    DEFAULT_MODEL_CACHE,
    MODEL_SNAPSHOTS,
)
from pronunciation_coach.word_calibration import calibrate_word_errors, load_posterior_report

DEFAULT_MANIFEST = Path("benchmarks/manifests/development-v1.json")
DEFAULT_CALIBRATION_MANIFEST = Path("benchmarks/manifests/calibration-speakers-v1.json")
DEFAULT_POLICY = Path("benchmarks/decision_policy.v1.json")
DEFAULT_REPORT = Path("benchmarks/reports/stage1-v1.json")
DEFAULT_POSTERIOR_REPORT = Path("benchmarks/reports/stage1b-ctc-development.json")
DEFAULT_CALIBRATION_EVIDENCE = Path("benchmarks/reports/stage1c-calibration-evidence-v1.json")
DEFAULT_CALIBRATION_REPORT = Path("benchmarks/reports/stage1c-word-calibration-v1.json")
DEFAULT_ENGLISH_INVENTORY = Path("benchmarks/inventories/english-espeak-train-v1.json")
DEFAULT_HELDOUT_MANIFEST = Path("benchmarks/manifests/heldout-validation-v1.json")
DEFAULT_FINAL_MANIFEST = Path("benchmarks/manifests/final-test-v1.json")
DEFAULT_POSTERIOR_POLICY = Path("benchmarks/policies/ctc-posterior-v1.json")
DEFAULT_HELDOUT_REPORT = Path("benchmarks/reports/stage1b-ctc-heldout-v1.json")
DEFAULT_CALIBRATED_HELDOUT_REPORT = Path(
    "benchmarks/reports/stage1c-word-calibration-heldout-v1.json"
)


def _add_run_arguments(parser: argparse.ArgumentParser, *, output: Path) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)

    prepare_calibration = subparsers.add_parser("prepare-calibration-manifest")
    prepare_calibration.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_MANIFEST)
    prepare_calibration.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)

    prepare_heldout = subparsers.add_parser("prepare-heldout-manifests")
    prepare_heldout.add_argument("--validation-output", type=Path, default=DEFAULT_HELDOUT_MANIFEST)
    prepare_heldout.add_argument("--final-output", type=Path, default=DEFAULT_FINAL_MANIFEST)
    prepare_heldout.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)

    run = subparsers.add_parser("run")
    _add_run_arguments(run, output=DEFAULT_REPORT)

    posterior = subparsers.add_parser("run-ctc")
    _add_run_arguments(posterior, output=DEFAULT_POSTERIOR_REPORT)
    posterior.add_argument(
        "--competing-phone-inventory",
        type=Path,
        help=f"restrict alternatives to an inventory such as {DEFAULT_ENGLISH_INVENTORY}",
    )
    posterior.add_argument("--model", choices=tuple(MODEL_SNAPSHOTS), default="lv60")

    calibrate = subparsers.add_parser("calibrate-word-errors")
    calibrate.add_argument("--report", type=Path, default=DEFAULT_CALIBRATION_EVIDENCE)
    calibrate.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_REPORT)
    posterior.add_argument(
        "--include-speechocean-variant",
        action="store_true",
        help="score eSpeak and normalized SpeechOcean canonical sentence variants",
    )

    heldout = subparsers.add_parser("run-ctc-heldout")
    heldout.add_argument("--manifest", type=Path, default=DEFAULT_HELDOUT_MANIFEST)
    heldout.add_argument("--posterior-policy", type=Path, default=DEFAULT_POSTERIOR_POLICY)
    heldout.add_argument("--output", type=Path, default=DEFAULT_HELDOUT_REPORT)
    heldout.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)
    heldout.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    heldout.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    heldout.add_argument("--competing-phone-inventory", type=Path)
    heldout.add_argument("--include-speechocean-variant", action="store_true")

    calibrated_heldout = subparsers.add_parser("run-calibrated-heldout")
    calibrated_heldout.add_argument("--manifest", type=Path, default=DEFAULT_HELDOUT_MANIFEST)
    calibrated_heldout.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    calibrated_heldout.add_argument(
        "--competing-phone-inventory", type=Path, default=DEFAULT_ENGLISH_INVENTORY
    )
    calibrated_heldout.add_argument(
        "--output", type=Path, default=DEFAULT_CALIBRATED_HELDOUT_REPORT
    )
    calibrated_heldout.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)
    calibrated_heldout.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    calibrated_heldout.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare-manifest":
        dataset = load_speechocean("train", arguments.dataset_cache)
        manifest = prepare_manifest(dataset)
        write_json(arguments.output, manifest)
        print(manifest.model_dump_json(indent=2))
        return 0

    if arguments.command == "prepare-calibration-manifest":
        dataset = load_speechocean("train", arguments.dataset_cache)
        manifest = prepare_calibration_manifest(dataset)
        write_json(arguments.output, manifest)
        print(manifest.model_dump_json(indent=2))
        return 0

    if arguments.command == "prepare-heldout-manifests":
        dataset = load_speechocean("test", arguments.dataset_cache)
        validation, final = prepare_heldout_manifests(dataset)
        validate_heldout_pair(validation, final)
        write_json(arguments.validation_output, validation)
        write_json(arguments.final_output, final)
        summary = {
            "validation": {
                "path": str(arguments.validation_output),
                "speakers": validation.speaker_count,
                "utterances": len(validation.items),
                "sha256": file_sha256(arguments.validation_output),
            },
            "final": {
                "path": str(arguments.final_output),
                "speakers": final.speaker_count,
                "utterances": len(final.items),
                "sha256": file_sha256(arguments.final_output),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if arguments.command == "calibrate-word-errors":
        report = calibrate_word_errors(
            load_posterior_report(arguments.report),
            source_report_sha256=file_sha256(arguments.report),
        )
        write_json(arguments.output, report)
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "metrics": report["metrics"],
                    "report": str(arguments.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if arguments.command == "run-ctc-heldout":
        manifest = load_heldout_manifest(arguments.manifest)
        policy = load_posterior_evaluation_policy(arguments.posterior_policy)
        inventory_path = arguments.competing_phone_inventory
        inventory = (
            load_competing_phone_inventory(inventory_path) if inventory_path is not None else None
        )
        report = run_posterior_heldout_benchmark(
            manifest,
            policy,
            manifest_sha256=file_sha256(arguments.manifest),
            policy_sha256=file_sha256(arguments.posterior_policy),
            dataset_cache=arguments.dataset_cache,
            model_cache=arguments.model_cache,
            device=arguments.device,
            competing_phone_inventory=inventory,
            competing_phone_inventory_sha256=(
                file_sha256(inventory_path) if inventory_path is not None else None
            ),
            include_speechocean_variant=arguments.include_speechocean_variant,
        )
        write_json(arguments.output, report)
        summary = {
            "decision": report["decision"],
            "checks": report["checks"],
            "metrics": report["metrics"],
            "report": str(arguments.output),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if arguments.command == "run-calibrated-heldout":
        manifest = load_heldout_manifest(arguments.manifest)
        calibration_report = load_posterior_report(arguments.calibration_report)
        inventory = load_competing_phone_inventory(arguments.competing_phone_inventory)
        report = run_calibrated_word_heldout_benchmark(
            manifest,
            calibration_report,
            inventory,
            manifest_sha256=file_sha256(arguments.manifest),
            calibration_report_sha256=file_sha256(arguments.calibration_report),
            competing_phone_inventory_sha256=file_sha256(arguments.competing_phone_inventory),
            dataset_cache=arguments.dataset_cache,
            model_cache=arguments.model_cache,
            device=arguments.device,
        )
        write_json(arguments.output, report)
        summary = {
            "decision": report["decision"],
            "checks": report["checks"],
            "metrics": report["metrics"],
            "report": str(arguments.output),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    manifest = load_manifest(arguments.manifest)
    policy = load_policy(arguments.policy)
    if arguments.command == "run-ctc":
        inventory_path = arguments.competing_phone_inventory
        inventory = (
            load_competing_phone_inventory(inventory_path) if inventory_path is not None else None
        )
        model_id, model_revision = MODEL_SNAPSHOTS[arguments.model]
        report = run_posterior_benchmark(
            manifest,
            policy,
            dataset_cache=arguments.dataset_cache,
            model_cache=arguments.model_cache,
            device=arguments.device,
            competing_phone_inventory=inventory,
            competing_phone_inventory_sha256=(
                file_sha256(inventory_path) if inventory_path is not None else None
            ),
            include_speechocean_variant=arguments.include_speechocean_variant,
            model_id=model_id,
            model_revision=model_revision,
        )
    else:
        report = run_benchmark(
            manifest,
            policy,
            dataset_cache=arguments.dataset_cache,
            model_cache=arguments.model_cache,
            device=arguments.device,
        )
    write_json(arguments.output, report)
    summary = {
        "decision": report["decision"],
        "checks": report["checks"],
        "metrics": report["metrics"],
        "report": str(arguments.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
