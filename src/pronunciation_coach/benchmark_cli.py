"""Commands for freezing and running the Stage 1 feasibility benchmark."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
from pronunciation_coach.intentional_benchmark import (
    load_intentional_manifest,
    run_intentional_benchmark,
)
from pronunciation_coach.l2arctic import (
    load_l2arctic_manifest,
    prepare_l2arctic_manifest,
    validate_l2arctic,
)
from pronunciation_coach.l2arctic_experiment import (
    calibrate_l2arctic_phone_errors,
    evaluate_frozen_l2arctic_candidate,
    evaluate_l2arctic_speechocean_regression,
    load_json_report,
    run_l2arctic_evidence,
)
from pronunciation_coach.phone_policy import load_phone_policy
from pronunciation_coach.runtime import get_analyzer
from pronunciation_coach.schemas import ContractModel
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
DEFAULT_INTENTIONAL_MANIFEST = Path("benchmarks/intentional-errors/manifest.v1.json")
DEFAULT_INTENTIONAL_REPORT = Path("benchmarks/reports/intentional-errors-v1.json")
DEFAULT_L2ARCTIC_ROOT = Path("l2arctic_release_v5.0")
DEFAULT_L2ARCTIC_MANIFEST = Path("benchmarks/manifests/l2arctic-speakers-v1.json")
DEFAULT_L2ARCTIC_INTEGRITY = Path("benchmarks/reports/l2arctic-integrity-v1.json")
DEFAULT_L2ARCTIC_TRAIN_EVIDENCE = Path("benchmark-data/l2arctic-train-evidence-v1.json")
DEFAULT_L2ARCTIC_DEVELOPMENT_EVIDENCE = Path("benchmark-data/l2arctic-development-evidence-v1.json")
DEFAULT_L2ARCTIC_FINAL_EVIDENCE = Path("benchmark-data/l2arctic-final-evidence-v1.json")
DEFAULT_L2ARCTIC_CALIBRATION = Path("benchmarks/reports/l2arctic-phone-calibration-v1.json")
DEFAULT_L2ARCTIC_FINAL_REPORT = Path("benchmarks/reports/l2arctic-phone-final-v1.json")
DEFAULT_L2ARCTIC_SPEECHOCEAN_REGRESSION = Path(
    "benchmarks/reports/l2arctic-speechocean-regression-v1.json"
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

    intentional = subparsers.add_parser("run-intentional-errors")
    intentional.add_argument("--manifest", type=Path, default=DEFAULT_INTENTIONAL_MANIFEST)
    intentional.add_argument("--output", type=Path, default=DEFAULT_INTENTIONAL_REPORT)
    intentional.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    intentional.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")

    l2_prepare = subparsers.add_parser("prepare-l2arctic")
    l2_prepare.add_argument("--dataset-root", type=Path, default=DEFAULT_L2ARCTIC_ROOT)
    l2_prepare.add_argument("--manifest-output", type=Path, default=DEFAULT_L2ARCTIC_MANIFEST)
    l2_prepare.add_argument("--integrity-output", type=Path, default=DEFAULT_L2ARCTIC_INTEGRITY)
    l2_prepare.add_argument("--overwrite", action="store_true")

    l2_evidence = subparsers.add_parser("run-l2arctic-evidence")
    l2_evidence.add_argument("--partition", choices=("train", "development"), required=True)
    l2_evidence.add_argument("--dataset-root", type=Path, default=DEFAULT_L2ARCTIC_ROOT)
    l2_evidence.add_argument("--manifest", type=Path, default=DEFAULT_L2ARCTIC_MANIFEST)
    l2_evidence.add_argument(
        "--competing-phone-inventory", type=Path, default=DEFAULT_ENGLISH_INVENTORY
    )
    l2_evidence.add_argument("--output", type=Path)
    l2_evidence.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    l2_evidence.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    l2_evidence.add_argument("--overwrite", action="store_true")

    l2_calibrate = subparsers.add_parser("calibrate-l2arctic")
    l2_calibrate.add_argument("--train-report", type=Path, default=DEFAULT_L2ARCTIC_TRAIN_EVIDENCE)
    l2_calibrate.add_argument(
        "--development-report", type=Path, default=DEFAULT_L2ARCTIC_DEVELOPMENT_EVIDENCE
    )
    l2_calibrate.add_argument("--output", type=Path, default=DEFAULT_L2ARCTIC_CALIBRATION)
    l2_calibrate.add_argument("--overwrite", action="store_true")

    l2_final = subparsers.add_parser("evaluate-l2arctic-final")
    l2_final.add_argument("--dataset-root", type=Path, default=DEFAULT_L2ARCTIC_ROOT)
    l2_final.add_argument("--manifest", type=Path, default=DEFAULT_L2ARCTIC_MANIFEST)
    l2_final.add_argument(
        "--competing-phone-inventory", type=Path, default=DEFAULT_ENGLISH_INVENTORY
    )
    l2_final.add_argument("--calibration-report", type=Path, default=DEFAULT_L2ARCTIC_CALIBRATION)
    l2_final.add_argument("--evidence-output", type=Path, default=DEFAULT_L2ARCTIC_FINAL_EVIDENCE)
    l2_final.add_argument("--output", type=Path, default=DEFAULT_L2ARCTIC_FINAL_REPORT)
    l2_final.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    l2_final.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    l2_final.add_argument("--overwrite", action="store_true")

    l2_regression = subparsers.add_parser("evaluate-l2arctic-speechocean")
    l2_regression.add_argument("--evidence-report", type=Path, default=DEFAULT_CALIBRATION_EVIDENCE)
    l2_regression.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_L2ARCTIC_CALIBRATION
    )
    l2_regression.add_argument(
        "--output", type=Path, default=DEFAULT_L2ARCTIC_SPEECHOCEAN_REGRESSION
    )
    l2_regression.add_argument("--overwrite", action="store_true")
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

    if arguments.command == "run-intentional-errors":
        manifest = load_intentional_manifest(arguments.manifest)
        report = run_intentional_benchmark(
            manifest,
            arguments.manifest.parent,
            lambda: get_analyzer(arguments.model_cache, arguments.device),
            manifest_sha256=file_sha256(arguments.manifest),
        )
        write_json(arguments.output, report)
        print(
            json.dumps(
                {"metrics": report["metrics"], "report": str(arguments.output)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if arguments.command == "prepare-l2arctic":
        manifest = prepare_l2arctic_manifest()
        integrity = validate_l2arctic(arguments.dataset_root)
        _write_experiment_json(arguments.manifest_output, manifest, arguments.overwrite)
        _write_experiment_json(arguments.integrity_output, integrity, arguments.overwrite)
        print(
            json.dumps(
                {
                    "manifest": str(arguments.manifest_output),
                    "integrity_report": str(arguments.integrity_output),
                    "checks": integrity["checks"],
                    "metrics": integrity["metrics"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if arguments.command == "run-l2arctic-evidence":
        manifest = load_l2arctic_manifest(arguments.manifest)
        inventory = load_competing_phone_inventory(arguments.competing_phone_inventory)
        output = arguments.output or (
            DEFAULT_L2ARCTIC_TRAIN_EVIDENCE
            if arguments.partition == "train"
            else DEFAULT_L2ARCTIC_DEVELOPMENT_EVIDENCE
        )
        report = run_l2arctic_evidence(
            arguments.dataset_root,
            manifest,
            arguments.partition,
            inventory,
            manifest_sha256=file_sha256(arguments.manifest),
            inventory_sha256=file_sha256(arguments.competing_phone_inventory),
            model_cache=arguments.model_cache,
            device=arguments.device,
        )
        _write_experiment_json(output, report, arguments.overwrite)
        print(json.dumps({"metrics": report["metrics"], "report": str(output)}, indent=2))
        return 0

    if arguments.command == "calibrate-l2arctic":
        report = calibrate_l2arctic_phone_errors(
            load_json_report(arguments.train_report),
            load_json_report(arguments.development_report),
            training_report_sha256=file_sha256(arguments.train_report),
            development_report_sha256=file_sha256(arguments.development_report),
        )
        _write_experiment_json(arguments.output, report, arguments.overwrite)
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

    if arguments.command == "evaluate-l2arctic-final":
        calibration = load_json_report(arguments.calibration_report)
        if calibration["decision"] != "development_candidate":
            raise ValueError("Refusing to consume final-test speakers for a stopped candidate")
        manifest = load_l2arctic_manifest(arguments.manifest)
        inventory = load_competing_phone_inventory(arguments.competing_phone_inventory)
        evidence = run_l2arctic_evidence(
            arguments.dataset_root,
            manifest,
            "final-test",
            inventory,
            manifest_sha256=file_sha256(arguments.manifest),
            inventory_sha256=file_sha256(arguments.competing_phone_inventory),
            model_cache=arguments.model_cache,
            device=arguments.device,
            allow_final_test=True,
        )
        _write_experiment_json(arguments.evidence_output, evidence, arguments.overwrite)
        report = evaluate_frozen_l2arctic_candidate(
            evidence,
            calibration,
            evidence_report_sha256=file_sha256(arguments.evidence_output),
            calibration_report_sha256=file_sha256(arguments.calibration_report),
        )
        _write_experiment_json(arguments.output, report, arguments.overwrite)
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

    if arguments.command == "evaluate-l2arctic-speechocean":
        phone_policy = load_phone_policy()
        report = evaluate_l2arctic_speechocean_regression(
            load_json_report(arguments.evidence_report),
            load_json_report(arguments.calibration_report),
            warning_probability_threshold=phone_policy.warning_probability_threshold,
            evidence_report_sha256=file_sha256(arguments.evidence_report),
            calibration_report_sha256=file_sha256(arguments.calibration_report),
        )
        _write_experiment_json(arguments.output, report, arguments.overwrite)
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


def _write_experiment_json(
    path: Path, value: ContractModel | dict[str, Any], overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite experiment artifact: {path}")
    write_json(path, value)


if __name__ == "__main__":
    raise SystemExit(main())
