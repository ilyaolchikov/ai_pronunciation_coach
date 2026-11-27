from pathlib import Path

import pytest

from pronunciation_coach.benchmark_cli import build_parser
from pronunciation_coach.benchmarking import file_sha256, load_heldout_manifest


def test_benchmark_cli_exposes_only_pronunciada_evaluations() -> None:
    parser = build_parser()

    assert (
        parser.parse_args(
            ["evaluate-performance", "--audio", "sample.wav", "--prompt", "thought-about-it"]
        ).command
        == "evaluate-performance"
    )
    assert parser.parse_args(["evaluate-corpora"]).command == "evaluate-corpora"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate-ui-smoke"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate-corpora", "--precision", "fp32"])
    with pytest.raises(SystemExit):
        parser.parse_args(["obsolete-command"])


def test_loads_the_pronunciada_speechocean_manifest() -> None:
    path = Path("benchmarks/manifests/final-test-v1.json")

    manifest = load_heldout_manifest(path)

    assert manifest.partition == "final-test"
    assert len(manifest.items) == 1_260
    assert len(file_sha256(path)) == 64


def test_rejects_an_incorrect_speaker_count(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"version":1,"dataset_id":"x","dataset_revision":"x","split":"test",'
        '"partition":"final-test","selection_seed":1,"selection_algorithm":"x",'
        '"speaker_count":2,"items":[{"utterance_id":"one","speaker":"speaker",'
        '"age_group":"adult","gender":"m"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="speaker_count"):
        load_heldout_manifest(path)
