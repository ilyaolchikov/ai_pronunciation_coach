"""Shared inputs for Pronunciada corpus benchmarks."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from pronunciation_coach.config import (
    DEFAULT_DATASET_CACHE,
    SPEECHOCEAN_DATASET_ID,
    SPEECHOCEAN_REVISION,
)
from pronunciation_coach.schemas import ContractModel


class HeldoutManifestItem(ContractModel):
    utterance_id: str
    speaker: str
    age_group: Literal["child", "adult"]
    gender: Literal["f", "m"]


class HeldoutManifest(ContractModel):
    version: int = 1
    dataset_id: str
    dataset_revision: str
    split: Literal["test"]
    partition: Literal["final-test"]
    selection_seed: int
    selection_algorithm: str
    speaker_count: int = Field(gt=0)
    items: tuple[HeldoutManifestItem, ...] = Field(min_length=1)


def load_heldout_manifest(path: Path) -> HeldoutManifest:
    manifest = HeldoutManifest.model_validate_json(path.read_text(encoding="utf-8"))
    speakers = {item.speaker for item in manifest.items}
    utterances = {item.utterance_id for item in manifest.items}
    if len(speakers) != manifest.speaker_count:
        raise ValueError("held-out speaker_count does not match its items")
    if len(utterances) != len(manifest.items):
        raise ValueError("held-out manifest contains duplicate utterance IDs")
    return manifest


def load_speechocean(split: str, cache_dir: Path = DEFAULT_DATASET_CACHE) -> Any:
    """Load the pinned SpeechOcean dataset only when the benchmark command needs it."""
    try:
        datasets: Any = import_module("datasets")
    except ImportError as error:
        raise RuntimeError("Install the 'benchmark' extra to load SpeechOcean") from error
    dataset = datasets.load_dataset(
        SPEECHOCEAN_DATASET_ID,
        revision=SPEECHOCEAN_REVISION,
        split=split,
        cache_dir=str(cache_dir),
    )
    return dataset.cast_column("audio", datasets.Audio(decode=False))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
