"""Validated, ZIP-backed access to the L2-ARCTIC read-speech annotations."""

import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast
from zipfile import BadZipFile, ZipFile

from pydantic import Field, model_validator

from pronunciation_coach.phone_mapping import map_arpabet_phone
from pronunciation_coach.schemas import ContractModel

L2ARCTIC_DATASET_ID = "l2-arctic-v5.0"
L2ARCTIC_LICENSE = "CC-BY-NC-4.0"
L2ARCTIC_SPLIT_SEED = 762
L2ARCTIC_EXPECTED_ANNOTATIONS = 3_599

ErrorType = Literal["correct", "substitution", "deletion", "addition"]
Partition = Literal["train", "development", "final-test"]


@dataclass(frozen=True, slots=True)
class Speaker:
    id: str
    gender: Literal["f", "m"]
    first_language: str


SPEAKERS = (
    Speaker("ABA", "m", "Arabic"),
    Speaker("SKA", "f", "Arabic"),
    Speaker("YBAA", "m", "Arabic"),
    Speaker("ZHAA", "f", "Arabic"),
    Speaker("BWC", "m", "Mandarin"),
    Speaker("LXC", "f", "Mandarin"),
    Speaker("NCC", "f", "Mandarin"),
    Speaker("TXHC", "m", "Mandarin"),
    Speaker("ASI", "m", "Hindi"),
    Speaker("RRBI", "m", "Hindi"),
    Speaker("SVBI", "f", "Hindi"),
    Speaker("TNI", "f", "Hindi"),
    Speaker("HJK", "f", "Korean"),
    Speaker("HKK", "m", "Korean"),
    Speaker("YDCK", "f", "Korean"),
    Speaker("YKWK", "m", "Korean"),
    Speaker("EBVS", "m", "Spanish"),
    Speaker("ERMS", "m", "Spanish"),
    Speaker("MBMPS", "f", "Spanish"),
    Speaker("NJS", "f", "Spanish"),
    Speaker("HQTV", "m", "Vietnamese"),
    Speaker("PNV", "f", "Vietnamese"),
    Speaker("THV", "f", "Vietnamese"),
    Speaker("TLV", "m", "Vietnamese"),
)
SPEAKER_BY_ID = {speaker.id: speaker for speaker in SPEAKERS}


class L2ArcticSplitManifest(ContractModel):
    version: int = 1
    dataset_id: Literal["l2-arctic-v5.0"] = L2ARCTIC_DATASET_ID
    license: Literal["CC-BY-NC-4.0"] = L2ARCTIC_LICENSE
    selection_seed: int = L2ARCTIC_SPLIT_SEED
    selection_algorithm: Literal["two-train-one-development-one-final-per-l1-v1"] = (
        "two-train-one-development-one-final-per-l1-v1"
    )
    train_speakers: tuple[str, ...] = Field(min_length=12, max_length=12)
    development_speakers: tuple[str, ...] = Field(min_length=6, max_length=6)
    final_test_speakers: tuple[str, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_partitions(self) -> Self:
        partitions = (self.train_speakers, self.development_speakers, self.final_test_speakers)
        flattened = tuple(speaker for partition in partitions for speaker in partition)
        if len(set(flattened)) != len(flattened):
            raise ValueError("L2-ARCTIC speaker partitions overlap")
        if set(flattened) != set(SPEAKER_BY_ID):
            raise ValueError("L2-ARCTIC manifest must partition all known speakers")
        for partition in partitions:
            languages = Counter(SPEAKER_BY_ID[speaker].first_language for speaker in partition)
            expected = 2 if len(partition) == 12 else 1
            if set(languages.values()) != {expected}:
                raise ValueError("L2-ARCTIC partition is not balanced by first language")
        for partition in (self.development_speakers, self.final_test_speakers):
            genders = Counter(SPEAKER_BY_ID[speaker].gender for speaker in partition)
            if genders != {"f": 3, "m": 3}:
                raise ValueError("L2-ARCTIC evaluation partition is not gender balanced")
        return self

    def speakers_for(self, partition: Partition) -> tuple[str, ...]:
        if partition == "train":
            return self.train_speakers
        if partition == "development":
            return self.development_speakers
        return self.final_test_speakers


@dataclass(frozen=True, slots=True)
class Interval:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class AnnotatedPhone:
    start: float
    end: float
    word_index: int | None
    expected_arpabet: str | None
    perceived_arpabet: str | None
    error_type: ErrorType
    deviation: bool

    @property
    def expected_ipa(self) -> str | None:
        return _map_l2_phone(self.expected_arpabet)

    @property
    def perceived_ipa(self) -> str | None:
        return _map_l2_phone(self.perceived_arpabet)


@dataclass(frozen=True, slots=True)
class AnnotatedUtterance:
    speaker: Speaker
    utterance_id: str
    transcript: str
    words: tuple[str, ...]
    phones: tuple[AnnotatedPhone, ...]
    audio_member: str
    audio_bytes: bytes | None = None


ITEM_PATTERN = re.compile(r"^item \[\d+\]:$")
INTERVAL_PATTERN = re.compile(r"^intervals \[\d+\]:$")
ASSIGNMENT_PATTERN = re.compile(r"^(?P<key>name|xmin|xmax|text) = (?P<value>.*)$")


def prepare_l2arctic_manifest(seed: int = L2ARCTIC_SPLIT_SEED) -> L2ArcticSplitManifest:
    by_language: dict[str, list[Speaker]] = {}
    for speaker in SPEAKERS:
        by_language.setdefault(speaker.first_language, []).append(speaker)

    training: list[str] = []
    development: list[str] = []
    final_test: list[str] = []
    for language_index, language in enumerate(sorted(by_language)):
        group = by_language[language]
        females = sorted(
            (speaker for speaker in group if speaker.gender == "f"),
            key=lambda speaker: _speaker_hash(speaker.id, seed),
        )
        males = sorted(
            (speaker for speaker in group if speaker.gender == "m"),
            key=lambda speaker: _speaker_hash(speaker.id, seed),
        )
        if len(females) != 2 or len(males) != 2:
            raise ValueError(f"Expected two female and two male speakers for {language}")
        training.extend((females[0].id, males[0].id))
        if language_index % 2 == 0:
            development.append(females[1].id)
            final_test.append(males[1].id)
        else:
            development.append(males[1].id)
            final_test.append(females[1].id)

    return L2ArcticSplitManifest(
        train_speakers=tuple(sorted(training)),
        development_speakers=tuple(sorted(development)),
        final_test_speakers=tuple(sorted(final_test)),
    )


def load_l2arctic_manifest(path: Path) -> L2ArcticSplitManifest:
    return L2ArcticSplitManifest.model_validate_json(path.read_text(encoding="utf-8"))


def iter_l2arctic_utterances(
    dataset_root: Path,
    speakers: tuple[str, ...],
    *,
    include_audio: bool,
) -> Iterator[AnnotatedUtterance]:
    _validate_requested_speakers(speakers)
    for speaker_id in speakers:
        speaker = SPEAKER_BY_ID[speaker_id]
        archive_path = dataset_root / f"{speaker_id}.zip"
        try:
            with ZipFile(archive_path) as archive:
                members = frozenset(archive.namelist())
                annotation_members = sorted(
                    member
                    for member in members
                    if member.startswith(f"{speaker_id}/annotation/")
                    and member.endswith(".TextGrid")
                )
                for annotation_member in annotation_members:
                    utterance_id = Path(annotation_member).stem
                    transcript_member = f"{speaker_id}/transcript/{utterance_id}.txt"
                    audio_member = f"{speaker_id}/wav/{utterance_id}.wav"
                    missing = tuple(
                        member
                        for member in (transcript_member, audio_member)
                        if member not in members
                    )
                    if missing:
                        raise ValueError(
                            f"{speaker_id}/{utterance_id} is missing archive members: {missing}"
                        )
                    annotation = archive.read(annotation_member).decode("utf-8-sig")
                    transcript = archive.read(transcript_member).decode("utf-8-sig").strip()
                    yield _annotated_utterance(
                        speaker,
                        utterance_id,
                        transcript,
                        annotation,
                        audio_member,
                        archive.read(audio_member) if include_audio else None,
                    )
        except (BadZipFile, OSError) as error:
            raise ValueError(f"Cannot read L2-ARCTIC archive: {archive_path}") from error


def validate_l2arctic(dataset_root: Path) -> dict[str, object]:
    operation_counts: Counter[str] = Counter()
    contrasts: Counter[str] = Counter()
    annotations_by_speaker: dict[str, int] = {}
    mapping_total = 0
    mapping_failures: Counter[str] = Counter()
    utterance_count = 0

    for speaker in SPEAKERS:
        speaker_count = 0
        for utterance in iter_l2arctic_utterances(dataset_root, (speaker.id,), include_audio=False):
            utterance_count += 1
            speaker_count += 1
            for phone in utterance.phones:
                operation_counts[phone.error_type] += 1
                if phone.expected_arpabet is not None:
                    mapping_total += 1
                    if phone.expected_ipa is None:
                        mapping_failures[phone.expected_arpabet] += 1
                if phone.error_type == "substitution":
                    contrasts[f"{phone.expected_arpabet}->{phone.perceived_arpabet}"] += 1
        annotations_by_speaker[speaker.id] = speaker_count

    return {
        "report_version": 1,
        "experiment": "l2-arctic-integrity-v1",
        "dataset": {
            "id": L2ARCTIC_DATASET_ID,
            "license": L2ARCTIC_LICENSE,
            "root": str(dataset_root),
        },
        "checks": {
            "speaker_archives_present": len(annotations_by_speaker) == len(SPEAKERS),
            "annotation_count_matches_release": (utterance_count == L2ARCTIC_EXPECTED_ANNOTATIONS),
            "expected_phone_mapping_coverage_at_least_99_percent": (
                mapping_total > 0
                and (mapping_total - sum(mapping_failures.values())) / mapping_total >= 0.99
            ),
        },
        "metrics": {
            "speakers": len(annotations_by_speaker),
            "annotated_utterances": utterance_count,
            "annotations_by_speaker": annotations_by_speaker,
            "operation_counts": dict(sorted(operation_counts.items())),
            "expected_phone_mapping_total": mapping_total,
            "expected_phone_mapping_failures": dict(sorted(mapping_failures.items())),
            "expected_phone_mapping_coverage": (
                (mapping_total - sum(mapping_failures.values())) / mapping_total
                if mapping_total
                else 0.0
            ),
            "selected_contrasts": {
                contrast: contrasts[contrast] for contrast in ("TH->F", "TH->S", "R->L", "L->R")
            },
        },
    }


def _annotated_utterance(
    speaker: Speaker,
    utterance_id: str,
    transcript: str,
    annotation: str,
    audio_member: str,
    audio_bytes: bytes | None,
) -> AnnotatedUtterance:
    tiers = parse_textgrid(annotation)
    if "words" not in tiers or "phones" not in tiers:
        raise ValueError(f"{speaker.id}/{utterance_id} lacks words or phones tier")
    word_intervals = tuple(interval for interval in tiers["words"] if interval.text.strip())
    words = tuple(interval.text.strip() for interval in word_intervals)
    phones = tuple(
        phone
        for interval in tiers["phones"]
        if (phone := _parse_phone_interval(interval, word_intervals)) is not None
    )
    if not words or not phones:
        raise ValueError(f"{speaker.id}/{utterance_id} has no annotated words or phones")
    return AnnotatedUtterance(
        speaker=speaker,
        utterance_id=utterance_id,
        transcript=transcript,
        words=words,
        phones=phones,
        audio_member=audio_member,
        audio_bytes=audio_bytes,
    )


def parse_textgrid(content: str) -> dict[str, tuple[Interval, ...]]:
    tiers: dict[str, list[Interval]] = {}
    tier_name: str | None = None
    interval: dict[str, float] | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if ITEM_PATTERN.fullmatch(line):
            tier_name = None
            interval = None
            continue
        if INTERVAL_PATTERN.fullmatch(line):
            if tier_name is None:
                raise ValueError("TextGrid interval appears before its tier name")
            interval = {}
            continue
        match = ASSIGNMENT_PATTERN.fullmatch(line)
        if match is None:
            continue
        key = match.group("key")
        value = match.group("value")
        if key == "name":
            tier_name = _unquote(value)
            tiers.setdefault(tier_name, [])
            continue
        if interval is None or tier_name is None:
            continue
        if key in {"xmin", "xmax"}:
            interval[key] = float(value)
            continue
        if tier_name not in {"words", "phones"}:
            interval = None
            continue
        if "xmin" not in interval or "xmax" not in interval:
            raise ValueError("TextGrid interval text appears before its bounds")
        tiers[tier_name].append(
            Interval(
                start=interval["xmin"],
                end=interval["xmax"],
                text=_unquote(value),
            )
        )
        interval = None
    return {name: tuple(intervals) for name, intervals in tiers.items()}


def _parse_phone_interval(
    interval: Interval, word_intervals: tuple[Interval, ...]
) -> AnnotatedPhone | None:
    label = "".join(interval.text.split())
    if not label or label in {"sil", "sp"}:
        return None
    parts = label.split(",")
    if len(parts) == 1:
        expected = parts[0]
        perceived = expected
        error_type: ErrorType = "correct"
    elif len(parts) == 3 and parts[2].lower() in {"s", "d", "a"}:
        expected, perceived, raw_operation = parts
        operation = raw_operation.lower()
        error_type = cast(
            ErrorType, {"s": "substitution", "d": "deletion", "a": "addition"}[operation]
        )
        if expected == "sil":
            expected = None
        if perceived == "sil":
            perceived = None
    else:
        raise ValueError(f"Unsupported L2-ARCTIC phone annotation: {interval.text!r}")
    deviation = bool(perceived and "*" in perceived)
    expected = _clean_arpabet(expected)
    perceived = _clean_arpabet(perceived)
    return AnnotatedPhone(
        start=interval.start,
        end=interval.end,
        word_index=_word_index(interval, word_intervals),
        expected_arpabet=expected,
        perceived_arpabet=perceived,
        error_type=error_type,
        deviation=deviation,
    )


def _clean_arpabet(phone: str | None) -> str | None:
    if phone is None:
        return None
    cleaned = phone.replace("*", "")
    if cleaned == "err":
        return None
    return cleaned


def _map_l2_phone(phone: str | None) -> str | None:
    if phone is None:
        return None
    if phone == "AX":
        return "ə"
    mapped = map_arpabet_phone(phone)
    if mapped == "ɝː":
        return "ɜː"
    return mapped


def _word_index(phone: Interval, words: tuple[Interval, ...]) -> int | None:
    overlaps = [max(0.0, min(phone.end, word.end) - max(phone.start, word.start)) for word in words]
    if overlaps and max(overlaps) > 0.0:
        return max(range(len(overlaps)), key=overlaps.__getitem__)
    midpoint = (phone.start + phone.end) / 2
    return next(
        (index for index, word in enumerate(words) if word.start <= midpoint <= word.end),
        None,
    )


def _unquote(value: str) -> str:
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise ValueError(f"TextGrid value is not quoted: {value!r}")
    return value[1:-1].replace('""', '"')


def _speaker_hash(speaker: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{speaker}".encode()).digest()


def _validate_requested_speakers(speakers: tuple[str, ...]) -> None:
    unknown = tuple(sorted(set(speakers) - SPEAKER_BY_ID.keys()))
    if unknown:
        raise ValueError(f"Unknown L2-ARCTIC speakers: {unknown}")
    if len(set(speakers)) != len(speakers):
        raise ValueError("L2-ARCTIC speaker list contains duplicates")
