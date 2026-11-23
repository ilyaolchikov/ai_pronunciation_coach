"""Versioned SpeechOcean ARPAbet to normalized IPA mapping."""

import json
import re
from importlib.resources import files
from typing import cast

from pronunciation_coach.phones import normalize_phone

STRESS_PATTERN = re.compile(r"^(?P<base>[A-Z]+)(?P<stress>[012])?$")


def load_arpabet_mapping() -> tuple[dict[str, str], dict[str, str]]:
    resource = files("pronunciation_coach.data").joinpath("arpabet_to_ipa.v1.json")
    raw = cast(dict[str, object], json.loads(resource.read_text(encoding="utf-8")))
    base = cast(dict[str, str], raw["base"])
    overrides = cast(dict[str, str], raw["overrides"])
    return base, overrides


def map_arpabet_phone(phone: str) -> str | None:
    match = STRESS_PATTERN.fullmatch(phone)
    if match is None:
        return None
    base_mapping, overrides = load_arpabet_mapping()
    if phone in overrides:
        return normalize_phone(overrides[phone])
    mapped = base_mapping.get(match.group("base"))
    return normalize_phone(mapped) if mapped is not None else None


def map_arpabet_sequence(phones: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapped: list[str] = []
    unmapped: list[str] = []
    for phone in phones:
        ipa = map_arpabet_phone(phone)
        if ipa is None:
            unmapped.append(phone)
            continue
        mapped.append(ipa)
    return tuple(mapped), tuple(unmapped)
