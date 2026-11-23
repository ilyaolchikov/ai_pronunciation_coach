"""Phone token parsing and Unicode normalization."""

import unicodedata
from collections.abc import Iterable


def normalize_phone(phone: str) -> str:
    normalized = unicodedata.normalize("NFC", phone.strip())
    if not normalized:
        raise ValueError("phone must not be blank")
    if any(character.isspace() for character in normalized):
        raise ValueError(f"phone token must not contain whitespace: {phone!r}")
    return normalized


def normalize_phones(phones: Iterable[str]) -> tuple[str, ...]:
    return tuple(normalize_phone(phone) for phone in phones)


def parse_phone_sequence(sequence: str) -> tuple[str, ...]:
    if not sequence.strip():
        return ()
    return normalize_phones(sequence.split())
