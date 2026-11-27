"""Phone token Unicode normalization for corpus annotations."""

import unicodedata


def normalize_phone(phone: str) -> str:
    normalized = unicodedata.normalize("NFC", phone.strip())
    if not normalized:
        raise ValueError("phone must not be blank")
    if any(character.isspace() for character in normalized):
        raise ValueError(f"phone token must not contain whitespace: {phone!r}")
    return normalized
