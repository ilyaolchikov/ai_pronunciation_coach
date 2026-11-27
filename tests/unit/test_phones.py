import pytest

from pronunciation_coach.phones import normalize_phone


def test_normalizes_unicode_to_nfc() -> None:
    assert normalize_phone("e\u0301") == "é"


def test_rejects_whitespace_inside_phone() -> None:
    with pytest.raises(ValueError, match="must not contain whitespace"):
        normalize_phone("a b")
