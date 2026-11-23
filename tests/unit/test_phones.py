import pytest

from pronunciation_coach.phones import normalize_phone, parse_phone_sequence


def test_normalizes_unicode_to_nfc() -> None:
    assert normalize_phone("e\u0301") == "é"


def test_rejects_whitespace_inside_phone() -> None:
    with pytest.raises(ValueError, match="must not contain whitespace"):
        normalize_phone("a b")


def test_parses_empty_sequence() -> None:
    assert parse_phone_sequence("   ") == ()


def test_parses_space_separated_sequence() -> None:
    assert parse_phone_sequence(" aɪ   θ ") == ("aɪ", "θ")
