from pronunciation_coach.phone_mapping import map_arpabet_phone, map_arpabet_sequence


def test_maps_stress_independent_phone() -> None:
    assert map_arpabet_phone("IY1") == "iː"


def test_maps_unstressed_vowel_override() -> None:
    assert map_arpabet_phone("AH0") == "ə"
    assert map_arpabet_phone("AH1") == "ʌ"
    assert map_arpabet_phone("ER0") == "ɚ"


def test_reports_unmapped_phone_without_guessing() -> None:
    mapped, unmapped = map_arpabet_sequence(("TH", "UNKNOWN", "R"))

    assert mapped == ("θ", "ɹ")
    assert unmapped == ("UNKNOWN",)
