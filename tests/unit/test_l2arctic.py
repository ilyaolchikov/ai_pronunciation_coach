from pathlib import Path
from zipfile import ZipFile

from pronunciation_coach.l2arctic import (
    SPEAKER_BY_ID,
    iter_l2arctic_utterances,
    parse_textgrid,
    prepare_l2arctic_manifest,
)

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

item [1]:
    class = "IntervalTier"
    name = "words"
    intervals [1]:
        xmin = 0
        xmax = 0.5
        text = "three"
item [2]:
    class = "IntervalTier"
    name = "phones"
    intervals [1]:
        xmin = 0
        xmax = 0.2
        text = "TH, F, S"
    intervals [2]:
        xmin = 0.2
        xmax = 0.3
        text = "R"
    intervals [3]:
        xmin = 0.3
        xmax = 0.5
        text = "IY1"
"""


def test_manifest_is_deterministic_balanced_and_disjoint() -> None:
    first = prepare_l2arctic_manifest()
    second = prepare_l2arctic_manifest()

    assert first == second
    assert len(first.train_speakers) == 12
    assert len(first.development_speakers) == 6
    assert len(first.final_test_speakers) == 6
    all_speakers = first.train_speakers + first.development_speakers + first.final_test_speakers
    assert set(all_speakers) == set(SPEAKER_BY_ID)


def test_textgrid_parser_retains_only_requested_tiers() -> None:
    tiers = parse_textgrid(TEXTGRID)

    assert tuple(tiers) == ("words", "phones")
    assert tiers["words"][0].text == "three"
    assert tiers["phones"][0].text == "TH, F, S"


def test_zip_loader_parses_explicit_substitution(tmp_path: Path) -> None:
    archive_path = tmp_path / "ABA.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("ABA/annotation/arctic_a0001.TextGrid", TEXTGRID)
        archive.writestr("ABA/transcript/arctic_a0001.txt", "Three")
        archive.writestr("ABA/wav/arctic_a0001.wav", b"not-read-in-this-test")

    utterance = next(iter_l2arctic_utterances(tmp_path, ("ABA",), include_audio=False))

    assert utterance.words == ("three",)
    assert utterance.audio_bytes is None
    assert utterance.phones[0].expected_ipa == "θ"
    assert utterance.phones[0].perceived_ipa == "f"
    assert utterance.phones[0].error_type == "substitution"
    assert utterance.phones[0].word_index == 0
