from pronunciation_coach.analysis import aggregate_words, analyze
from pronunciation_coach.feedback import load_feedback
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.recognizers.fake import FakePhoneRecognizer
from pronunciation_coach.schemas import Operation, PhoneEvidence, WordStatus


def test_exact_sequence_marks_every_word_as_match() -> None:
    prompt = get_prompt("thought-about-it")
    observed = ("aɪ", "θ", "ɔː", "t", "ɐ", "b", "aʊ", "t", "ɪ", "t")

    result = analyze(prompt, FakePhoneRecognizer(observed))

    assert {word.status for word in result.words} == {WordStatus.MATCH}
    assert all(not word.tips for word in result.words)


def test_supported_substitution_creates_review_and_tip() -> None:
    prompt = get_prompt("thought-about-it")
    observed = ("aɪ", "s", "ɔː", "t", "ɐ", "b", "aʊ", "t", "ɪ", "t")

    result = analyze(prompt, FakePhoneRecognizer(observed))
    thought = result.words[1]

    assert thought.status is WordStatus.REVIEW
    assert thought.tips == ("theta-vs-s",)
    substitution = next(
        item for item in thought.evidence if item.operation is Operation.SUBSTITUTION
    )
    assert substitution.displayable is True


def test_unsupported_substitution_never_creates_tip() -> None:
    prompt = get_prompt("thought-about-it")
    observed = ("aɪ", "f", "ɔː", "t", "ɐ", "b", "aʊ", "t", "ɪ", "t")

    result = analyze(prompt, FakePhoneRecognizer(observed))

    assert result.words[1].status is WordStatus.REVIEW
    assert result.words[1].tips == ()


def test_all_expected_phones_deleted_marks_word_not_detected() -> None:
    prompt = get_prompt("thought-about-it")
    observed = ("aɪ", "ɐ", "b", "aʊ", "t", "ɪ", "t")

    result = analyze(prompt, FakePhoneRecognizer(observed))

    assert result.words[1].status is WordStatus.NOT_DETECTED


def test_ambiguous_insertion_is_not_displayable_and_has_no_tip() -> None:
    prompt = get_prompt("thought-about-it")
    evidence = (
        PhoneEvidence(
            expected="aɪ",
            observed="aɪ",
            operation=Operation.MATCH,
            word_index=0,
            phone_index=0,
        ),
        PhoneEvidence(
            expected=None,
            observed="s",
            operation=Operation.INSERTION,
            word_index=0,
            boundary_ambiguous=True,
        ),
    )

    words = aggregate_words(prompt, evidence, load_feedback())

    assert words[0].status is WordStatus.REVIEW
    assert words[0].evidence[1].displayable is False
    assert words[0].tips == ()


def test_every_expected_phone_is_aggregated_to_its_word() -> None:
    prompt = get_prompt("thought-about-it")
    observed = ("aɪ", "θ", "ɔː", "t", "ɐ", "b", "aʊ", "t", "ɪ", "t")

    result = analyze(prompt, FakePhoneRecognizer(observed))

    assert tuple(len(word.evidence) for word in result.words) == (1, 3, 4, 2)
