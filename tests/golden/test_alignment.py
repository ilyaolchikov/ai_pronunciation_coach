from pronunciation_coach.alignment import ExpectedPhone, align_prompt, align_sequences
from pronunciation_coach.schemas import Operation, Prompt, PromptWord


def expected(*phones: str) -> tuple[ExpectedPhone, ...]:
    return tuple(ExpectedPhone(phone, 0, index) for index, phone in enumerate(phones))


def operations(*phones: str, observed: tuple[str, ...]) -> tuple[Operation, ...]:
    result = align_sequences(expected(*phones), observed)
    return tuple(item.operation for item in result.evidence)


def test_exact_match() -> None:
    result = align_sequences(expected("θ", "ɔː", "t"), ("θ", "ɔː", "t"))

    assert result.cost == 0
    assert tuple(item.operation for item in result.evidence) == (Operation.MATCH,) * 3


def test_substitution() -> None:
    result = align_sequences(expected("θ"), ("s",))

    assert result.cost == 1
    assert result.evidence[0].model_dump(mode="json") == {
        "expected": "θ",
        "observed": "s",
        "operation": "substitution",
        "word_index": 0,
        "phone_index": 0,
        "boundary_ambiguous": False,
        "displayable": False,
    }


def test_deletion() -> None:
    assert operations("a", "b", observed=("a",)) == (Operation.MATCH, Operation.DELETION)


def test_insertion() -> None:
    assert operations("a", observed=("a", "b")) == (Operation.MATCH, Operation.INSERTION)


def test_empty_expected_sequence() -> None:
    result = align_sequences((), ("a",))

    assert result.cost == 1
    assert result.evidence[0].operation is Operation.INSERTION
    assert result.evidence[0].word_index is None


def test_empty_observed_sequence() -> None:
    assert operations("a", "b", observed=()) == (Operation.DELETION, Operation.DELETION)


def test_diagonal_wins_a_repeated_phone_tie() -> None:
    result = align_sequences(expected("a", "a"), ("a",))

    assert tuple(item.operation for item in result.evidence) == (
        Operation.DELETION,
        Operation.MATCH,
    )


def test_insertion_at_word_boundary_is_marked_ambiguous() -> None:
    sequence = (
        ExpectedPhone("a", word_index=0, phone_index=0),
        ExpectedPhone("b", word_index=1, phone_index=0),
    )

    result = align_sequences(sequence, ("a", "x", "b"))
    insertion = next(item for item in result.evidence if item.operation is Operation.INSERTION)

    assert insertion.word_index == 0
    assert insertion.boundary_ambiguous is True


def test_selects_lowest_cost_pronunciation_variant() -> None:
    prompt = Prompt(
        id="variant-test",
        text="A word.",
        locale="en-us",
        words=(
            PromptWord(text="A", phone_variants=(("a",),)),
            PromptWord(text="word", phone_variants=(("w", "ɜː", "d"), ("w", "ɜː", "ɹ", "d"))),
        ),
        catalogue_version=1,
    )

    result = align_prompt(prompt, ("a", "w", "ɜː", "ɹ", "d"))

    assert result.cost == 0
    assert result.selected_variant_indices == (0, 1)


def test_variant_tie_prefers_earlier_catalogue_variant() -> None:
    prompt = Prompt(
        id="variant-tie",
        text="Word.",
        locale="en-us",
        words=(PromptWord(text="Word", phone_variants=(("a",), ("b",))),),
        catalogue_version=1,
    )

    result = align_prompt(prompt, ("x",))

    assert result.selected_variant_indices == (0,)
