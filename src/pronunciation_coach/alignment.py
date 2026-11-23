"""Deterministic global phone-sequence alignment."""

from dataclasses import dataclass
from itertools import product

from pronunciation_coach.phones import normalize_phones
from pronunciation_coach.schemas import AlignmentResult, Operation, PhoneEvidence, Prompt


@dataclass(frozen=True, slots=True)
class ExpectedPhone:
    value: str
    word_index: int
    phone_index: int


@dataclass(frozen=True, slots=True)
class _BackPointer:
    operation: Operation


def align_prompt(prompt: Prompt, observed: tuple[str, ...]) -> AlignmentResult:
    normalized_observed = normalize_phones(observed)
    variant_ranges = tuple(range(len(word.phone_variants)) for word in prompt.words)

    best: AlignmentResult | None = None
    for variant_indices in product(*variant_ranges):
        expected = _flatten_expected(prompt, variant_indices)
        candidate = align_sequences(expected, normalized_observed, variant_indices)
        candidate_key = (candidate.cost, candidate.selected_variant_indices)
        if best is None or candidate_key < (best.cost, best.selected_variant_indices):
            best = candidate

    if best is None:
        raise ValueError("prompt must contain at least one pronunciation combination")
    return best


def align_sequences(
    expected: tuple[ExpectedPhone, ...],
    observed: tuple[str, ...],
    variant_indices: tuple[int, ...] = (),
) -> AlignmentResult:
    row_count = len(expected) + 1
    column_count = len(observed) + 1
    costs = [[0] * column_count for _ in range(row_count)]
    back: list[list[_BackPointer | None]] = [[None] * column_count for _ in range(row_count)]

    for row in range(1, row_count):
        costs[row][0] = row
        back[row][0] = _BackPointer(Operation.DELETION)
    for column in range(1, column_count):
        costs[0][column] = column
        back[0][column] = _BackPointer(Operation.INSERTION)

    for row in range(1, row_count):
        for column in range(1, column_count):
            is_match = expected[row - 1].value == observed[column - 1]
            diagonal_operation = Operation.MATCH if is_match else Operation.SUBSTITUTION
            diagonal_cost = costs[row - 1][column - 1] + (0 if is_match else 1)
            candidates = (
                (diagonal_cost, _BackPointer(diagonal_operation)),
                (costs[row - 1][column] + 1, _BackPointer(Operation.DELETION)),
                (costs[row][column - 1] + 1, _BackPointer(Operation.INSERTION)),
            )
            costs[row][column], back[row][column] = min(candidates, key=lambda item: item[0])

    evidence = _backtrack(expected, observed, back)
    return AlignmentResult(
        cost=costs[-1][-1],
        selected_variant_indices=variant_indices,
        expected_phones=tuple(phone.value for phone in expected),
        evidence=evidence,
    )


def _flatten_expected(
    prompt: Prompt, variant_indices: tuple[int, ...]
) -> tuple[ExpectedPhone, ...]:
    flattened: list[ExpectedPhone] = []
    for word_index, (word, variant_index) in enumerate(
        zip(prompt.words, variant_indices, strict=True)
    ):
        variant = normalize_phones(word.phone_variants[variant_index])
        for phone_index, phone in enumerate(variant):
            flattened.append(ExpectedPhone(phone, word_index, phone_index))
    return tuple(flattened)


def _backtrack(
    expected: tuple[ExpectedPhone, ...],
    observed: tuple[str, ...],
    back: list[list[_BackPointer | None]],
) -> tuple[PhoneEvidence, ...]:
    row = len(expected)
    column = len(observed)
    reversed_evidence: list[PhoneEvidence] = []

    while row > 0 or column > 0:
        pointer = back[row][column]
        if pointer is None:
            raise RuntimeError("alignment backpointer is missing")

        if pointer.operation in (Operation.MATCH, Operation.SUBSTITUTION):
            expected_phone = expected[row - 1]
            reversed_evidence.append(
                PhoneEvidence(
                    expected=expected_phone.value,
                    observed=observed[column - 1],
                    operation=pointer.operation,
                    word_index=expected_phone.word_index,
                    phone_index=expected_phone.phone_index,
                )
            )
            row -= 1
            column -= 1
            continue

        if pointer.operation is Operation.DELETION:
            expected_phone = expected[row - 1]
            reversed_evidence.append(
                PhoneEvidence(
                    expected=expected_phone.value,
                    observed=None,
                    operation=Operation.DELETION,
                    word_index=expected_phone.word_index,
                    phone_index=expected_phone.phone_index,
                )
            )
            row -= 1
            continue

        word_index, boundary_ambiguous = _attribute_insertion(expected, row)
        reversed_evidence.append(
            PhoneEvidence(
                expected=None,
                observed=observed[column - 1],
                operation=Operation.INSERTION,
                word_index=word_index,
                boundary_ambiguous=boundary_ambiguous,
            )
        )
        column -= 1

    reversed_evidence.reverse()
    return tuple(reversed_evidence)


def _attribute_insertion(
    expected: tuple[ExpectedPhone, ...], boundary: int
) -> tuple[int | None, bool]:
    if not expected:
        return None, False
    if boundary == 0:
        return expected[0].word_index, False
    if boundary == len(expected):
        return expected[-1].word_index, False

    left_word = expected[boundary - 1].word_index
    right_word = expected[boundary].word_index
    return left_word, left_word != right_word
