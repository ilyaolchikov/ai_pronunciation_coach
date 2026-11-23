# pyright: reportUnknownMemberType=false
"""Constrained CTC Viterbi alignment and phone-posterior evidence."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatMatrix: TypeAlias = NDArray[np.float32] | NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PhonePosteriorScore:
    target_index: int
    phone: str
    start_frame: int
    end_frame: int
    expected_mean_log_probability: float
    expected_mean_probability: float
    alternative_phone: str
    alternative_mean_log_probability: float
    log_posterior_margin: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class ConstrainedCTCScore:
    phones: tuple[str, ...]
    phone_scores: tuple[PhonePosteriorScore, ...]
    path_mean_log_probability: float

    @property
    def mean_phone_margin(self) -> float:
        return float(np.mean([score.log_posterior_margin for score in self.phone_scores]))

    @property
    def p10_phone_margin(self) -> float:
        return float(np.percentile([score.log_posterior_margin for score in self.phone_scores], 10))


def constrained_ctc_score(
    log_probabilities: FloatMatrix,
    target_phones: tuple[str, ...],
    vocabulary: dict[str, int],
    *,
    blank_id: int,
    excluded_alternative_ids: frozenset[int] = frozenset(),
) -> ConstrainedCTCScore:
    """Align a known phone sequence and retain posterior evidence for every phone."""
    _validate_inputs(log_probabilities, target_phones, vocabulary, blank_id)
    target_ids = tuple(vocabulary[phone] for phone in target_phones)
    expanded_ids = _expand_with_blanks(target_ids, blank_id)
    state_path, path_score = _viterbi_path(log_probabilities, expanded_ids, blank_id)
    inverse_vocabulary = {token_id: token for token, token_id in vocabulary.items()}
    alternative_ids = tuple(
        token_id
        for token_id in sorted(inverse_vocabulary)
        if token_id != blank_id and token_id not in excluded_alternative_ids
    )
    if not alternative_ids:
        raise ValueError("vocabulary has no eligible alternative phone tokens")

    phone_scores = tuple(
        _score_phone(
            log_probabilities,
            state_path,
            target_index,
            phone,
            target_ids[target_index],
            alternative_ids,
            inverse_vocabulary,
        )
        for target_index, phone in enumerate(target_phones)
    )
    return ConstrainedCTCScore(
        phones=target_phones,
        phone_scores=phone_scores,
        path_mean_log_probability=path_score / log_probabilities.shape[0],
    )


def _validate_inputs(
    log_probabilities: FloatMatrix,
    target_phones: tuple[str, ...],
    vocabulary: dict[str, int],
    blank_id: int,
) -> None:
    if log_probabilities.ndim != 2:
        raise ValueError("log_probabilities must have shape [frames, vocabulary]")
    if log_probabilities.shape[0] == 0 or log_probabilities.shape[1] == 0:
        raise ValueError("log_probabilities must not be empty")
    if not np.isfinite(log_probabilities).all():
        raise ValueError("log_probabilities must be finite")
    if not target_phones:
        raise ValueError("target_phones must not be empty")
    missing = tuple(phone for phone in target_phones if phone not in vocabulary)
    if missing:
        raise ValueError(f"target phones are absent from the model vocabulary: {missing}")
    token_ids = tuple(vocabulary[phone] for phone in target_phones)
    if blank_id in token_ids:
        raise ValueError("target phones must not contain the CTC blank token")
    if blank_id < 0 or blank_id >= log_probabilities.shape[1]:
        raise ValueError("blank_id is outside the model vocabulary")
    if any(token_id < 0 or token_id >= log_probabilities.shape[1] for token_id in token_ids):
        raise ValueError("target phone ID is outside the probability matrix")


def _expand_with_blanks(target_ids: tuple[int, ...], blank_id: int) -> tuple[int, ...]:
    expanded = [blank_id]
    for token_id in target_ids:
        expanded.extend((token_id, blank_id))
    return tuple(expanded)


def _viterbi_path(
    log_probabilities: FloatMatrix,
    expanded_ids: tuple[int, ...],
    blank_id: int,
) -> tuple[NDArray[np.int64], float]:
    frame_count = log_probabilities.shape[0]
    state_count = len(expanded_ids)
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    backpointers = np.full((frame_count, state_count), -1, dtype=np.int64)
    previous[0] = float(log_probabilities[0, blank_id])
    if state_count > 1:
        previous[1] = float(log_probabilities[0, expanded_ids[1]])

    for frame_index in range(1, frame_count):
        current = np.full(state_count, -np.inf, dtype=np.float64)
        for state_index, token_id in enumerate(expanded_ids):
            candidates = [state_index]
            if state_index >= 1:
                candidates.append(state_index - 1)
            if (
                state_index >= 2
                and token_id != blank_id
                and token_id != expanded_ids[state_index - 2]
            ):
                candidates.append(state_index - 2)
            best_previous = max(candidates, key=lambda index: previous[index])
            if np.isneginf(previous[best_previous]):
                continue
            current[state_index] = previous[best_previous] + float(
                log_probabilities[frame_index, token_id]
            )
            backpointers[frame_index, state_index] = best_previous
        previous = current

    final_candidates = (state_count - 1, state_count - 2)
    final_state = max(final_candidates, key=lambda index: previous[index])
    if np.isneginf(previous[final_state]):
        raise ValueError("audio has too few CTC frames for the expected phone sequence")

    state_path = np.empty(frame_count, dtype=np.int64)
    state_path[-1] = final_state
    for frame_index in range(frame_count - 1, 0, -1):
        previous_state = backpointers[frame_index, state_path[frame_index]]
        if previous_state < 0:
            raise RuntimeError("CTC alignment backpointer is missing")
        state_path[frame_index - 1] = previous_state
    return state_path, float(previous[final_state])


def _score_phone(
    log_probabilities: FloatMatrix,
    state_path: NDArray[np.int64],
    target_index: int,
    phone: str,
    expected_id: int,
    alternative_ids: tuple[int, ...],
    inverse_vocabulary: dict[int, str],
) -> PhonePosteriorScore:
    token_state = 2 * target_index + 1
    frame_indexes = np.flatnonzero(state_path == token_state)
    if frame_indexes.size == 0:
        raise RuntimeError("constrained CTC path did not allocate a frame to a target phone")
    segment = log_probabilities[frame_indexes]
    expected_logs = segment[:, expected_id]
    eligible_alternatives = tuple(
        token_id for token_id in alternative_ids if token_id != expected_id
    )
    alternative_means = np.mean(segment[:, eligible_alternatives], axis=0)
    best_offset = int(np.argmax(alternative_means))
    best_alternative_id = eligible_alternatives[best_offset]
    expected_mean_log = float(np.mean(expected_logs))
    alternative_mean_log = float(alternative_means[best_offset])
    expected_mean_probability = float(np.mean(np.exp(expected_logs)))
    return PhonePosteriorScore(
        target_index=target_index,
        phone=phone,
        start_frame=int(frame_indexes[0]),
        end_frame=int(frame_indexes[-1]) + 1,
        expected_mean_log_probability=expected_mean_log,
        expected_mean_probability=min(max(expected_mean_probability, 0.0), 1.0),
        alternative_phone=inverse_vocabulary[best_alternative_id],
        alternative_mean_log_probability=alternative_mean_log,
        log_posterior_margin=expected_mean_log - alternative_mean_log,
    )
