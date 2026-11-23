"""Pipeline orchestration, word aggregation, and conservative feedback policy."""

from pronunciation_coach.alignment import align_prompt
from pronunciation_coach.feedback import feedback_index, load_feedback
from pronunciation_coach.recognizers.fake import FakePhoneRecognizer
from pronunciation_coach.schemas import (
    AnalysisResult,
    AudioMetadata,
    FeedbackTip,
    ModelMetadata,
    Operation,
    PhoneEvidence,
    Prompt,
    RecognitionResult,
    WordResult,
    WordStatus,
)

CORE_LIMITATION = "Phone alignment is deterministic evidence, not a proficiency grade."
SYNTHETIC_LIMITATIONS = (
    "Synthetic Stage 0 evidence; no audio or pronunciation was assessed.",
    CORE_LIMITATION,
)
ACOUSTIC_LIMITATIONS = (
    "Experimental acoustic-model estimate; not a proficiency grade.",
    CORE_LIMITATION,
)


def analyze(
    prompt: Prompt,
    recognizer: FakePhoneRecognizer,
    tips: tuple[FeedbackTip, ...] | None = None,
) -> AnalysisResult:
    recognition = recognizer.recognize()
    return analyze_recognition(prompt, recognition, tips=tips, limitations=SYNTHETIC_LIMITATIONS)


def analyze_recognition(
    prompt: Prompt,
    recognition: RecognitionResult,
    *,
    audio: AudioMetadata | None = None,
    tips: tuple[FeedbackTip, ...] | None = None,
    limitations: tuple[str, ...] = ACOUSTIC_LIMITATIONS,
) -> AnalysisResult:
    alignment = align_prompt(prompt, recognition.phones)
    feedback = load_feedback() if tips is None else tips
    words = aggregate_words(prompt, alignment.evidence, feedback)
    return AnalysisResult(
        prompt_id=prompt.id,
        prompt_catalogue_version=prompt.catalogue_version,
        expected_text=prompt.text,
        selected_variant_indices=alignment.selected_variant_indices,
        alignment_cost=alignment.cost,
        expected_phones=alignment.expected_phones,
        observed_phones=recognition.phones,
        audio=audio,
        model=ModelMetadata(
            backend=recognition.backend,
            revision=recognition.revision,
            inference_ms=recognition.inference_ms,
            device=recognition.device,
        ),
        words=words,
        limitations=limitations,
    )


def aggregate_words(
    prompt: Prompt,
    evidence: tuple[PhoneEvidence, ...],
    tips: tuple[FeedbackTip, ...],
) -> tuple[WordResult, ...]:
    tips_by_contrast = feedback_index(tips)
    results: list[WordResult] = []

    for word_index, prompt_word in enumerate(prompt.words):
        word_evidence = tuple(
            _apply_display_policy(item) for item in evidence if item.word_index == word_index
        )
        status = _word_status(word_evidence)
        tip_ids = _tip_ids(word_evidence, tips_by_contrast)
        results.append(
            WordResult(
                text=prompt_word.text,
                status=status,
                evidence=word_evidence,
                tips=tip_ids,
            )
        )
    return tuple(results)


def _apply_display_policy(evidence: PhoneEvidence) -> PhoneEvidence:
    displayable = evidence.operation is not Operation.MATCH and not evidence.boundary_ambiguous
    return evidence.model_copy(update={"displayable": displayable})


def _word_status(evidence: tuple[PhoneEvidence, ...]) -> WordStatus:
    expected_evidence = tuple(
        item for item in evidence if item.operation is not Operation.INSERTION
    )
    if expected_evidence and all(
        item.operation is Operation.DELETION for item in expected_evidence
    ):
        return WordStatus.NOT_DETECTED
    if any(item.operation is not Operation.MATCH for item in evidence):
        return WordStatus.REVIEW
    return WordStatus.MATCH


def _tip_ids(
    evidence: tuple[PhoneEvidence, ...],
    tips_by_contrast: dict[tuple[str, str], FeedbackTip],
) -> tuple[str, ...]:
    selected: list[str] = []
    for item in evidence:
        if item.operation is not Operation.SUBSTITUTION or not item.displayable:
            continue
        if item.expected is None or item.observed is None:
            continue
        tip = tips_by_contrast.get((item.expected, item.observed))
        if tip is not None and tip.id not in selected:
            selected.append(tip.id)
    return tuple(selected)
