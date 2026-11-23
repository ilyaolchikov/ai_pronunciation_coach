"""Reviewed deterministic coaching-tip catalogue."""

import json
from importlib.resources import files

from pydantic import TypeAdapter, ValidationError

from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.schemas import FeedbackTip

TIP_ADAPTER = TypeAdapter(tuple[FeedbackTip, ...])


def load_feedback() -> tuple[FeedbackTip, ...]:
    resource = files("pronunciation_coach.data").joinpath("feedback.v1.json")
    try:
        raw = json.loads(resource.read_text(encoding="utf-8"))
        tips = TIP_ADAPTER.validate_python(raw)
    except (json.JSONDecodeError, OSError, ValidationError) as error:
        raise CoachError(ErrorCode.CATALOGUE_INVALID, "Feedback catalogue is invalid") from error

    tip_ids = [tip.id for tip in tips]
    contrasts = [(tip.expected, tip.observed) for tip in tips]
    if len(tip_ids) != len(set(tip_ids)) or len(contrasts) != len(set(contrasts)):
        raise CoachError(ErrorCode.CATALOGUE_INVALID, "Feedback entries must be unique")
    return tips


def feedback_index(tips: tuple[FeedbackTip, ...]) -> dict[tuple[str, str], FeedbackTip]:
    return {(tip.expected, tip.observed): tip for tip in tips}
