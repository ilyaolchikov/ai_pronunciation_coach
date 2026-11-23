"""Loading and validation for the versioned prompt catalogue."""

import json
from importlib.resources import files

from pydantic import TypeAdapter, ValidationError

from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.schemas import Prompt

PROMPT_ADAPTER = TypeAdapter(tuple[Prompt, ...])


def load_prompts() -> tuple[Prompt, ...]:
    resource = files("pronunciation_coach.data").joinpath("prompts.v1.json")
    try:
        raw = json.loads(resource.read_text(encoding="utf-8"))
        prompts = PROMPT_ADAPTER.validate_python(raw)
    except (json.JSONDecodeError, OSError, ValidationError) as error:
        raise CoachError(ErrorCode.CATALOGUE_INVALID, "Prompt catalogue is invalid") from error

    prompt_ids = [prompt.id for prompt in prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise CoachError(ErrorCode.CATALOGUE_INVALID, "Prompt IDs must be unique")
    return prompts


def get_prompt(prompt_id: str, prompts: tuple[Prompt, ...] | None = None) -> Prompt:
    catalogue = load_prompts() if prompts is None else prompts
    for prompt in catalogue:
        if prompt.id == prompt_id:
            return prompt
    raise CoachError(ErrorCode.PROMPT_UNKNOWN, f"Unknown prompt: {prompt_id}")
