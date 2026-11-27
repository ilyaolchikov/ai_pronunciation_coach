"""Loading and validation for the versioned prompt catalogue."""

import json
from importlib.resources import files

from pydantic import TypeAdapter, ValidationError

from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.schemas import Prompt

PROMPT_ADAPTER = TypeAdapter(tuple[Prompt, ...])


def load_prompts() -> tuple[Prompt, ...]:
    data = files("pronunciation_coach.data")
    try:
        frozen: object = json.loads(data.joinpath("prompts.v1.json").read_text(encoding="utf-8"))
        additions: object = json.loads(
            data.joinpath("practice-prompts.v2.json").read_text(encoding="utf-8")
        )
        if not isinstance(frozen, list) or not isinstance(additions, list):
            raise ValueError("prompt catalogue resources must be arrays")
        prompts = tuple(
            prompt.model_copy(update={"catalogue_version": 2})
            for prompt in PROMPT_ADAPTER.validate_python([*frozen, *additions])
        )
    except (json.JSONDecodeError, OSError, ValidationError, ValueError) as error:
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
