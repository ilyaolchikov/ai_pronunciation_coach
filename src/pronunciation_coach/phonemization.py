# pyright: reportUnknownVariableType=false
"""eSpeak-backed canonical phone generation for catalogue review."""

from importlib.metadata import version
from typing import cast

from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from pronunciation_coach.phones import parse_phone_sequence
from pronunciation_coach.schemas import GenerationMetadata, Prompt

SEPARATOR = Separator(phone=" ", word="|")


def phonemize_words(words: tuple[str, ...], locale: str = "en-us") -> tuple[tuple[str, ...], ...]:
    raw_output = phonemize(
        list(words),
        language=locale,
        backend="espeak",
        separator=SEPARATOR,
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",
        words_mismatch="warn",
    )
    if isinstance(raw_output, str):
        output = [raw_output]
    else:
        if not all(isinstance(item, str) for item in raw_output):
            raise RuntimeError("phonemizer returned text pairs unexpectedly")
        output = cast(list[str], raw_output)
    return tuple(parse_phone_sequence(item.replace("|", " ")) for item in output)


def generation_metadata() -> GenerationMetadata:
    espeak_version = ".".join(str(part) for part in EspeakBackend.version())
    return GenerationMetadata(phonemizer=version("phonemizer"), espeak_ng=espeak_version)


def validate_prompt_phones(prompt: Prompt) -> tuple[tuple[int, tuple[str, ...]], ...]:
    generated = phonemize_words(tuple(word.text for word in prompt.words), prompt.locale)
    mismatches: list[tuple[int, tuple[str, ...]]] = []
    for word_index, (word, generated_variant) in enumerate(
        zip(prompt.words, generated, strict=True)
    ):
        if generated_variant not in word.phone_variants:
            mismatches.append((word_index, generated_variant))
    return tuple(mismatches)
