"""Process-local Pronunciada analyzer construction and caching."""

import os
from functools import lru_cache
from pathlib import Path

from pronunciation_coach.config import DEFAULT_PRONUNCIADA_CACHE
from pronunciation_coach.errors import CoachError, ErrorCode
from pronunciation_coach.pronunciada_analysis import PronunciadaAnalyzer, verify_pronunciada_assets


@lru_cache(maxsize=2)
def get_analyzer(
    cache_dir: Path = DEFAULT_PRONUNCIADA_CACHE,
    precision: str = "int8",
    threads: int = 4,
) -> PronunciadaAnalyzer:
    os.environ["PRONUNCIADA_CACHE"] = str(cache_dir.resolve())
    analyzer = PronunciadaAnalyzer.create(precision=precision, threads=threads)
    try:
        verify_pronunciada_assets(cache_dir, precision)
    except ValueError as error:
        raise CoachError(
            ErrorCode.MODEL_UNAVAILABLE, "Pronunciada model assets are missing or invalid"
        ) from error
    return analyzer
