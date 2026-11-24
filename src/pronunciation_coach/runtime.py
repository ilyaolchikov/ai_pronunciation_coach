"""Shared construction and process-local caching for the Stage 2 analyzer."""

from functools import lru_cache
from pathlib import Path

from pronunciation_coach.calibrated_analysis import CalibratedAnalyzer
from pronunciation_coach.config import DEFAULT_MODEL_CACHE
from pronunciation_coach.recognizers.wav2vec2_espeak import Wav2Vec2EspeakRecognizer


@lru_cache(maxsize=4)
def get_analyzer(
    cache_dir: Path = DEFAULT_MODEL_CACHE,
    device: str = "auto",
) -> CalibratedAnalyzer:
    return CalibratedAnalyzer(
        Wav2Vec2EspeakRecognizer(cache_dir=cache_dir, device=device, local_files_only=True)
    )
