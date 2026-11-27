import os

import numpy as np
import pytest

from pronunciation_coach.audio import normalize_audio
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.runtime import get_analyzer

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 to use the pinned Pronunciada model cache",
    ),
]


def test_cached_pronunciada_model_analyzes_a_prompt() -> None:
    get_analyzer.cache_clear()
    time = np.arange(32_000, dtype=np.float32) / 16_000
    audio = normalize_audio(0.2 * np.sin(2 * np.pi * 220 * time), 16_000)

    result = get_analyzer().analyze(get_prompt("thought-about-it"), audio)

    assert result.model.backend == "pronunciada"
    assert result.scoring.strictness == 0.75
    assert len(result.words) == 4
