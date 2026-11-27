import json
from pathlib import Path

import numpy as np
import soundfile as sf

from pronunciation_coach.performance_benchmark import measure_release_performance
from pronunciation_coach.pronunciada_analysis import ZIPA_MODEL


class FakeAnalyzer:
    def analyze(self, prompt: object, audio: object) -> object:
        return object()


def test_measures_cached_offline_performance(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.full(8_000, 0.1, dtype=np.float32), 16_000)
    model_dir = tmp_path / "models" / ZIPA_MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "model.int8.test.onnx").write_bytes(b"model")
    (model_dir / "model.int8.test.onnx.json").write_text(
        json.dumps({"source_size": 100}), encoding="utf-8"
    )

    report = measure_release_performance(
        FakeAnalyzer,
        audio_path=audio_path,
        prompt_id="thought-about-it",
        cache_dir=tmp_path,
        warm_runs=2,
    )

    metrics = report["metrics"]
    assert metrics["completion_rate"] == 1.0
    assert metrics["valid_duration_seconds"] == {"0.25": True, "15.0": True}
    assert metrics["no_network_after_cache"] is True
    assert report["model"]["source_download_size_bytes"] == 100
