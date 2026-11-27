"""Release performance and cached-offline measurements."""

from __future__ import annotations

import json
import os
import platform
import resource
import socket
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from pronunciation_coach.audio import (
    NormalizedAudio,
    detect_speech_region,
    load_audio,
    normalize_audio,
)
from pronunciation_coach.errors import CoachError
from pronunciation_coach.prompts import get_prompt
from pronunciation_coach.pronunciada_analysis import ZIPA_MODEL
from pronunciation_coach.schemas import Prompt


class PerformanceAnalyzer(Protocol):
    def analyze(self, prompt: Prompt, audio: NormalizedAudio) -> object: ...


def measure_release_performance(
    analyzer_provider: Callable[[], PerformanceAnalyzer],
    *,
    audio_path: Path,
    prompt_id: str,
    cache_dir: Path,
    warm_runs: int,
) -> dict[str, Any]:
    if warm_runs < 1:
        raise ValueError("warm_runs must be positive")
    prompt = get_prompt(prompt_id)
    audio = detect_speech_region(load_audio(audio_path))
    network_attempts: list[str] = []
    failures: list[dict[str, str]] = []

    with _block_network(network_attempts):
        started = perf_counter()
        analyzer = analyzer_provider()
        cold_load_ms = (perf_counter() - started) * 1000.0

        first_inference_ms = _time_analysis(analyzer, prompt, audio, "first-inference", failures)
        warm_latencies = [
            value
            for index in range(warm_runs)
            if (
                value := _time_analysis(
                    analyzer, prompt, audio, f"warm-inference-{index + 1}", failures
                )
            )
            is not None
        ]
        boundary_results = {
            "0.25": _boundary_check(analyzer, prompt, audio, 0.25, failures),
            "15.0": _boundary_check(analyzer, prompt, audio, 15.0, failures),
        }

    expected = 1 + warm_runs + len(boundary_results)
    completed = expected - len(failures)
    source_size, cached_size = _model_sizes(cache_dir)
    return {
        "report_version": 1,
        "experiment": "pronunciada-release-performance-v1",
        "environment": {
            "machine": platform.machine(),
            "processor": _processor(),
            "physical_memory_bytes": _physical_memory_bytes(),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
        },
        "model": {
            "source_download_size_bytes": source_size,
            "cached_runtime_assets_bytes": cached_size,
        },
        "metrics": {
            "cold_model_load_ms": cold_load_ms,
            "first_inference_ms": first_inference_ms,
            "warm_runs": warm_runs,
            "warm_inference_p50_ms": median(warm_latencies) if warm_latencies else None,
            "warm_inference_p95_ms": _percentile(warm_latencies, 0.95),
            "peak_memory_bytes": _peak_memory_bytes(),
            "expected_analyses": expected,
            "completed_analyses": completed,
            "failed_analyses": len(failures),
            "completion_rate": completed / expected,
            "valid_duration_seconds": boundary_results,
            "no_network_after_cache": not network_attempts,
            "blocked_network_attempts": network_attempts,
        },
        "failures": failures,
        "notes": [
            "Cold load measures analyzer construction from the existing on-disk model cache.",
            "Peak memory is the process high-water mark after model load and all analyses.",
        ],
    }


def _time_analysis(
    analyzer: PerformanceAnalyzer,
    prompt: Prompt,
    audio: NormalizedAudio,
    stage: str,
    failures: list[dict[str, str]],
) -> float | None:
    started = perf_counter()
    try:
        analyzer.analyze(prompt, audio)
    except (CoachError, OSError, RuntimeError, TypeError, ValueError) as error:
        failures.append({"stage": stage, "error": str(error)})
        return None
    return (perf_counter() - started) * 1000.0


def _boundary_check(
    analyzer: PerformanceAnalyzer,
    prompt: Prompt,
    audio: NormalizedAudio,
    duration_seconds: float,
    failures: list[dict[str, str]],
) -> bool:
    frames = round(16_000 * duration_seconds)
    region = audio.metadata.speech_region
    if region is None:
        raise ValueError("Reference audio has no detected speech region")
    source = audio.samples[region.start_ms * 16 : region.end_ms * 16]
    repeated = np.empty(frames, dtype=np.float32)
    for start in range(0, frames, len(source)):
        end = min(start + len(source), frames)
        repeated[start:end] = source[: end - start]
    stage = f"duration-{duration_seconds}"
    try:
        boundary_audio = detect_speech_region(normalize_audio(repeated, 16_000))
    except (CoachError, TypeError, ValueError) as error:
        failures.append({"stage": stage, "error": str(error)})
        return False
    return _time_analysis(analyzer, prompt, boundary_audio, stage, failures) is not None


def _model_sizes(cache_dir: Path) -> tuple[int, int]:
    model_dir = cache_dir / "models" / ZIPA_MODEL
    sidecars = list(model_dir.glob("model.int8*.onnx.json"))
    if len(sidecars) != 1:
        raise ValueError("Expected one cached int8 model sidecar")
    metadata = json.loads(sidecars[0].read_text(encoding="utf-8"))
    source_size = metadata.get("source_size")
    if not isinstance(source_size, int) or source_size <= 0:
        raise ValueError("Cached model sidecar has no valid source_size")
    cached_size = sum(path.stat().st_size for path in model_dir.iterdir() if path.is_file())
    return source_size, cached_size


@contextmanager
def _block_network(attempts: list[str]):  # type: ignore[no-untyped-def]
    original_socket = socket.socket

    class BlockedSocket(original_socket):  # type: ignore[misc, valid-type]
        def connect(self, address: object) -> None:
            attempts.append(repr(address))
            raise OSError("network disabled for cached-offline benchmark")

        def connect_ex(self, address: object) -> int:
            attempts.append(repr(address))
            return 1

    socket.socket = BlockedSocket
    try:
        yield
    finally:
        socket.socket = original_socket


def _processor() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        profile = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            check=False,
            text=True,
        )
        for line in profile.stdout.splitlines():
            if line.strip().startswith("Chip:"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None


def _peak_memory_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if platform.system() == "Darwin" else maximum * 1024)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]
