import inspect

import pronunciation_coach.analysis as analysis


def test_stage_zero_does_not_import_heavy_future_dependencies() -> None:
    forbidden = {"torch", "transformers", "gradio", "phonemizer"}
    source = inspect.getsource(analysis)
    assert all(f"import {package}" not in source for package in forbidden)
