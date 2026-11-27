import inspect

import pronunciation_coach.runtime as runtime
import pronunciation_coach.ui as ui


def test_ui_and_runtime_have_only_the_pronunciada_backend() -> None:
    source = inspect.getsource(ui) + inspect.getsource(runtime)

    assert "pronunciada" in source.casefold()
    assert "wav2vec" not in source.casefold()
    assert "transformers" not in source
    assert "torch" not in source
