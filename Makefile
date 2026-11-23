.PHONY: check format stage1-check test

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest

format:
	uv run ruff check --fix .
	uv run ruff format .

stage1-check: check
	uv run pronunciation-coach-catalogue validate
	RUN_MODEL_TESTS=1 HF_HUB_OFFLINE=1 uv run pytest -m model

test:
	uv run pytest
