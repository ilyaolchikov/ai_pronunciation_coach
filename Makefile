.PHONY: check format test

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest
