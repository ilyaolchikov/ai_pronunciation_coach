"""Local-only Gradio UI for the bounded Stage 2 analyzer."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gradio as gr

from pronunciation_coach.config import DEFAULT_MODEL_CACHE
from pronunciation_coach.prompts import load_prompts
from pronunciation_coach.runtime import get_analyzer
from pronunciation_coach.schemas import Prompt
from pronunciation_coach.ui_adapter import BrowserAudio, analyze_for_ui, prompt_text

READY_STATUS = "Record or upload audio, then select **Analyze pronunciation**."
READY_NOTES = "A highlighted word is model evidence, not proof of a pronunciation error."


def navigate_prompt(
    prompt_index: int, offset: int, prompts: tuple[Prompt, ...]
) -> tuple[int, str, None, list[tuple[str, str]], str, str, None]:
    new_index = (prompt_index + offset) % len(prompts)
    return new_index, prompt_text(prompts[new_index].id), None, [], READY_STATUS, READY_NOTES, None


def build_app(
    cache_dir: Path = DEFAULT_MODEL_CACHE,
    device: str = "auto",
) -> gr.Blocks:
    prompts = load_prompts()
    first_prompt = prompts[0]

    def run_analysis(prompt_index: int, audio: BrowserAudio | None) -> tuple[Any, ...]:
        result = analyze_for_ui(
            prompts[prompt_index].id,
            audio,
            lambda: get_analyzer(cache_dir, device),
        )
        return result.highlights, result.status, result.notes, result.details

    def previous_prompt(prompt_index: int) -> tuple[Any, ...]:
        return navigate_prompt(prompt_index, -1, prompts)

    def next_prompt(prompt_index: int) -> tuple[Any, ...]:
        return navigate_prompt(prompt_index, 1, prompts)

    def start_analysis() -> tuple[gr.Button, str]:
        return gr.Button(interactive=False), "Loading the local model and analyzing audio..."

    with gr.Blocks(title="AI Pronunciation Coach") as app:
        gr.Markdown(
            "# AI Pronunciation Coach\n\n"
            "Read a short English sentence. Processing stays in this local app, and the "
            "result is evidence—not a pronunciation grade. Submitted audio is saved locally "
            "under `recordings/ui/`."
        )
        prompt_index = gr.State(0)
        expected_text = gr.Markdown(prompt_text(first_prompt.id))
        with gr.Row():
            previous_button = gr.Button("← Previous sentence")
            next_button = gr.Button("Next sentence →")
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Your recording (0.25-15 seconds)",
        )
        analyze_button = gr.Button("Analyze pronunciation", variant="primary")
        highlights = gr.HighlightedText(
            label="Word review",
            color_map={
                "Review": "#B42318",
                "Check again": "#D97706",
                "No problem detected": "#027A48",
            },
            show_legend=True,
            show_inline_category=True,
            combine_adjacent=False,
            adjacent_separator=" ",
        )
        status = gr.Markdown(READY_STATUS)
        notes = gr.Markdown(READY_NOTES)
        with gr.Accordion("Technical evidence and timing", open=False):
            details = gr.JSON()

        navigation_outputs = [
            prompt_index,
            expected_text,
            audio,
            highlights,
            status,
            notes,
            details,
        ]
        previous_button.click(
            previous_prompt,
            inputs=prompt_index,
            outputs=navigation_outputs,
            queue=False,
        )
        next_button.click(
            next_prompt,
            inputs=prompt_index,
            outputs=navigation_outputs,
            queue=False,
        )
        submission = analyze_button.click(
            start_analysis,
            outputs=[analyze_button, status],
            queue=False,
            trigger_mode="once",
        ).then(
            run_analysis,
            inputs=[prompt_index, audio],
            outputs=[highlights, status, notes, details],
            concurrency_limit=1,
            trigger_mode="once",
        )
        submission.then(
            lambda: gr.Button(interactive=True),
            outputs=analyze_button,
            queue=False,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pronunciation-coach-ui")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    app = build_app(arguments.cache_dir, arguments.device)
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=arguments.port,
        share=False,
        show_error=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
