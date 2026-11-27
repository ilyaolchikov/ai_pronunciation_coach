# AI Pronunciation Coach

A local English read-aloud coach powered exclusively by
[Pronunciada](https://github.com/moglearn/pronunciada). The browser records or accepts a short
sentence, scores it locally, and colors each word green, yellow, or red from Pronunciada's
acceptance probability.

Pronunciada's code is Apache-2.0, while its bundled language data keeps the source licences
documented upstream. Its downloaded ZIPA model is CC BY 4.0. Keep the attributions in
`THIRD_PARTY_NOTICES.md` when distributing the application or model.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Run the UI

```bash
uv sync --all-groups --extra ui
uv run pronunciation-coach-ui
```

Open `http://127.0.0.1:7860`. The first analysis downloads Pronunciada's approximately 71 MB
English int8 model; later analyses run locally. The application does not retain a copy of
recordings or uploads after processing. Gradio may use its managed temporary upload area while
receiving and decoding browser audio; no recording is written to the project.

The current policy uses Pronunciada strictness `0.75`:

- red: acceptance probability below `0.5`;
- yellow: acceptance probability from `0.5` to below `0.8`;
- green: acceptance probability `0.8` or higher.

The technical panel contains only model timing, Pronunciada model and scoring settings, word
acceptance probabilities, and per-phone expected/heard probabilities.

## Analyze one file

```bash
uv run pronunciation-coach-audio \
  --prompt thought-about-it \
  --audio recording.wav
```

## Pronunciada benchmarks

The maintainer release suite contains human-corpus accuracy and performance checks. UI navigation
and audio non-retention are covered by the automated test suite. SpeechOcean requires the optional
`benchmark` dependency. L2-ARCTIC v5.0 must be
available locally at `l2arctic_release_v5.0/`; neither corpus is redistributed by this project.

```bash
uv sync --all-groups --extra benchmark
uv run pronunciation-coach-benchmark evaluate-corpora
uv run pronunciation-coach-benchmark evaluate-performance \
  --audio /path/to/reference-recording.wav \
  --prompt <matching-prompt-id>
```

The performance command accepts any local recording that matches its prompt. Use `--overwrite`
to replace an existing report. Release benchmarks always use the frozen int8 model configuration.

## Checks

```bash
make check
```
