# AI Pronunciation Coach

A local-first English read-aloud coach. Stage 3 adds a local browser UI over the Stage 2
engine: microphone/upload input, calibrated red/yellow/green word states, limitations, and
expandable technical evidence. The deterministic Stage 0 kernel and all feasibility
experiments remain available as reproducible foundations.

Stage 0 provides the deterministic kernel. Stage 1 adds bounded audio processing, a
pinned local Wav2Vec2 phone recognizer, eSpeak catalogue validation, and a frozen
SpeechOcean762 feasibility benchmark.

The Stage 1 decision is **stop before the learner-facing UI**: aggregate signal and local
performance are promising, but the raw edit policy incorrectly flags 93.75% of highly
rated benchmark recordings. See [the feasibility report](docs/stage1-feasibility.md).

The Stage 1B constrained-CTC experiment reduces that development false-review rate to 25%
by using frame-level posterior margins, at the cost of detecting only 25.6% of marked
problem words. It is a held-out-validation candidate, not learner-ready feedback. See
[the Stage 1B report](docs/stage1b-ctc-posterior.md).

The frozen 1,240-recording held-out evaluation subsequently returned **stop**: correlation,
recall, false-review, performance, and reliability gates passed, but severe-problem word
precision was 26.2% against a precommitted 40% minimum.

English-only competitors and canonical variants improved LV-60 to 34.2% held-out
precision and 43.2% recall. An identical XLSR-53 A/B reached 24.5% precision and 71.9%
recall. Neither raw-margin candidate advanced.

Stage 1C calibrates word-error probabilities from the existing CTC evidence with
speaker-disjoint five-fold development predictions. Its frozen `0.30` classifier passed
the consumed held-out-validation split at 43.7% precision, 35.9% recall, and 11.5%
high-rated false reviews. This unblocks bounded Stage 2 implementation; it does not make
the system learner-ready. The 63-speaker final-test partition remains unused. See
[the Stage 1C report](docs/stage1c-word-calibration.md).

An L2-ARCTIC phone calibration now supplements that classifier. On six untouched L2
speakers it improved red-candidate precision and recall, but a SpeechOcean regression
showed that it should not replace the general red policy. It is therefore deployed only
as a yellow phone-sensitive warning at probability `0.42`. See
[the L2-ARCTIC calibration report](docs/l2arctic-phone-calibration.md).

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup and checks

```bash
uv sync --all-groups
make check
```

Install the optional acoustic and benchmark stack with:

```bash
uv sync --all-groups --extra stage1 --extra benchmark
uv run pronunciation-coach-model prefetch
# Optional pinned A/B checkpoint (about 1.26 GB):
uv run pronunciation-coach-model prefetch --model xlsr53
uv run pronunciation-coach-catalogue validate
```

## Run the local browser UI

Install the acoustic engine and UI, then prefetch the pinned model once while online:

```bash
uv sync --all-groups --extra stage1 --extra ui
uv run pronunciation-coach-model prefetch
uv run pronunciation-coach-ui
```

Open `http://127.0.0.1:7860`, use Previous/Next to choose among 15 sentences, and record
or upload 0.25–15 seconds of audio. Navigating clears the previous recording and results.
The app binds only to localhost and Gradio public sharing is disabled. Browser
audio is saved as a WAV under the Git-ignored `recordings/ui/` directory before analysis.
Uploads are saved there too because the NumPy input does not distinguish their source.
The first analysis can be slower while the cached model loads. See
[the Stage 3 UI guide](docs/stage3-ui.md) for result meanings, troubleshooting, and the
manual acceptance checklist.

## Try the deterministic CLI

Exact sequence:

```bash
uv run pronunciation-coach analyze \
  --prompt thought-about-it \
  --fake-observed "aɪ θ ɔː t ɐ b aʊ t ɪ t"
```

Simulated `/θ/` → `/s/` substitution:

```bash
uv run pronunciation-coach analyze \
  --prompt thought-about-it \
  --fake-observed "aɪ s ɔː t ɐ b aʊ t ɪ t"
```

The command emits the versioned JSON result contract. See [plan.md](plan.md) for stage
boundaries, risks, and the acoustic feasibility gate. See
[docs/stage0.md](docs/stage0.md) for the deterministic contract and alignment rules.

Analyze a supported prompt from a local WAV/FLAC/AIFF recording after prefetching:

```bash
uv run pronunciation-coach-audio \
  --prompt thought-about-it \
  --audio recording.wav
```

The result marks words as `review`, `warning`, or `no_problem_detected`. A review or warning
is calibrated evidence, not proof of an error; no problem detected does not mean perfect
pronunciation.
The JSON includes speech timing, model revision, policy version, calibration provenance,
and limitations. See [the Stage 2 engine documentation](docs/stage2-analysis.md).

Inspect raw experimental constrained-CTC phone margins with:

```bash
uv run pronunciation-coach-audio \
  --prompt thought-about-it \
  --audio recording.wav \
  --posterior
```

Reproduce the frozen Stage 1B development experiment after caching the model and dataset:

```bash
HF_HUB_OFFLINE=1 uv run pronunciation-coach-benchmark run-ctc \
  --device cpu \
  --output /tmp/stage1b-ctc-development.json
```

The held-out runner is intentionally separate: it reads the frozen `-8.0` policy, performs
no threshold selection, and refuses the locked final-test partition.

```bash
HF_HUB_OFFLINE=1 uv run pronunciation-coach-benchmark prepare-heldout-manifests
HF_HUB_OFFLINE=1 uv run pronunciation-coach-benchmark run-ctc-heldout --device cpu
```

Reproduce the Stage 1C development calibration and its fixed held-out evaluation:

```bash
HF_HUB_OFFLINE=1 uv run pronunciation-coach-benchmark calibrate-word-errors
HF_HUB_OFFLINE=1 uv run pronunciation-coach-benchmark run-calibrated-heldout --device cpu
```

Run the small development-only intentional substitution benchmark after recording its six
local audio files:

```bash
HF_HUB_OFFLINE=1 uv run pronunciation-coach-benchmark run-intentional-errors \
  --device cpu
```

It reports target-word localization, exact localization, wrong-word-only warnings, and
correct-control false warnings. See
[the intentional-error benchmark guide](docs/intentional-error-benchmark.md).
