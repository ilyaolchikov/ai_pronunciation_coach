# AI Pronunciation Coach

A local-first English read-aloud coach. Stage 2 now provides a minimum end-to-end engine:
bounded audio loading, speech-region detection, pinned local inference, and calibrated
word-level review probabilities. The deterministic Stage 0 kernel and all feasibility
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

The result marks words as `review` or `no_problem_detected`. A review is calibrated
evidence, not proof of an error; no problem detected does not mean perfect pronunciation.
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
