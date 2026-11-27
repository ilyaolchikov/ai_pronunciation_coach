# Pronunciada v1 release report

## Decision

**Release-ready — all frozen acceptance gates pass.**

The human-corpus, performance, automated UI navigation, and audio non-retention gates pass at
strictness `0.75`. The policy is frozen against implementation commit
`f3b18df50a609ec911f278d3f111f3f1e86ca087`.

## Configuration

- Pronunciada: `1.1.0`
- Acoustic model: `zipa-small-crctc-500k`
- Model revision: `dfe87c97662606fa6db90ad06f627d4363314744`
- Precision: `int8`
- Strictness: `0.75`
- Yellow: acceptance probability below `0.80`
- Red: acceptance probability below `0.50`

## Acceptance gates

| Gate | Required | Result | Status |
|---|---:|---:|---|
| Corpus completion | 100% | 4,859/4,859 | Pass |
| SpeechOcean yellow/red recall | ≥80% | 90.25% | Pass |
| SpeechOcean red precision | ≥40% | 43.10% | Pass |
| SpeechOcean red recall | ≥60% | 70.74% | Pass |
| L2 final-test yellow/red precision | ≥75% | 76.37% | Pass |
| L2 final-test yellow/red recall | ≥15% | 21.78% | Pass |
| UI navigation resets recording state | Required | Automated test passed | Pass |
| UI retains no application-managed audio | Required | Automated test passed | Pass |
| Performance completion | 100% | 33/33 | Pass |
| Warm inference p95 | <200 ms | 17.14 ms | Pass |
| Valid 0.25–15 second audio | No crash | Both boundaries passed | Pass |
| No network after cache | Required | No connection attempted | Pass |
| Peak memory | Reported | 520.34 MiB | Pass |

## Human corpora

SpeechOcean completed 1,260 recordings with zero failures. Yellow/red precision was 24.82%,
yellow/red recall was 90.25%, red precision was 43.10%, and red recall was 70.74%.

L2-ARCTIC completed all 3,599 recordings with zero failures. On the 900-recording final-test
partition, yellow/red precision was 76.37%, yellow/red recall was 21.78%, red precision was
83.80%, and red recall was 10.08%. The detailed report contains partition, first-language,
gender, and SpeechOcean age-group results.

## Performance and checks

- Cold cached model load: 448.06 ms
- First inference: 59.12 ms
- Warm inference p50/p95: 16.50/17.14 ms
- Peak process memory: 520.34 MiB
- Ruff, formatting, and Pyright: passed
- Tests: 43 passed, 1 model-gated test skipped in the ordinary run
- Cached real-model integration test: passed separately
- UI navigation reset and audio non-retention tests: passed

The performance reference is human L2-ARCTIC audio scored against a fixed product prompt. It is
used only for systems timing and reliability; its scores are not accuracy evidence.

## Reproduction

```bash
UV_CACHE_DIR="$PWD/.uv-cache" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
uv run pronunciation-coach-benchmark evaluate-corpora --overwrite

UV_CACHE_DIR="$PWD/.uv-cache" \
HF_HUB_OFFLINE=1 \
uv run pronunciation-coach-benchmark evaluate-performance \
  --audio /path/to/human-reference.wav \
  --prompt <matching-prompt-id> \
  --warm-runs 30 \
  --overwrite
```
