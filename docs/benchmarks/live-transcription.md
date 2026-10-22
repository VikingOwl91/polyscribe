# Live transcription windows

Status: initial measurement, 2026-10-22. The five-second default remains a
provisional latency choice, not a quality-validated default.

## Method

The ignored `integration_live::measure_the_live_window` test composes a
30-second, 16 kHz mono recording from five committed English speech clips.
It inserts half-second pauses and shifts the first clip by 0, 0.41, or 0.82
seconds for three phases. The reference contains the text of each complete
clip; a clip cut off by the end of the recording is excluded. WER is word-level
Levenshtein distance divided by the number of reference words. The test reports
the paced first-finalized time, fast-run wall time, duplicate segments, and
the combined process peak RSS after both runs. Fast-run wall time includes
model loading, so it is not a steady-state decode benchmark. The RSS value is
not measured separately for the paced and fast runs.

The quality run used `ggml-large-v3-turbo-q8_0.bin` (SHA-256
`317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1`) with
English explicitly selected. It ran once per window and phase on an AMD Ryzen
7 3700X. The results are in
[`data/live-transcription.json`](data/live-transcription.json).

## Quality

The word errors and the reference words of a row are the sums of the three
phase runs, and the WER is their quotient.

| Policy | Window | Word errors / reference words | WER |
|---|---:|---:|---:|
| Batch | 30 s | 36 / 193 | 0.187 |
| Live | 10 s | 47 / 193 | 0.244 |
| Live | 8 s | 49 / 193 | 0.254 |
| Live | 5 s | 66 / 193 | 0.342 |
| Live | 3 s | 68 / 193 | 0.352 |

On this fixture, shorter windows lose more words than the 30-second batch
window. Ten seconds was closer to batch quality than five seconds. This is a
small synthetic English corpus with three phase shifts, so it does not predict
quality on natural speech or other languages. It does show that the five-second
default is not yet justified by quality measurements. The full CPU sweep was
rerun after the live filter changed from a check of the whole window to a check
of the energy in the audio interval of each segment. The per-phase word-error
counts did not change: no window of this fixture holds speech that a
window-level check would have dropped.

## Latency and inference cost

The CPU timings from the quality sweep are not useful: they varied widely while
the machine was running other CPU-heavy work. They are not used to choose a
default.

A separate HIP measurement used the same model on an AMD Radeon RX 9070 XT with
ROCm 7.2.4. It selected English explicitly, so the numbers below do not include
the longer startup window used for automatic language detection.

| Window | First finalized segment | Fast-run wall for 30 s | Whisper calls | Peak RSS, both runs |
|---:|---:|---:|---:|---:|
| 10 s | 10.34 s | 1.49 s | 4 | 388.3 MB |
| 5 s | 5.29 s | 2.42 s | 9 | 387.4 MB |

The five-second window returned text sooner on this GPU, but used more than
twice as many Whisper calls. Automatic language detection uses a longer startup
window, so these first-segment times are lower bounds for the default command.
The CPU/GPU ratio also depends on the model, backend and machine. The word
errors of the two runs (13 and 19 of 67 reference words) are the same as the
ones of the CPU runs of the same window and phase.

## Reproduce

Run a window and phase with a local model:

```bash
POLYSCRIBE_E2E_MODEL=/path/to/ggml-large-v3-turbo-q8_0.bin \
POLYSCRIBE_LIVE_POLICY=live \
POLYSCRIBE_LIVE_WINDOW_SECONDS=5 \
POLYSCRIBE_LIVE_PHASE=0 \
cargo test --release --locked -p polyscribe-cli --test integration_live \
  measure_the_live_window -- --ignored --nocapture
```

Use `POLYSCRIBE_LIVE_POLICY=batch` for the 30-second baseline. Set
`POLYSCRIBE_LIVE_PHASE` to `0`, `1`, or `2` to change the alignment of speech
and windows. The test defaults to the CPU backend; set
`POLYSCRIBE_TEST_BACKEND=hip` in a HIP build to measure that backend.
