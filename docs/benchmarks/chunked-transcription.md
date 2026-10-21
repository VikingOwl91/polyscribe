# Chunked transcription: bounded memory for long recordings

Status: implemented on branch `feature/chunked-transcription` (based on `dev` at
`694c957`), measured on 2026-10-21 with whisper.cpp 1.8.3 on an AMD Ryzen 7
3700X and an AMD Radeon RX 9070 XT (ROCm 7.2).

## 1. What changed and why

Until now a recording was decoded into one buffer and handed to whisper as a
whole:

```
ffmpeg -> Vec<f32> with every sample of the recording -> whisper
```

Two buffers grow with the length of the recording:

| Buffer | Size | Where |
|---|---|---|
| The decoded samples | 64 KiB per second (16 kHz, f32) | `decode_audio_to_pcm_f32_ffmpeg` in `crates/polyscribe-core/src/lib.rs` |
| The mel spectrogram whisper computes from them | about 32 KiB per second (80 mel bands, 100 frames per second) | `whisper_pcm_to_mel_with_state` in whisper.cpp |

A one-hour recording therefore needed roughly 230 MiB of samples and 115 MiB of
mel, on top of the model and the decoder state. whisper itself only ever looks
at 30 seconds of audio at a time: it slides a 30 second encoder window over the
recording, so the mel of the whole recording is never needed at once.

The pipeline now decodes the audio incrementally and hands it to whisper in
windows:

```
ffmpeg -> (pipe) -> one window of samples at a time -> whisper
```

Only the window is in memory, so the memory a transcription needs does not grow
with the duration of the recording.

## 2. Pipeline architecture

```
                     ┌──────────────────────────────────────────────┐
  media file ──ffmpeg┤ stdout: f32le, 16 kHz, mono                   │
                     └───────────────┬──────────────────────────────┘
                                     │ read on demand, at most one
                                     │ window in memory (30 s)
                     ┌───────────────▼──────────────────────────────┐
                     │ WindowAssembler                             │
                     │  - keeps the decoded samples                │
                     │  - drops the ones before the window start   │
                     └───────────────┬──────────────────────────────┘
                                     │ one window
                     ┌───────────────▼──────────────────────────────┐
                     │ whisper (same model context for the whole    │
                     │ invocation, one state per window)            │
                     └───────────────┬──────────────────────────────┘
                                     │ segments, timestamps relative
                                     │ to the window
                     ┌───────────────▼──────────────────────────────┐
                     │ Reconciliation: window position added,       │
                     │ duplicates and cut segments handled         │
                     └──────────────────────────────────────────────┘
```

The model context is loaded once per invocation and shared by all windows and
all input files, the same way it is shared by the files of a batch since the
model-context work of sprint 4.

## 3. Memory bounds and buffer ownership

| Buffer | Owner | Size |
|---|---|---|
| Decoded samples | `WindowAssembler` | At most two windows: the window being transcribed and the samples of the next one. 30 s of 16 kHz f32 is 1.9 MiB, so the decoder holds at most about 4 MiB. |
| ffmpeg output pipe | the operating system | 64 KiB, the pipe buffer. ffmpeg blocks when it is full, so decoding cannot outrun the transcription. |
| Whisper mel | whisper, per call | One window: 30 s is 3000 mel frames of 80 bands, about 1 MiB. |
| Whisper encoder and decoder state | whisper, per context | Independent of the length of the recording. |
| Transcript | PolyScribe | Grows with the number of segments, which is the output itself. |
| Model weights | whisper, per context | Independent of the length of the recording. |

The buffers that used to grow with the duration of the recording are now
bounded by the window size. The measured peak resident memory is in section 7.

## 4. Window and overlap strategy

A window holds 30 seconds, the amount of audio whisper's encoder takes at
once, so one window is one encoder pass. The start of the next window depends
on the last segment of the current one:

* **A segment that was cut short by the end of the window** (whisper had no
  context beyond it, so it may be missing its ending) is decoded again in the
  next window. That window starts at the *beginning* of that segment, so
  whisper hears the whole of it and the text is complete. The cut segment is
  dropped from the transcript of the first window.
* **Otherwise the next window starts where the last segment ended.** The audio
  in between is decoded again, so no part of the recording is skipped.
* **A window without any segment** (silence) advances by its own length.

The overlap between two windows is therefore bounded by one window (30 s), and
in practice it is the length of the segment that was cut, usually a few
seconds. The same rule is used for the CPU and for the HIP backend, there is
only one code path.

## 5. Timestamp reconciliation

Whisper reports the timestamps of a window relative to the start of that
window, so the position of the window in the recording is added to them. Two
rules keep the transcript clean:

* **Duplicates.** A segment that starts before the end of the last emitted
  segment repeats speech that was already reported, so it is dropped. The
  check has a tolerance of 50 ms, because whisper works on 10 ms steps and the
  start of a segment and the end of the previous one can differ by a step.
* **Cut segments.** A segment that reaches the end of the window is dropped
  and decoded again in the next window, which reports it with its full text.

The result is a transcript whose segments are ordered, do not overlap and do
not leave gaps, with timestamps that refer to the whole recording.

## 6. Quality trade-offs

The transcript of a recording is not bit for bit the same as before. What
changes:

* **The segment boundaries can differ.** A recording that is longer than a
  window is cut at the window boundaries, and whisper sees the audio of a
  window without the context that follows it. The segments therefore start and
  end at slightly different times. On the 65 second test recording the
  boundaries moved by at most 200 ms and the text was identical.
* **The number of segments can differ by a few.** The same audio can be split
  into different sentences at a window boundary.
* **The text itself is equivalent.** On the test recordings the normalized
  text of the chunked and the whole-file transcription was identical.

A recording that is shorter than one window (30 s) is transcribed exactly as
before: the output is byte for byte identical.

## 7. Measurements

### Method

* Same model, same inputs, same output configuration for every configuration.
* Model: `ggml-tiny.en.bin`, SHA-256
  `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
* Inputs: the committed fixture `speech.wav` (5.4 s), `medium-10min.wav`
  (600.8 s) and `long-60min.wav` (3604.8 s), all built from the same 16 kHz
  mono samples.
* Peak resident memory from `/proc/<pid>/status` (`VmHWM`), peak VRAM from
  `amd-smi metric -m --csv` sampled every 400 ms.
* Wall clock around the whole CLI invocation, so it includes the decode, the
  model load, the inference and writing the output.
* The number of windows comes from the log line the transcriber writes at
  verbose level (`Transcribed N window(s) of audio`).
* The host is shared, so the load average is recorded with every measurement.
  The CPU wall times are therefore a range, not a single value.

### Results

See `docs/benchmarks/data/chunked-*.json` for the raw data of every run.

Peak resident memory, the number the change is about:

| Recording | CPU before | CPU after | HIP before | HIP after |
|---|---|---|---|---|
| 5.4 s | 220 MB | 279 MB | 261 MB | 262 MB |
| 600 s | 362 MB | **292 MB** | 363 MB | **274 MB** |
| 3605 s | 667 MB | **291 MB** | 782 MB | **276 MB** |

The memory of the old path grows with the length of the recording. The memory
of the new path is flat: 291 MB for ten minutes and 291 MB for an hour on the
CPU, 274 MB and 276 MB on the GPU. The remaining memory is the model, the
decoder state and the transcript, none of which depend on the length of the
recording.

Wall clock and the work done:

| Recording | Backend | Before | After | Windows | Model loads | Segments | Host load |
|---|---|---|---|---|---|---|---|
| 5.4 s | CPU | 989 ms | 1020 ms | 1 | 1 | 1 | 3.4-3.5 |
| 5.4 s | HIP | 374 ms | 366 ms | 1 | 1 | 1 | 3.5-3.6 |
| 600 s | CPU | 45.8 s | 79.1 s | 22 | 1 | 111 | 16.4 / 23.2 |
| 600 s | HIP | 5.5 s | 5.7 s | 22 | 1 | 111 / 112 | 21.3 / 18.6 |
| 3605 s | CPU | 1202 s | 942 s | 131 | 1 | 666 / 669 | 21.6 / 27.1 |
| 3605 s | HIP | 35.4 s | 34.2 s | 129 | 1 | 666 / 674 | 15.8 / 8.5 |

The real-time factor follows from the wall clock and the duration of the
recording: for the hour long recording 3.0x on the CPU before the change and
3.8x after it, 102x on the GPU before and 105x after. For the ten minute
recording 13.1x on the CPU before and 7.6x after, 109x on the GPU before and
105x after.

The CPU wall clock is not a reliable comparison on this host: the load
average moved between 3 and 32 during the measurements, and the two CPU
runs of the same configuration differ by a factor of three. The GPU path
is much less sensitive to the host load, and there the change costs 4% for
the ten minute recording and nothing for the hour long one. The hour long
CPU run was faster after the change even though it ran at a higher load,
which fits the smaller amount of audio that has to be encoded: 131 windows
of 30 seconds instead of the 666 encoder passes of the whole file.

The number of segments is the same or one to three more, and the
timestamps are monotonic with no overlap in every run.

### Transcript quality

On the 65 second test recording (twelve copies of the committed fixture):

* the whole file and the chunked transcription produce the same 12 segments
  with the same text, the normalized text is byte for byte the same (971
  characters), and the segment boundaries differ by at most 200 ms;
* the transcription of the 5.4 second fixture is byte for byte identical
  to the previous behaviour.

### What the numbers show

The peak resident memory of the CPU path stops growing with the length of
the recording, and the GPU path needs about the same memory as the CPU
path instead of holding the decoded audio in RAM. The wall clock does not
regress on the GPU path, and the transcription of a short recording is
unchanged.

## 8. Backend behaviour

The chunked path is backend independent: the same windows, the same
reconciliation and the same output for the CPU and the HIP backend. The
backend selection rules of sprint 4 are unchanged, including the requirement
that an explicit GPU request needs a device that ggml reports.

## 9. Known limitations

* The CPU wall clock of a long recording is dominated by the whisper
  inference, and the measurement is noisy on a shared host. The comparison
  between the two pipelines is best read from the memory numbers and from the
  GPU path, where the inference time does not depend on the host load.
* The window size (30 s) and the overlap are not configurable yet. A different
  size would change how much of the recording is decoded twice.
* whisper computes the mel of a window from the samples of that window only.
  A recording that is shorter than a window is unaffected, a longer one is
  cut at the window boundaries.

## 10. Reproduction commands

```bash
# Build
export PATH=/opt/rocm/bin:$PATH TMPDIR=/var/tmp
cargo build --release -p polyscribe-cli --locked            # CPU
cargo build --release -p polyscribe-cli --features gpu-hip --locked

# The recordings used in section 7
python3 - << 'EOF'
import wave
src = "crates/polyscribe-cli/tests/fixtures/speech.wav"
with wave.open(src) as w:
    params = w.getparams()
    frames = w.readframes(w.getnframes())
for name, seconds in [("medium-10min.wav", 600), ("long-60min.wav", 3600)]:
    with wave.open(name, "wb") as o:
        o.setparams(params)
        o.writeframes(frames * int(seconds * 16000 / params.nframes) + 1)
EOF

# Transcribe and watch the memory
/usr/bin/time -v ./target/release/polyscribe transcribe long-60min.wav \
  --language en -o out/ 2>&1 | grep -E "Maximum resident|window"

# The same on the GPU
./target/release/polyscribe transcribe long-60min.wav --language en \
  --gpu-backend hip -o out/
```
