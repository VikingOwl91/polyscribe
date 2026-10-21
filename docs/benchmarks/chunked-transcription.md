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
  whisper hears the whole of it. The version of the first window is kept in
  the transcript until the re-decode replaces it, see section 5.
* **Otherwise the next window starts where the last segment ended.** The audio
  in between is decoded again, so no part of the recording is skipped.
* **A window advances by at least half a window (15 s).** whisper reports short
  segments for silence, so a window that started where the last segment ended
  would decode the same stretch of silence again and again, and every one of
  those windows adds a hallucinated segment. The measured cost of that was 16
  windows for a minute of silence, it is 2 now.

The minimum advance is not only a matter of speed. A window that starts where
the last segment ended starts in the middle of the speech, so whisper hears
the speech behind the boundary without the context in front of it and
recognizes it worse. Section 6 measures the effect: with the minimum advance
the transcription of the speech at a window boundary is as good as the one of
the implementation that hands the whole recording to whisper.

The overlap between two windows is therefore bounded by one window (30 s), and
in practice it is the length of the segment that was cut, usually a few
seconds. The same rule is used for the CPU and for the HIP backend, there is
only one code path.

## 5. Timestamp reconciliation

Whisper reports the timestamps of a window relative to the start of that
window, so the position of the window in the recording is added to them. Four
rules keep the transcript clean:

* **Duplicates.** A segment that starts before the end of the last emitted
  segment repeats speech that was already reported, so it is dropped. The
  check has a tolerance of 50 ms, because whisper works on 10 ms steps and the
  start of a segment and the end of the previous one can differ by a step.
* **Provisional segments.** The last segment of a window may have been cut
  short by the end of that window. It is kept, but marked as provisional. The
  next window decodes it again and the re-decode *replaces* it, because the
  cut version can be missing words. Dropping the cut version instead would
  lose the words it did hear when the re-decode starts a little before it and
  is then mistaken for a duplicate.
* **Segments without a duration.** whisper reports such a segment at the end
  of a window, with the text of the speech around it. It carries no time, so
  it is dropped: it would show up as a subtitle without a duration, on top of
  the segment next to it.
* **A window without any segment** (silence) advances by its own length.

The result is a transcript whose segments are ordered, do not overlap and do
not leave gaps, with timestamps that refer to the whole recording.

## 6. Boundary quality

The chunked path has to reconcile the segments of the windows with each other.
A wrong reconciliation shows up as missing words, duplicated words or text that
is out of order, so the boundary handling is measured with speech that crosses
a window boundary.

### Fixtures

The fixtures are built from five short speech clips that are committed to the
repository, `crates/polyscribe-cli/tests/fixtures/boundary/`. The clips were
generated with espeak-ng 1.51 (`en-us`, speed 140) and resampled to 16 kHz
mono with ffmpeg. espeak-ng is free software under the GNU General Public
License, the generated audio is test data. The text of every clip is known,
so it is the reference transcript of that clip.

The case definitions in `crates/polyscribe-cli/tests/fixtures/boundary/cases.json`
place the clips in a recording of silence, so that the speech sits at the point
where the pipeline cuts the recording into windows:

| Case | Speech | What it exercises |
|---|---|---|
| `straddle` | 28.0 s | A sentence that starts before the 30 second boundary and ends after it. |
| `ends-at-boundary` | 26.35 s | Speech that ends exactly at the boundary. |
| `starts-at-boundary` | 30.0 s | Speech that starts exactly at the boundary. |
| `continuous-across-boundary` | 27.0 s, 30.65 s | Two sentences back to back, so the speech continues across the boundary. |
| `long-utterance` | 24.0 s, 28.54 s, 33.08 s | One long utterance without a pause, so it is cut by the boundary. |
| `repeated-words` | 27.0 s, 30.65 s, 34.30 s | The same sentence three times around the boundary. |
| `silence-then-speech` | 31.0 s | The first window is silent, the speech starts in the second one. |
| `partial-final-window` | 35.0 s | Speech at the end of the recording, so the last window is shorter than the others. |
| `multiple-boundaries` | 28.0 s, 55.0 s, 85.0 s | Speech around three window boundaries. |

The fixtures are composed from the clips at run time, so the repository holds
half a megabyte of audio instead of megabytes of silence.

### Method

`scripts/benchmarks/boundary_quality.py` composes every case, transcribes it
with two binaries and compares the result with the reference transcript:

* the **chunked** binary is the current implementation;
* the **whole-file** binary is the implementation before the chunked pipeline,
  which hands the whole recording to whisper in one buffer.

The difference between the two is the error of the chunk reconciliation: both
see the same audio with the same model, so every difference comes from the
window handling. The difference against the reference transcript contains both
the recognition errors of whisper and the errors of the reconciliation, and is
reported as the word error rate.

Model `ggml-base.en.bin`, SHA-256 `a03779c8…6d002`, CPU and HIP backend, the
raw data is in `docs/benchmarks/data/boundary-quality-cpu.json` and
`docs/benchmarks/data/boundary-quality-hip.json`.

### Result

The word error rate against the reference transcript, and the difference of the
error count between the chunked and the whole-file transcription. The
difference is what the window handling costs; a positive number means that the
chunked transcription is worse than the one of the previous, not chunked
implementation.

| Case | Words | WER chunked | WER whole file | Difference | Before the fix |
|---|---|---|---|---|---|
| `straddle` | 9 | 0.222 | 0.222 | 0 | 0 |
| `ends-at-boundary` | 9 | 0.222 | 0.222 | +1 | 0 |
| `starts-at-boundary` | 9 | 0.111 | 0.222 | −1 | +4 |
| `continuous-across-boundary` | 18 | 0.111 | 0.111 | 0 | 0 |
| `long-utterance` | 30 | 0.500 | 0.833 | −3 | −3 |
| `repeated-words` | 27 | 0.000 | 0.000 | 0 | 0 |
| `silence-then-speech` | 8 | 0.500 | 0.875 | −3 | −3 |
| `partial-final-window` | 9 | 0.111 | 0.111 | 0 | +3 |
| `multiple-boundaries` | 26 | 0.308 | 0.346 | −1 | −1 |

The chunked transcription is better than the whole-file transcription in four
of the nine cases, equal in four, and worse in one case by a single word. The
CPU and the HIP backend produce the same transcripts.

The word error rate itself is high on some cases, because the reference is
synthetic speech and whisper replaces words by similar sounding ones: "she
sells sea shells by the sea shore" becomes "she's helmsy she'll bite my sea
shore". That happens with the chunked and with the whole-file implementation
alike, so it is a property of the model and the synthetic voice, not of the
window handling.

### Defects the fixtures found

Four defects were reproduced with these fixtures and fixed. The word counts
below are the difference between the chunked and the whole-file error count on
the fixture in question, before the fix.

1. **The speech at a window boundary was recognized worse** (`starts-at-boundary`,
   +4 words, `partial-final-window`, +3 words). The window started where the
   last segment ended, so whisper heard the speech behind the boundary without
   the context in front of it, and it recognized it with more errors. Fixed
   with the minimum advance of half a window: the window then starts in the
   silence in front of the speech, and the result is as good as the one of the
   whole-file implementation.
2. **A cut segment could lose its words** (`repeated-words`). The cut segment
   of a window was dropped and decoded again in the next window. The re-decode
   can start a little before the dropped segment, and was then mistaken for a
   duplicate and dropped as well, so the whole sentence was lost. Fixed by
   keeping the cut segment until the re-decode replaces it.
3. **Silence was decoded over and over** (60 seconds of silence). A window
   started where the last segment ended, and whisper reports short segments for
   silence, so the same silence was decoded again and again: 16 windows for a
   minute of silence, and one hallucination per window. Fixed with the minimum
   advance of half a window, the same minute now takes 2 windows.
4. **A segment without a duration ended up in the transcript.** whisper
   reports such a segment at the end of a window, with the text of the speech
   around it. It showed up as a subtitle without a duration, on top of the
   segment next to it. Found on the hour long recording, where 12 of the 96
   windows reported one. Fixed by dropping segments without a duration.

### What was tried and rejected

A window that holds 30 seconds of new audio and 10 seconds of the audio
behind it was measured as well. whisper slides its encoder window over the
mel it is given, so a longer window gives the speech at the boundary real
frames behind it, and the transcription of the boundary cases was the same as
the one of the whole-file implementation in all nine cases.

It was rejected because it changes where the windows start, and that moved
the pipeline onto a position where whisper skips the beginning of a window:
on the hour long recording, which repeats the same sentence, 22 seconds of
speech were missing from the transcript. A missing sentence is worse than a
misrecognized word, so the window stays at 30 seconds.

### Regression tests

`crates/polyscribe-cli/tests/integration_boundary.rs` runs the fixtures through
the CLI and checks that every stretch of speech is covered by a segment, that
the timestamps are ordered and do not overlap, and that the transcript holds
most of the words of the reference. It compares the words and not the
transcript character by character, because a wrong word is a recognition error
and not a defect of the window handling. A second test checks that a minute of
silence does not need more than four windows.

The tests need a Whisper model and skip themselves without one, like the other
end-to-end tests.

## 7. The recovery of a region that whisper did not transcribe

whisper does not always report a segment for the beginning of the audio it
is given. For some positions it reports its first segment many seconds into
the window, and for others it reports nothing at all. The pipeline advanced to
the last segment of the window, so the audio in front of it was never
transcribed and never offered again.

### How it happens

The behavior depends on the position of the window and on nothing else. The
same window, decoded again and again, gives the same result, with any number
of threads and with or without the context of the previous window. A window
that starts a fraction of a second earlier reports the speech that the first
one did not. The test
`crates/polyscribe-core/tests/whisper_decode_alignment.rs` measures it:

```
the window at 0 s: (22.08, 1), a quarter of a second later: (0.0, 5)
```

The first number is the position of the first segment and the second the
number of segments: the window at 0 s reports one segment, 22.08 s into it,
and the window a quarter of a second later reports five segments, the first
one at its beginning.

On the long recording of the benchmark, the window that started at 77.80 s
left 22.08 s untranscribed, and the whole-file transcription did not: its mel
is continuous over the whole recording, so the decoder always has the text it
has produced as its context.

### What the pipeline does

1. **The region is noticed.** A window whose first segment starts more than a
   second after the window begins has a region that whisper did not report.
   The region starts at the end of the transcribed timeline and ends where the
   first segment begins.
2. **Silence is not retried.** The share of the frames of the region that
   carry speech level energy decides: a region of silence that ends in the
   first words of a segment is silence, not speech that was missed. The level
   of the whole region is not enough for that, because the region of a window
   that starts with silence holds the words of the segment that whisper
   reported.
3. **The region is decoded again.** The window is decoded again from a
   slightly different position, because the position decides whether whisper
   reports the speech. The offsets are a quarter of a second in both
   directions, then half a second, then a second. The attempt that looks back
   comes first, so that the segment that is decoded again covers the region.
   The attempts are bounded: six of them, and a region that none of them
   recovers is left as it is.
4. **The text of the region is kept.** A segment of the recovery window that
   reaches beyond the transcribed timeline is the only text for the region, so
   it is kept. Its start is moved to the end of the transcribed timeline, so
   that the segments stay ordered and do not overlap.

The samples in front of a window are kept for two seconds, so that a window
can start before the region that was not transcribed. That is 320 kB of audio
and does not depend on the length of the recording.

### What it does

The reproducer is the long recording of the benchmark, cut at 77.80 s. The
model is `ggml-tiny.en.bin`, the model of the CI, because the position that
whisper skips depends on the model.

| Implementation | WER | Missing words | Duplicated words | Windows | Recovery attempts |
|---|---|---|---|---|---|
| Whole file | 0.679 | 79 | 4 | 1 | 0 |
| Chunked, without the recovery | 0.679 | 79 | 4 | 2 | 0 |
| Chunked, with the recovery | **0.277** | **34** | 4 | 2 | 1 |

The recovery gives back the 22 seconds of speech and the transcript of the
reproducer is better than the one of the whole-file implementation. On the
nine boundary fixtures and on the recording with varied speech the recovery
does not change the result: the word error rate is the same as without it,
because no region of those recordings is skipped.

The CPU and the HIP backend give the same result: the same word error rate,
the same recovery attempt, the same text.

### What it cannot do

The recovery cannot prove that whisper recognized every word of a recording.
It can prove that a region that whisper did not report was offered to
whisper again, from a position where whisper does report speech, and that
the text of that region is in the transcript. A region that no attempt
recovers keeps its hole, and the trace reports it:

```
TRACE   hole start_sample=0 end_sample=353280 len=22.080s level=-20.5dBFS speech=1.000 action=accept
```

The trace is written to stderr when `POLYSCRIBE_WINDOW_TRACE` is set, and it
reports the sample range of every window, the segments of the window with
their window relative and global timestamps, the region that was not
transcribed and what was done about it.

## 8. Quality trade-offs

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
  text of the chunked and the whole-file transcription was identical, and on
  the boundary fixtures of section 6 the two differ in one case by one word,
  in favour of the chunked transcription.

A recording that is shorter than one window (30 s) is transcribed exactly as
before: the output is byte for byte identical.

## 9. Measurements

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

See `docs/benchmarks/data/chunked-*.json` for the raw data of every run. The
"before" runs are the implementation that hands the whole recording to
whisper in one buffer, the "after" runs the chunked pipeline.

Peak resident memory, the number the change is about:

| Recording | CPU before | CPU after | HIP before | HIP after |
|---|---|---|---|---|
| 5.4 s | 220 MB | 278 MB | 261 MB | 262 MB |
| 600 s | 362 MB | **292 MB** | 363 MB | **275 MB** |
| 3605 s | 667 MB | **292 MB** | 782 MB | **277 MB** |

The memory of the old path grows with the length of the recording. The memory
of the new path is flat: 292 MB for ten minutes and 292 MB for an hour on the
CPU, 275 MB and 277 MB on the GPU. The remaining memory is the model, the
decoder state and the transcript, none of which depend on the length of the
recording.

The "after" numbers of the table are one series of runs with the same binary
on the same machine state. A single run of the same configuration on this
host varies by up to 60 MB, the load average of the machine was between 13
and 37 during the measurements.

Wall clock and the work done:

| Recording | Backend | Before | After | Windows | Model loads | Segments | Host load |
|---|---|---|---|---|---|---|---|
| 5.4 s | CPU | 989 ms | 900 ms | 1 | 1 | 1 | 12.8 |
| 5.4 s | HIP | 374 ms | 358 ms | 1 | 1 | 1 | 3.5-3.6 |
| 600 s | CPU | 45.8 s | 57.3 s | 22 | 1 | 111 | 20.0 |
| 600 s | HIP | 5.5 s | 5.6 s | 22 | 1 | 111 / 112 | 18.9 |
| 3605 s | CPU | 1202.5 s | 481.0 s | 131 | 1 | 669 | 19.6 |
| 3605 s | HIP | 35.4 s | 32.2 s | 129 | 1 | 669 / 674 | 12.8 |

The real-time factor follows from the wall clock and the duration of the
recording: for the hour long recording 3.0x on the CPU before the change and
7.5x after it, 102x on the GPU before and 112x after. For the ten minute
recording 13.1x on the CPU before and 10.5x after, 109x on the GPU before and
107x after.

The CPU wall clock is not a reliable comparison on this host: the load average
moved between 3 and 37 during the measurements, and the "before" runs are from
an earlier session with a different load. The GPU path is much less sensitive
to the host load, and there the change costs nothing for the hour long
recording. The number of encoder passes is the reason for the difference on
the CPU: 131 windows of 30 seconds instead of the 666 encoder passes of the
whole file.

The number of segments is the same or one to five more, and the timestamps are
monotonic with no overlap in every run. On the hour long recording the
chunked transcription covers 3604.2 s of the 3604.8 s of the audio, the
whole-file transcription 3604.4 s.

### Transcript quality

On the 65 second test recording (twelve copies of the committed fixture):

* the whole file and the chunked transcription produce the same 12 segments
  with the same text, the normalized text is byte for byte the same (959
  characters), and the segment boundaries differ by at most 120 ms;
* the transcription of the 5.4 second fixture is byte for byte identical
  to the previous behaviour.

### What the numbers show

The peak resident memory of the CPU path stops growing with the length of
the recording, and the GPU path needs about the same memory as the CPU
path instead of holding the decoded audio in RAM. The wall clock does not
regress on the GPU path, and the transcription of a short recording is
unchanged.

## 10. Backend behaviour

The chunked path is backend independent: the same windows, the same
reconciliation and the same output for the CPU and the HIP backend. The
backend selection rules of sprint 4 are unchanged, including the requirement
that an explicit GPU request needs a device that ggml reports.

## 11. Known limitations

* The CPU wall clock of a long recording is dominated by the whisper
  inference, and the measurement is noisy on a shared host. The comparison
  between the two pipelines is best read from the memory numbers and from the
  GPU path, where the inference time does not depend on the host load.
* The window size (30 s) and the overlap are not configurable yet. A different
  size would change how much of the recording is decoded twice.
* whisper computes the mel of a window from the samples of that window only.
  A recording that is shorter than a window is unaffected, a longer one is
  cut at the window boundaries. The speech behind a boundary is recognized
  with the context in front of it, but without the context behind it, so it
  can be recognized worse than by the implementation that hands the whole
  recording to whisper. On the boundary fixtures of section 6 that costs one
  word in one of nine cases.
* whisper can skip the beginning of a window: on the hour long recording,
  which repeats the same sentence, it skips 22 seconds when the window starts
  at exactly 77.8 s. The pipeline decodes such a region again from a
  slightly different position, which gives the speech back (section 7). A
  region that none of the attempts recovers keeps its hole, and the trace
  reports it. The pipeline cannot prove that whisper recognized every word of
  a recording: it can prove that a region that whisper did not report was
  offered to whisper again.
* The boundary fixtures are synthetic speech. They cover the positions of the
  window boundaries, but they cannot cover the variety of a real recording.
  The reference transcript is the text of the synthesizer, so the word error
  rate against it contains the recognition errors of the model on a synthetic
  voice. Only the difference between the chunked and the whole-file
  transcription measures the window handling.
* whisper collapses a phrase that follows the same phrase directly into one
  segment and reports the text once. That happens with the chunked and with
  the whole-file implementation, but the window boundaries decide how often it
  happens, so a recording with many repetitions can lose a repetition.

## 12. Reproduction commands

```bash
# Build
export PATH=/opt/rocm/bin:$PATH TMPDIR=/var/tmp
cargo build --release -p polyscribe-cli --locked            # CPU
cargo build --release -p polyscribe-cli --features gpu-hip --locked

# The boundary fixtures of section 6
python3 scripts/benchmarks/boundary_quality.py \
  --chunked target/release/polyscribe \
  --whole-file /path/to/the/whole-file/binary \
  --model /path/to/ggml-base.en.bin \
  --backend cpu \
  --out docs/benchmarks/data/boundary-quality-cpu.json

# The same on the GPU
python3 scripts/benchmarks/boundary_quality.py \
  --chunked target/release/polyscribe \
  --whole-file /path/to/the/whole-file/binary \
  --model /path/to/ggml-base.en.bin \
  --backend hip \
  --out docs/benchmarks/data/boundary-quality-hip.json

# The boundary regression tests
POLYSCRIBE_E2E_MODEL=/path/to/ggml-base.en.bin \
  cargo test -p polyscribe-cli --test integration_boundary

# The recovery of a region that whisper did not transcribe, against the
# implementation without the recovery and the whole-file implementation
python3 scripts/benchmarks/recovery_quality.py \
  --chunked target/release/polyscribe \
  --previous /path/to/the/previous/binary \
  --whole-file /path/to/the/whole-file/binary \
  --model /path/to/ggml-tiny.en.bin \
  --backend cpu \
  --out docs/benchmarks/data/recovery-quality-tiny-cpu.json

# The window handling of a transcription, with the sample range of every
# window, the segments and the regions that whisper did not transcribe
POLYSCRIBE_WINDOW_TRACE=1 ./target/release/polyscribe transcribe recording.wav \
  --language en -o out/ 2> window-trace.log

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
