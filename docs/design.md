# Design

## Overview

PolyScribe is a Cargo workspace CLI that orchestrates:

- `polyscribe-cli` (`crates/polyscribe-cli`): CLI parsing (`cli.rs`), command orchestration (`main.rs`), the `transcribe` pipeline (`transcribe.rs`), and stdout modes (`output.rs`).
- `polyscribe-core` (`crates/polyscribe-core`): the engine —
  - `backend.rs`: backend selection, the `TranscribeBackend` implementations (CPU/CUDA/HIP via whisper-rs) and the reconciliation of the windows,
  - `chunked.rs`: the incremental decoding of a media file into windows of audio,
  - `models.rs`: model discovery, manifest caching, downloads, and the interactive downloader,
  - `model_manager.rs`: `ModelManager` (manifest, cache, ranged concurrent downloads, SHA-256 verification, gc/rm/verify),
  - `output.rs`: transcript types helpers, sorting/id reassignment, and JSON/TOML/SRT serialization,
  - `lib.rs`: runtime flags, logging macros, stderr silencing, model discovery, and ffmpeg audio decoding,
  - `ui.rs` / `ui/progress.rs`: cliclack-based interactive UI (prompts, spinners, progress bars),
  - `config.rs` / `error.rs`: env-var configuration and the core error type.
- `polyscribe-host` / `polyscribe-protocol` (`crates/`): plugin discovery/process management and the PSP/1 protocol types.
- `plugins/polyscribe-plugin-tubescribe`: reference plugin.

## Data flow (transcribe)

1) CLI collects inputs (media or JSON), options (merge, speaker names, language, backend), and output path.
2) For media, audio is streamed from ffmpeg and decoded to 16 kHz mono PCM f32 in memory.
3) A Whisper model is located (`WHISPER_MODEL`, or the largest `*.bin` in the models dir).
4) The selected backend transcribes via whisper-rs, producing segments.
5) Segments are sorted, reindexed, and written to JSON + TOML + SRT (or JSON to stdout).

### Long recordings

The audio of a media file is decoded incrementally by ffmpeg and handed to
the transcriber in windows of 30 seconds, the amount of audio whisper's
encoder takes at once. Only the window is in memory, so the memory a
transcription needs does not grow with the length of the recording. The
window boundaries, the reconciliation of the segments and the measurements
are described in `docs/benchmarks/chunked-transcription.md`.

### Missing speech at a window boundary

whisper does not always report a segment for the beginning of the audio it is
given: for some positions it reports its first segment many seconds into the
window, and for others it reports nothing at all. The behaviour depends on
where the window starts, not on the model, the threads or the context of the
previous window.

The pipeline notices such a region and decodes it again from a slightly
different position, because a window that starts a fraction of a second
earlier reports the speech that the first one did not. The attempts are
bounded, and the audio in front of the window is kept for two seconds so that
a window can start before the region. A region that carries speech and that
none of the attempts recovers is reported in a warning on stderr, with the
file and the time range.

The check that decides whether a region carries speech is a level and a share
of frames, not a voice activity detector, so the pipeline cannot prove that
every word of a recording is in the transcript. `docs/benchmarks/chunked-transcription.md`
has the measurements and the limits.

### Streaming input

The file path decodes a media file with ffmpeg and hands the audio to
whisper in windows of 30 seconds. A live source cannot be a file, so
`polyscribe-core::stream` provides the way in:

* `AudioSource` delivers **decoded** samples: 16 kHz, mono, f32. The
  pipeline does not resample; a source that captures at another rate has
  to convert before it delivers.
* The samples are **pulled**, never pushed: `next_block` is called when
  the pipeline needs more audio. That is the backpressure. A slow
  consumer slows the producer down, and the audio that waits in the
  pipeline is bounded by one window (30 s) plus the lookback (2 s).
* The end of the stream is **explicit**: `next_block` returns `None`.
* A source **error** ends the run and is reported to the caller.
* `CancellationToken` stops the run at the next safe boundary, which is
  before the next window is submitted for inference. A whisper call that
  is already running is not interrupted.

`StreamedAudio` turns a source into the same windows the file path
produces, and the window loop in `backend.rs` is shared by both paths.
There is one window assembler, one reconciliation and one recovery; the
file path keeps its behaviour.

#### The lifecycle of a streaming transcription

```text
  samples received  ->  window submitted  ->  segment finalized
  (source)              (inference)           (reconciliation)
```

* **Samples received** are audio that arrived from the source. They are
  not transcribed yet.
* **A window submitted** is a block of 30 seconds that was handed to
  whisper. The windows are numbered in the order they were submitted.
* **A segment** belongs to a window. The last segment of a window can
  be cut short by the end of the window, so it stays *provisional* until
  the next window confirms it. Only finalized segments reach the caller:
  a provisional segment is never presented as final.
* **Stream completion** means the source reached the end and the last
  window was flushed. A cancelled run reports the cancellation instead,
  and keeps only the segments that were finalized.

The events of a run are reported through `StreamEvent`: the samples that
arrived, the windows that were submitted, the segments that became final,
and either the completion or the cancellation of the run. The per-window
trace (`POLYSCRIBE_WINDOW_TRACE=1`) stays opt-in and separate from them.

#### The language

`Language::Auto` detects the language on the first window and pins it
for the windows that follow, because the detector is not stable enough
to run on every window and a language that changes from window to
window is worse than one that is slightly wrong. `Language::Explicit`
uses the language it is given, which is what the command line does
today. The command line still requires `--language`; the default changes
with the workflow that brings the automatic detection to the user.

#### What a microphone source has to do

A microphone source implements `AudioSource`: it captures 16 kHz mono
PCM, delivers it in blocks, and returns `None` when the capture ends.
The recording, the device selection and the hotkey belong to that
source, not to the pipeline. The source is the only part that has to
know about PipeWire.

#### Known limitations

* The windows are 30 seconds long, so the first finalized segment of a
  live stream arrives after the second window. A live workflow that
  needs a shorter latency needs a shorter window, which is a separate
  decision because the window size also decides the overlap and the
  recovery behaviour.
* The stream is transcribed in windows, so the transcript of a live
  recording is complete up to the last finalized segment, not up to the
  last sample.
* The cancellation is cooperative. A whisper call that is running is
  not interrupted, so the cancellation takes effect after it returns.

### Model context lifecycle

The backend owns a session that holds the loaded Whisper model. The model is
located and loaded on the first media input of an invocation and then reused
for every following input, so a batch of N files loads the model once instead
of N times. On a GPU build that means one upload of the weights into VRAM
instead of N.

Each input file gets its own inference state, created from the shared model.
The state is what carries the decoder state of one file, so the outputs,
timestamps and error handling of the files stay independent. The model and
the states are released when the invocation ends; nothing is cached across
invocations.

### Backend selection

`select_backend` runs once per invocation, before any model is loaded:

- The GPU backends compiled into the binary are known from the cargo
  features; the devices that are actually present come from the ggml backend
  registry, the same registry whisper.cpp picks its GPU from.
- `cpu` always runs on the CPU.
- `cuda` and `hip` require both the feature and a matching device, and fail
  with an actionable error otherwise. A GPU request is never silently
  continued on the CPU.
- `auto` prefers CUDA, then HIP, and falls back to the CPU when no device is
  available, which it reports.

A device being present means the runtime found it. It does not claim that
every kernel of a transcription ran on the GPU; `-vv` shows the device that
ggml reported and the backend whisper.cpp selected.

## Data flow (models)

1) The Hugging Face manifest is fetched (API, with an on-disk ETag/TTL cache and an HTML-scrape fallback), enriched via HEAD requests.
2) `ModelManager` downloads files with concurrent ranged GETs (streaming fallback), verifies SHA-256, and renames `.part` into place atomically.
3) The manifest (`models.json`) tracks aliases, revisions, hashes, and sizes; `ls/rm/verify/update/gc` operate on it.

## Data flow (plugins)

1) The host discovers `polyscribe-plugin-*` executables on `PATH` and in the XDG plugin dir.
2) `plugins run` spawns `<plugin> run`, writes one JSON-RPC request line to its stdin, and forwards its NDJSON stdout stream.
3) `<plugin> info` prints capabilities JSON.

## Key decisions

- Local-first: network access only for the `models` subcommands (Hugging Face over HTTPS, rustls). Everything else runs fully offline.
- GPU selection is a build-time property: a binary built with `gpu-cuda`/`gpu-hip` links the corresponding runtime and can use it; without the feature the backend is unavailable and requesting it produces an actionable error. Runtime dlopen probing was tried and removed (unstable, never detected anything). A Vulkan backend is not wired up.
- Speaker labels are manual (per-input prompts). There is no automatic speaker diarization.
- Logging: `elog!`/`ilog!`/`dlog!` macros plus `ui::*` helpers; quiet mode suppresses info/success/progress, errors and warnings remain. stderr is used for diagnostics, stdout for results.
- Model integrity: downloads verify SHA-256 when the hash is known and use atomic `.part` → final renames; temp files are guarded against leaks.
- Path safety: model file names from remote listings/manifests are validated (no path separators in `ModelManager` cache names; no absolute paths or `..` components for manifest file paths).

## Extensibility

- New transcription engines: implement `TranscribeBackend` and register them in `select_backend` (`backend.rs`).
- New output formats: extend `output.rs` (`render_srt` is the pattern for text formats).
- New model sources: extend `models.rs` / `model_manager.rs` (the `HttpClient` trait makes the manager testable with stub clients).
- New plugins: implement the PSP/1 interface (`info` + `run`); see `plugins/polyscribe-plugin-tubescribe`.
- Packaging: respect XDG dirs; allow `POLYSCRIBE_*` overrides; avoid hard-coding system paths.

## Binary naming and CLI surface

- Binary is `polyscribe`; subcommands are `transcribe`, `models`, `plugins`, `completions`, `man`.
- Keep CLI flags stable and documented; add new flags conservatively.
