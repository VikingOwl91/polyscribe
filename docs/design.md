# Design

## Overview

PolyScribe is a Cargo workspace CLI that orchestrates:

- `polyscribe-cli` (`crates/polyscribe-cli`): CLI parsing (`cli.rs`), command orchestration (`main.rs`), the `transcribe` pipeline (`transcribe.rs`), the `youtube` pipeline (`youtube.rs`, `youtube_process.rs`, `youtube_cache.rs`), the `record` pipeline (`record.rs`), and stdout modes (`output.rs`).
- `polyscribe-core` (`crates/polyscribe-core`): the engine —
  - `backend.rs`: backend selection, the `TranscribeBackend` implementations (CPU/CUDA/HIP via whisper-rs) and the reconciliation of the windows,
  - `capture.rs`: the live capture from PipeWire (`PipeWireCapture`),
  - `chunked.rs`: the incremental decoding of a media file into windows of audio,
  - `models.rs`: model discovery, manifest caching, downloads, and the interactive downloader,
  - `model_manager.rs`: `ModelManager` (manifest, cache, ranged concurrent downloads, SHA-256 verification, gc/rm/verify),
  - `output.rs`: transcript types helpers, sorting/id reassignment, and JSON/TOML/SRT serialization,
  - `stream.rs`: the `AudioSource` contract, the cancellation token, and the assembly of the windows of a stream,
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

## Data flow (youtube)

1) The CLI validates the HTTPS YouTube URL and request language, then checks the versioned transcript cache unless `--refresh` was given.
2) The downloader runs with an explicit argv, isolated temporary home/cache, bounded output, a wall-clock timeout, and process-group cleanup. Metadata and supported WebVTT tracks are read from JSON.
3) A selected WebVTT file is size-checked, parsed, and normalized by the pure `polyscribe-core::youtube` functions. The cache stores source identity, original cues, normalized cues, and normalization counts, independent of TXT/SRT/JSON rendering.
4) Local audio transcription is allowed only when captions are confirmed absent or a selected caption artifact has disappeared. HTTP errors, network failures, unavailable requested languages, and malformed subtitle content remain errors.
5) Audio downloads are capped and removed with their private temporary directory. The same Whisper backend and window reconciliation used by file transcription process the downloaded audio; only this YouTube call opts into the supervised ffmpeg decoder deadline.

Cache entries live below `$XDG_CACHE_HOME/polyscribe/youtube/v1/`. Directory-relative opens refuse symlinks, cache directories must be private and user-owned, and each record/index update is fsynced and atomically renamed. The format version is part of the cache identity so a parser/normalizer change cannot reuse stale cues.

The subprocess deadline does not interrupt native Whisper inference. A stuck in-process inference therefore has no hard end-to-end timeout; providing that guarantee would require moving inference to a supervised worker process.

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
* **A window submitted** is a block of audio that was handed to
  whisper. Its length follows the policy: 30 seconds for the batch policy,
  the shorter live window for the live policy. The windows are numbered in
  the order they were submitted.
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

`Language::Auto` detects the language on the first window of the batch
policy and on the startup windows of the live policy, and pins it for the
windows that follow, because the detector is not stable enough to run on
every window. An English-only model selects English directly: language
detection with that model can return a language it cannot transcribe.
`Language::Explicit` uses the language it is given; an English-only model
rejects a non-English hint. The file path keeps its existing
explicit-language behavior.

The workflow that brings automatic language selection to the user exists
now: `polyscribe record` detects the language by default with a
multilingual model, selects English with an English-only model, and lets
`--language` override either choice. The file path still requires
`--language` and is unchanged.

#### What a microphone source has to do

A microphone source implements `AudioSource`: it captures 16 kHz mono
PCM, delivers it in blocks, and returns `None` when the capture ends.
The recording, the device selection and the hotkey belong to that
source, not to the pipeline. The source is the only part that has to
know about PipeWire. The source that ships with the pipeline is
`PipeWireCapture`, described in the section below.

#### Known limitations

* The batch policy uses windows of 30 seconds, so the first finalized
  segment of a live stream arrives after 30 seconds of audio have been
  collected and decoded, and a segment that is cut short by the end of the
  window waits for the next one. The live policy uses shorter windows, and
  the window size also decides the overlap and the recovery behaviour.
* The stream is transcribed in windows, so the transcript of a live
  recording is complete up to the last finalized segment, not up to the
  last sample.
* The cancellation is cooperative. A whisper call that is running is
  not interrupted, so the cancellation takes effect after it returns.
* Neither policy re-decodes a segment that covers a whole window: such a
  segment is complete, and a window of the same length cannot give it more
  context. A segment that is genuinely cut short by the end of the window is
  still re-decoded in the batch policy.
* The language of a live recording can only be detected on the longer
  startup windows. A recording that carries no speech in them is
  transcribed in the language that the last startup window reported,
  and a warning names that language as a guess.
* A live segment is dropped when its own timestamped audio interval holds
  less than two 20 ms frames above the speech-energy threshold. This filters
  Whisper's guessed phrases for a silent window without dropping a short
  utterance because it occupies little of a longer window.
* The live path also drops segments with a no-speech probability above 0.6.
  The batch/file path does not apply this extra filter, so it retains the
  behavior of the previous release.
* The live capture needs a running PipeWire session and the program
  `pw-record`. Without them there is no source, and the file path stays
  the way to transcribe a recording.

### Live microphone capture

`polyscribe record` records the microphone and transcribes it while the
recording runs. The capture is `PipeWireCapture`
(`crates/polyscribe-core/src/capture.rs`), an `AudioSource` of
`polyscribe_core::stream`, so the streaming path of the pipeline
consumes it like every other source. It needs a running PipeWire session
and the program `pw-record`.

#### The capture

The capture runs `pw-record` as a supervised child process. The command
is built with an explicit executable and an explicit argument list, so
nothing goes through a shell:

```text
pw-record --verbose --raw --format s16 --rate 16000 --channels 1 [--target NODE] -
```

The trailing `-` makes `pw-record` write the audio to its standard
output. `--raw` selects header-less PCM. The default WAV container is
not used, because a container needs a header that is only written when
the recording ends, and the pipeline needs the samples while the
recording runs. The samples are 16 kHz mono signed 16 bit little
endian, decoded to f32 for the pipeline (a sample is divided by 32768).

`pw-record` writes the audio to its standard output and its diagnostics
to its standard error. Two threads read them, so the capture cannot
deadlock on a full pipe. The reader thread decodes the PCM and puts
blocks of 4096 samples into a queue with 8 places. When the queue is
full the reader waits, the pipe of the child fills up and the child
waits. The memory of the capture therefore does not grow with the length
of the recording. The stderr thread keeps only the last 8 KiB of the
diagnostics, so a long recording cannot fill the memory.

`--verbose` makes `pw-record` report the format of its raw output on
stderr:

```text
raw: rate=16000 channels=1 fmt=s16 samplesize=2 stride=2
```

That line is parsed and checked against the format that was asked for.
A capture that delivers another rate, another channel count or another
format is refused with an error, instead of feeding misaligned audio
into whisper.

The child runs in its own process group, so a stop signal that reaches
the terminal reaches the run first and the capture stops the whole group
in a defined order. Normal shutdown sends SIGTERM to the group and escalates
to SIGKILL after two seconds. The direct child is reaped before the reader
threads are joined.

Linux sends the direct child a catchable SIGTERM when the thread that created
it exits (PR_SET_PDEATHSIG). This is best effort: it does not propagate to
descendants, and the child may catch or ignore it. Descendants that remain in
the process group are stopped during normal shutdown; descendants that leave
it are outside the cleanup guarantee. The configured capture program runs
with the caller's privileges and is not sandboxed. Keep its owning thread
alive while the capture is in use.

`pw-record` leaves with the exit code 1 when it is stopped by a signal.
The capture therefore does not judge the exit code on its own: it knows
whether it asked the child to stop, and only a child that failed without
being asked to stop is an error.

`pw-record` falls back to the default source when it is given a node
that does not exist, and it does not say so. A device that is asked for
is therefore checked against the nodes of the running session before the
capture starts (the list comes from `pw-cli ls Node`). An unknown device
is refused with the list of the sources of the session. `pw-record`
belongs to the Arch package `pipewire-audio`, `pw-cli` to the package
`pipewire`, and `pipewire-audio` depends on `pipewire`. The device
check therefore needs no package that the capture does not already need.

The capture cooperates with the cancellation of the run: the token of
the pipeline is shared with the capture, and the capture reports "no
audio right now" instead of blocking forever, so the pipeline notices
the cancellation.

#### The live window policy

The batch policy keeps the windows of 30 seconds and is unchanged. The
file path and the batch streaming path decode the same audio the same
way.

The live policy uses shorter windows, so that the text of a recording
appears while the recording is still running. The default live window
is 5 seconds and can be set with `--window`.

The window advances by at least half of its own length. The fixed 15
seconds of the batch policy would make a live pipeline wait for audio
that is further away than the window itself.

The first windows of a live run are longer (10 seconds by default). A
multilingual model detects the language on them, because a short window
carries too little speech for a reliable detection. An English-only model
selects English directly and skips this startup phase.

For a multilingual model, the language is only pinned when a window
carries speech. A language detected in silence is a guess, and a wrong
guess is worse than a late one. When the startup windows are used up,
the language is pinned anyway, so that the rest of the recording is
transcribed in one language, and a warning names the guess.

An explicit `--language` skips the startup phase, because the language
is known from the start.

A segment that reaches the end of its window can be cut short by it. It
stays provisional until the next window decodes it again, with the
context it needs. The live policy does not decode the same window
again: a segment that covers the whole window cannot get more context
from a window of the same length, so it is final. The batch policy
keeps the behaviour of the previous release.

The number of times a segment is decoded again is bounded, so that a
segment that stays cut short cannot cost an endless series of decodes.

The recovery of a region that whisper did not transcribe is the one of
the batch path. It is not switched off by the shorter window.

The segments of a live window are filtered by the energy of their own audio
interval. A segment is kept when its interval holds at least two 20 ms frames
above the speech-energy threshold. A window that only holds silence
contributes nothing, and a short utterance is kept even when it is a small
part of the window. A segment that whisper reports for a silence that lies
next to real speech keeps the interval of that speech and survives the
filter, because the filter cannot tell an invented segment from a real one
that carries energy.

The measured quality and latency of the live windows are in
`docs/benchmarks/live-transcription.md`. The current five-second default
has not been validated against natural speech.

#### The output of the record command

The standard output carries the transcript: one finalized segment per
line, as the segments become final. With the global `--json` flag the
lines are JSON objects, one per line, with the fields speaker, start,
end and text.

The standard error carries everything else: the readiness of the
capture, the text that is still provisional (prefixed with a "… "), and
the diagnostics of the run.

A provisional segment may still change. Only a finalized segment is
final, and it is never reported twice.

Ctrl+C stops the recording, the segments that were finalized are kept
and the exit status is 0. The stop is cooperative: a whisper call that is
running is not interrupted, so the run ends when the call returns. A
second Ctrl+C ends the run at once: the capture is stopped and the
process ends, because a user that presses the key again waits for
nothing. A capture that failed exits with a nonzero status.

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

- Binary is `polyscribe`; subcommands are `transcribe`, `record`, `models`, `plugins`, `completions`, `man`.
- Keep CLI flags stable and documented; add new flags conservatively.
