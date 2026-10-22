# PolyScribe backlog

Prioritized, realistic backlog for a small local-first CLI. Priorities:
P0 = do next, P1 = near-term, P2 = medium-term, P3 = optional / ideas.
Effort: S = hours, M = days, L = weeks.

This file distinguishes four kinds of items: **defects/debt** (things that
are wrong or fragile today), **missing functionality** (features users ask
for), **optional improvements** (nice-to-have polish), and **long-term
ideas** (directional, not scheduled).

---

## Defects and technical debt

### D1. Global stderr silencing is not thread-safe (P1, S)
- `with_suppressed_stderr`/`StderrSilencer` (core `lib.rs`) redirects the
  process-global fd 2 with unsafe libc calls around whisper model load and
  inference. It is panic-safe (catch_unwind + Drop) but not thread-safe; any
  future parallel transcription would race on it.
- AC: replace fd redirection with whisper-rs/ggml log callbacks so no global
  state is touched, then enable parallel file processing (F3).

### D3. Long-file memory usage is O(duration) (done in v0.1.0-alpha.2)
- ffmpeg output is streamed into a `Vec<f32>` (64 KiB/s of audio), but
  `whisper_full` still needs the whole sample buffer in memory (~230 MiB per
  hour of audio). True streaming/chunked transcription would cap memory.
- AC: prototype chunked transcription with a long fixture; measure peak RSS
  before/after.
- Shipped in `v0.1.0-alpha.2`: the audio is decoded and transcribed in
  windows of 30 seconds. The peak memory of an hour long recording is about
  235 MB on the CPU instead of 670 MB, and it does not grow with the length
  of the recording. See `docs/benchmarks/chunked-transcription.md`.

### D8. A transcript can be incomplete at a window boundary (P2, M)
- whisper does not always report a segment for the beginning of the audio it
  is given. The chunked pipeline decodes such a region again from a slightly
  different position, which recovers it in the recorded cases, but it cannot
  prove that every word of a recording is in the transcript: a region that
  carries speech and that no attempt recovers keeps its hole. The
  transcription warns about it on stderr.
- AC: a reliable speech presence signal, so that the pipeline can tell
  ordinary silence from speech that was missed, and so that the region is
  recovered without a blind retry. The current check is a level and a share of
  frames, not a voice activity detector.

### D9. The batch path decodes the same window again and again (P2, S)
- A segment that covers a whole window stays provisional, because it
  reaches the end of the window. The next window decodes it again, from
  the position of the segment, and the position drifts by one sample per
  attempt, until the segment is not cut short any more. The recovery
  fixture of the repository needs 12 such attempts for its first window.
- The live policy does not do this: a segment that covers the whole window
  cannot get more context from a window of the same length, so it is
  final. The batch policy keeps the behaviour of the previous release,
  because the text of the recording comes out of that search.
- The attempts are bounded now, so the pipeline cannot loop forever, but
  the bound is high enough to keep the output of the batch path.
- AC: decide whether the batch path should keep the search, and measure
  what the text of the recordings with a segment over a whole window looks
  like with and without it.

### D5. Non-Unix TTY detection (P3, S)
- `stdin_is_tty` uses `std::io::IsTerminal` (correct); the remaining gap is
  that Windows/macOS are not CI-tested.
- AC: add macOS/Windows CI jobs (CPU builds).

### D6. Language table is hand-maintained (P3, S)
- `normalize_lang_code` (core `lib.rs`) maps a fixed list of names/codes.
- AC: replace with a maintained crate (e.g. `isolang`) or generate the table
  from whisper.cpp's language list; keep the existing unit tests passing.

### D7. `--language auto` / unknown codes pass through (P3, S)
- `transcribe --language auto` (or an unknown code) falls back to the literal
  string and is handed to whisper instead of being rejected early.
- AC: reject values that do not normalize to a known code with a helpful
  error listing examples.

## Missing functionality

### F1. Automatic speaker diarization (P2, M)
- Speaker labels are manual (`--set-speaker-names`). The downloader lists
  `tinydiarize` models, but transcription never uses diarization.
- AC: wire whisper.cpp diarization behind a `--diarize` flag; label segments
  by speaker turn; keep manual labels as the default; document that
  diarization is approximate.

### F2. Live microphone ingestion (done in sprint 12, partly)
- `polyscribe record` captures the microphone through PipeWire and
  transcribes it while the recording runs. The capture is a supervised
  `pw-record` child, the samples are decoded incrementally and the memory
  does not grow with the length of the recording.
- Still open:
  - URL ingestion is not wired up. The engine/CLI split that it needs
    exists now.
  - The capture is Linux only, and only PipeWire. A native binding or an
    ALSA backend would widen it, at the price of a dependency.
- AC: a `--source` that takes a URL, with the same streaming contract.

### F5. A live window that keeps the quality of the batch window (P2, M)
- The live policy uses shorter windows, so that the text appears while the
  recording runs. A short window splits the speech at its end, so a word
  can be cut in two, and the window that follows decodes it again.
- The measured word error rate of the live windows against the reference is
  in `docs/benchmarks/live-transcription.md`.
- The five-second default remains provisional. The current synthetic test
  set favors 10 seconds for quality, but it is too small to choose a default
  for natural speech or other languages.
- AC: a window policy that overlaps the windows by the length of a word
  without paying for the extra decoding, with the same measurement as the
  evidence.

### F3. Parallel transcription of multiple files (P3, M)
- Inputs are processed sequentially. A `--jobs N` flag could parallelize
  across files (bounded by memory; blocked by D1).
- AC: `--jobs` with deterministic output ordering; unit test for ordering.

### F4. Additional output formats (P3, S)
- Outputs are JSON, TOML, and SRT. A `--out-format json|srt|toml|all`
  selector would avoid writing formats the user does not need.
- AC: flag with validation and per-format tests; stdout stays JSON-only.

## Optional improvements

### O1. CI matrix beyond Linux (P2, S)
- CI runs on Linux only. Add macOS/Windows jobs (CPU builds) once D5 is
  verified locally.
- AC: green CI on three OSes for build + clippy + test.

### O2. Structured logging behind a flag (P3, S)
- Logs are human-oriented (cliclack/tracing on stderr). A `--log-format json`
  option would help machine parsing in CI.
- AC: JSON lines with level/timestamp/message; default stays human output.

### O3. Configuration file (P3, M)
- Options come from flags/env only. A `polyscribe.toml` (XDG config) could
  hold defaults (language, backend, models dir).
- AC: flags override config; config overrides defaults; documented schema.

### O4. Transcription progress beyond log lines (P3, S)
- Audio transcription reports progress only via the backend callback
  (currently logged at `-v`). A spinner/elapsed indicator for long files
  would help.
- AC: spinner during inference when interactive; suppressed by `--quiet` /
  `--no-progress` / non-TTY.

## Long-term ideas (not scheduled)

### L1. Additional frontends (L)
- The engine (core) is decoupled from the CLI; a TUI or GUI frontend can be
  added as another crate consuming `polyscribe-core` without touching the
  CLI. Do not split crates further before there is a second consumer.

### L2. Additional STT engines (L)
- `TranscribeBackend` (core `backend.rs`) is the extension point. Future
  engines (remote APIs, other local models) implement the trait and register
  in `select_backend`.

### L3. Consolidation with the separate STT and YTT projects (L)
- The STT and YTT projects are separate today. Consolidating them into this
  repository would give one codebase, one release train and one plugin
  protocol, instead of three.
- The decisions are taken (2026-10-21):
  - **License**: MIT for PolyScribe, STT and YTT. Both Python repositories are
    the work of one author and hold no third-party code, so the license can be
    set on them; their code may then be used as the specification of a port.
  - **Local inference**: whisper.cpp is the only supported local engine. The
    openai-whisper CLI path of the Python tools is retired, not ported.
  - **Microphone**: genuine live audio streaming, not
    record-to-WAV-then-transcribe. The streaming foundation is in place
    (`polyscribe-core::stream`); a capture source is what is missing.
  - **Language**: automatic detection is the default, with an explicit
    override. The core detects the language once and pins it; the command
    line still requires `--language` and changes with the workflow that
    brings the automatic default to the user.
  - **Command line**: the workflows become `polyscribe` subcommands. No
    compatibility wrappers for `stt` and `ytt` are kept.
  - **Home Assistant**: the YTT bridge is not migrated.
- Nothing about the migration is implemented yet, apart from the streaming
  foundation it needs. The decisions that remain: the shape of the source
  adapters, whether the remote engine (Voxtral) is a first-class engine, and
  the versioning of the combined project.

### L4. ytt / STT-CLI integration (L)
- Long-term direction: a reusable speech-processing foundation (file, mic,
  URL ingestion; multiple engines; CLI/TUI/GUI frontends). The current
  workspace (core/cli/host/protocol) is the foundation for that.
