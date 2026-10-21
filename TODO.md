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

### F2. Live microphone / URL ingestion (P3, M)
- Only local files are supported today.
- AC (future): mic capture via `cpal`, YouTube/URL ingestion via an external
  downloader. Requires the engine/CLI split to be exercised first.

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

### L3. ytt / STT-CLI integration (L)
- Long-term direction: a reusable speech-processing foundation (file, mic,
  URL ingestion; multiple engines; CLI/TUI/GUI frontends). The current
  workspace (core/cli/host/protocol) is the foundation for that.
