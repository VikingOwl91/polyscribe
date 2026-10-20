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
- `with_suppressed_stderr`/`StderrSilencer` (src/lib.rs) redirects the
  process-global fd 2 with unsafe libc calls around whisper model load and
  inference. It is correct for the current single-threaded CLI, but any
  future parallel transcription (see F3) would race on it.
- AC: either gate transcription to a single thread with a comment, or replace
  fd redirection with whisper-rs's logging hooks (ggml log callback) so no
  global state is touched.

### D2. `.last_model` side-effect file is a workaround (P2, S)
- Model selection writes `.last_model` into the models dir and a `Drop` guard
  in main.rs removes it at exit. If the process is killed, the file leaks
  (harmless, but sloppy).
- AC: keep last-used model in memory only, or accept the file and document it.

### D3. Long-file memory usage is O(duration) (P2, M)
- ffmpeg output is streamed into a `Vec<f32>` (64 KiB/s of audio), but
  `whisper_full` still needs the whole sample buffer in memory (~230 MiB per
  hour of audio). True streaming/chunked transcription would cap memory.
- AC: prototype chunked transcription with a long fixture; measure peak RSS
  before/after.

### D4. Interrupted downloads can leave `.part` files (P2, S)
- A killed `--download-models`/`--update-models` run leaves
  `ggml-*.bin.part` behind. They are ignored on later runs but never cleaned.
- AC: remove stale `.part` files at the start of a download/update run.

### D5. Language table is hand-maintained (P3, S)
- `normalize_lang_code` (src/lib.rs) maps a fixed list of names/codes.
- AC: replace with a maintained crate (e.g. `isolang`) or generate the table
  from whisper.cpp's language list; keep the existing unit tests passing.

### D6. No end-to-end transcription test (P1, M)
- Tests cover the JSON merge path and the ffmpeg failure path, but nothing
  runs a real transcription (needs a model + ffmpeg + audio fixture).
- AC: add a CI job that downloads the `tiny` model once (cached), generates a
  short WAV with ffmpeg, transcribes it, and asserts a non-empty transcript
  with sane timestamps.

### D7. Non-Unix TTY detection is a heuristic (P2, S)
- `stdin_is_tty` on non-Unix only checks `CI`/`GITHUB_ACTIONS` env vars.
- AC: at least verify behavior on Windows; consider the `is-terminal` crate.

## Missing functionality

### F1. Automatic speaker diarization (P2, M)
- Speaker labels are manual (`--set-speaker-names`). The downloader already
  lists `tinydiarize` models, but transcription never uses diarization.
- AC: wire whisper.cpp diarization (`whisper_full` with diarize params) behind
  a `--diarize` flag; label segments by speaker turn; keep manual labels as
  the default. Document clearly that diarization is approximate.

### F2. Live microphone / URL ingestion (P3, M)
- Only local files are supported today.
- AC (future): mic capture via `cpal`, YouTube/URL ingestion via `yt-dlp` as
  an external tool. Requires the engine/CLI split to be exercised first.

### F3. Parallel transcription of multiple files (P3, M)
- Inputs are processed sequentially. A `--jobs N` flag could parallelize
  across files (bounded by memory; see D1/D3).
- AC: `--jobs` with deterministic output ordering; unit test for ordering.

### F4. Additional output formats (P3, S)
- Outputs are JSON, TOML, and SRT. A `--out-format json|srt|toml|all` selector
  would avoid writing formats the user does not need.
- AC: flag with validation and per-format tests; stdout stays JSON-only.

## Optional improvements

### O1. CI matrix beyond Linux (P2, S)
- CI runs on Linux only. Add macOS/Windows jobs (CPU builds) once D7 is
  verified locally.
- AC: green CI on three OSes for build + clippy + test.

### O2. Transcription progress indication (P3, S)
- Today only log lines mark progress. A simple "processed Xs of Ys audio"
  line (or indicatif progress bar, stderr only) would help on long files.
- AC: progress line updates at most ~1/s; suppressed by `-q` and when not a TTY.

### O3. Structured logging behind a flag (P3, S)
- Logs are plain text on stderr. A `--log-format json` option would help
  machine parsing in CI.
- AC: JSON lines with level/timestamp/message; default stays plain text.

### O4. Configuration file (P3, M)
- Options come from flags/env only. A `polyscribe.toml` (XDG config) could
  hold defaults (language, backend, models dir).
- AC: flags override config; config overrides defaults; documented schema.

## Long-term ideas (not scheduled)

### L1. Workspace split for additional frontends (L)
- Today: one crate with `src/lib.rs` (engine) + `src/main.rs` (CLI). When a
  TUI or GUI frontend appears, split into `polyscribe-core` (engine: backends,
  models, output) and thin frontend crates. Do not split before there is a
  second consumer.

### L2. Additional STT engines (L)
- `TranscribeBackend` (src/backend.rs) is the extension point. Future engines
  (remote APIs, other local models) implement the trait and register in
  `select_backend`.

### L3. Integration with ytt / STT-CLI (L)
- Long-term direction: a reusable speech-processing foundation (file, mic,
  URL ingestion; multiple engines; CLI/TUI/GUI frontends). See the
  modernization report for a concrete integration assessment.
