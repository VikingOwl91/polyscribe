# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## Unreleased

### Changed
- Transcribe: long recordings are decoded and transcribed in windows of 30 seconds instead of being loaded into memory as a whole. The peak memory of a transcription no longer grows with the length of the recording: an hour long recording needs about 290 MB instead of 670 MB (CPU) and 780 MB (HIP). The output contracts do not change; a recording shorter than a window is transcribed exactly as before, a longer one keeps the same text with segment boundaries that can move slightly at a window boundary.
- Transcribe: a window now advances by at least half a window. whisper reports short segments for silence, so a window that started where the last segment ended decoded the same stretch of silence again and again: a minute of silence took 16 windows and produced a hallucinated segment per window, it now takes 2 windows. The speech at a window boundary is also recognized better, because the window no longer starts in the middle of a sentence.
- Backends: an explicit `--gpu-backend cuda` or `--gpu-backend hip` now requires a matching GPU device to be present. The device list comes from ggml's backend registry, the same registry whisper.cpp selects its GPU from. Without a device the request fails with an actionable error instead of silently continuing on the CPU; `--gpu-backend auto` falls back to the CPU and reports it.
- Transcribe: the Whisper model is loaded once per invocation and reused for every input file, instead of being loaded again for each file. Every file still gets its own inference state, so outputs, timestamps and error handling stay independent.

### Fixed
- Transcribe: a GPU-enabled build ran every transcription on the GPU, also the ones that asked for the CPU, while the log still reported the selected backend. The selected backend now reaches the model parameters, so a CPU request keeps the model in RAM and a GPU request offloads it.
- Transcribe: a segment that was cut short by the end of a window is no longer dropped before it is decoded again. The re-decode can start a little before the dropped segment and was then mistaken for a duplicate, so a whole sentence could be missing from the transcript. The cut segment is kept until the re-decode replaces it.
- Transcribe: segments without a duration are no longer written to the transcript. whisper reports such a segment at the end of a window, with the text of the speech around it; it showed up as a subtitle without a duration, on top of the segment next to it.

### Added
- Core: `chunked` module that decodes a media file incrementally and hands it to the transcriber in windows, so the memory of a transcription is independent of the length of the recording
- Docs: `docs/benchmarks/chunked-transcription.md` describes the pipeline, the memory bounds, the window and reconciliation strategy, the boundary quality measurement and the measured before/after results
- Core: `gpu` module with a device probe for the GPU backends, and unit tests for the backend selection contract (explicit CPU, explicit GPU with and without a device, and the automatic fallback).
- Tests: CLI integration tests for the backend selection contract.
- Tests: CLI integration tests for the window boundaries, with speech fixtures that cross a boundary. They check that every stretch of speech is covered by a segment, that the timestamps are ordered and do not overlap, and that a minute of silence does not need more than four windows.
- Tests: speech fixtures for the boundary tests, generated with espeak-ng and resampled to 16 kHz mono, with the reference transcript of every clip.
- Scripts: `scripts/benchmarks/boundary_quality.py` composes the boundary fixtures, transcribes them with two binaries and reports the word error rate, the missing and the duplicated words against the reference transcript.
- Docs: the backend selection contract and the model context lifecycle in `docs/usage.md` and `docs/design.md`, plus the before/after measurements of the model context reuse in `docs/benchmarks/amd-rx-9070-xt.md`.

## v0.1.0-alpha.1 - 2026-10-21

### Added
- Tests (audit F-05): end-to-end transcription smoke test (`crates/polyscribe-cli/tests/e2e_transcribe.rs`) that runs the built CLI on the committed `speech.wav` fixture with a real Whisper model on CPU, asserting successful execution, valid JSON/TOML/SRT output, and sane timestamps. It skips itself with a note unless `POLYSCRIBE_E2E_MODEL` points at a model file, so the workspace suite still needs no model.
- CI: new `e2e-transcribe` job that downloads a SHA-256-pinned `ggml-tiny.en.bin` (cached; the checksum is verified on every run so unverified cache contents are never trusted) and executes the e2e smoke test against the compiled CLI.
- CI: the `test` and `e2e-transcribe` jobs install ffmpeg when it is missing (current ubuntu-latest runners no longer ship it), so ffmpeg-dependent tests run instead of skipping; the first-run model-discovery tests skip themselves with a note when ffmpeg is absent locally.

### Changed
- Docs: Replace `--download-models`/`--update-models` flags with `models download`/`models update` subcommands in `README.md`, `docs/usage.md`, and `docs/development.md`.
- Host: Plugin discovery now scans `$XDG_DATA_HOME/polyscribe/plugins` (platform equivalent via `directories`) in addition to `PATH`.
- CI: Add GitHub Actions workflow to run fmt, clippy (warnings as errors), and tests for pushes and PRs.
- Models (audit F-02): the transcription models directory is resolved by one canonical policy in every build mode: `POLYSCRIBE_MODELS_DIR`, then `$XDG_DATA_HOME/polyscribe/models`, then `~/.local/share/polyscribe/models`. The debug-only `./models` default is gone, so `cargo run -- models download` and `cargo run -- transcribe` agree on the directory. `ConfigService::models_dir` delegates to the shared resolver; the divergent `directories`-crate-based `default_models_dir` was removed. Docs (README, usage, development, release-packaging, faq) updated to match.

### Integrated (merge of the modernization line into the workspace)
- Dependency: `whisper-rs` upgraded from 0.14.3 (git) to the versioned crates.io release 0.16; segment access migrated to the 0.16 API. MSRV declared as 1.88 (`rust-version`), verified with `cargo +1.88 check --workspace`.
- Backends: removed the non-functional Vulkan stub (`gpu-vulkan` feature, `VulkanBackend`, `vulkan` CLI value); GPU availability is now a build-time property (`--features gpu-cuda` / `--features gpu-hip` on `polyscribe-cli`), with actionable errors when a backend was not compiled in.
- HTTPS: `reqwest` uses `rustls-tls` (pure-Rust TLS, bundled Mozilla roots) with charset/http2/system-proxy enabled; no system OpenSSL requirement.
- Transcribe: the `transcribe` subcommand now runs the full pipeline (backend selection, ffmpeg decode, whisper inference, JSON/TOML/SRT output) instead of a stub; output writing is deduplicated in `polyscribe-core`'s `output` module.
- Audio: ffmpeg decoding streams PCM instead of using a fixed-name temp file (no cross-run collisions, half the peak memory).
- Logging: quiet mode now consistently suppresses info/success/progress output (ProgressReporter, Spinner, BytesProgress, ui::info/success/println_above_bars); errors and warnings remain visible.
- Security: model file names from remote listings/manifests are validated against path traversal (`ModelManager::add_or_update`, manifest file paths).
- Plugins: fixed the PSP/1 interface — the protocol crate gained `Capabilities`/`StreamItem`/`JsonRpcRequest`, the reference plugin now implements `info`/`run` subcommands, the host resolves plugins from the XDG plugin dir as well as `PATH`, and `plugins run --payload` no longer clashes with the global `--json` flag (renamed from `--json`).
- Tests: deterministic fixtures live in `crates/polyscribe-cli/tests/fixtures/`; the transcribe integration suite (20 tests), models smoke tests, and plugin end-to-end tests run without network, models, or GPU; env-mutating tests are serialized with a shared lock.
- CI: workflow runs fmt, locked workspace build, locked workspace tests, clippy with warnings as errors, plus an MSRV (1.88) job; triggers on all pushes and PRs.
- Docs: README, CONTRIBUTING, and docs/ rewritten to match the workspace CLI (subcommands, models/plugins, caching, GPU behavior, limitations).

### Fixed
- `plugins run --json` collided with the global `--json` output flag, making the subcommand unparseable; the payload flag is now `--payload`.
- `transcribe` previously validated inputs and reported success without transcribing anything.
- First run (audit F-01): with no model installed and no model cache dir, `transcribe` now shows the actionable "No Whisper model files found ... download a model (polyscribe models add) or set WHISPER_MODEL" message instead of `Failed to read models dir: ...`. A missing cache dir counts as "no cached models"; genuine I/O errors on existing directories still propagate.
- Tests: model-discovery unit tests (empty/missing dirs, cache-dir fallback, `WHISPER_MODEL` overrides, unreadable cache dir) and CLI first-run integration tests running the binary with an isolated environment.


