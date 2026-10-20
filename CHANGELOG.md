# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## Unreleased

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


