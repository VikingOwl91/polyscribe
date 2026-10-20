# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## Unreleased

### Changed
- Docs: Replace `--download-models`/`--update-models` flags with `models download`/`models update` subcommands in `README.md`, `docs/usage.md`, and `docs/development.md`.
- Host: Plugin discovery now scans `$XDG_DATA_HOME/polyscribe/plugins` (platform equivalent via `directories`) in addition to `PATH`.
- CI: Add GitHub Actions workflow to run fmt, clippy (warnings as errors), and tests for pushes and PRs.

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


