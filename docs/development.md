# Development

This document describes how to build, test, and run PolyScribe locally, and how to work with models during development.

## Prerequisites

- Rust 1.88 or newer (the MSRV; `rustup` recommended). `rust-toolchain.toml` pins 1.89 for development/CI.
- A C/C++ toolchain, `cmake`, and `libclang` — `whisper-rs-sys` compiles whisper.cpp and generates Rust bindings with bindgen.
- `ffmpeg` installed and on `PATH` (runtime for media transcription, and one integration test).
- For GPU builds: the matching toolchains and libraries (CUDA, ROCm).

## Workspace layout

- `crates/polyscribe-core`: engine — backends and the transcription of a recording in windows (`backend.rs`), the incremental audio decoding (`chunked.rs`), transcription glue, model discovery (`models.rs`), `ModelManager` (`model_manager.rs`), config (`config.rs`), error type (`error.rs`), output rendering (`output.rs`), and the cliclack UI helpers (`ui.rs`, `ui/progress.rs`).
- `crates/polyscribe-cli`: the `polyscribe` binary — CLI parsing (`cli.rs`), command orchestration (`main.rs`), the `transcribe` pipeline (`transcribe.rs`), and stdout modes (`output.rs`).
- `crates/polyscribe-host`: plugin discovery and process management.
- `crates/polyscribe-protocol`: PSP/1 plugin protocol types.
- `plugins/polyscribe-plugin-tubescribe`: reference plugin (workspace member; built and tested with the workspace).
- `crates/polyscribe-cli/tests/fixtures/`: deterministic fixtures for the integration tests.

## Build

- CPU-only (default):
  - `cargo build`
  - `cargo test --workspace`
- Enable GPU features at build time (availability is a build-time property):
  - CUDA: `cargo build -p polyscribe-cli --features gpu-cuda`
  - HIP: `cargo build -p polyscribe-cli --features gpu-hip`
- There is no `gpu-vulkan` feature: the Vulkan backend is not wired up.
- MSRV check after dependency changes:
  - `cargo +1.88 check --workspace --all-targets`

## Run locally

- The model directory is the same in debug and release builds: `$XDG_DATA_HOME/polyscribe/models` or `~/.local/share/polyscribe/models` (models installed with `models add` in the cache dir are found automatically).
- You can override:
  - `POLYSCRIBE_MODELS_DIR=/path/to/models`
  - `WHISPER_MODEL=/path/to/model.bin` (forces a specific file)
- Example run:
  - `cargo run -p polyscribe-cli -- transcribe -v -o output --language en samples/example.mp3`

## Models during development

- Interactive downloader:
  - `cargo run -p polyscribe-cli -- models download`
- Non-interactive add/update (verifies sizes/hashes, downloads if missing):
  - `cargo run -p polyscribe-cli -- models add ggerganov/whisper.cpp ggml-tiny.en.bin`
  - `cargo run -p polyscribe-cli -- models update`
- The `models` subcommands use `$XDG_CACHE_HOME/polyscribe/models` for files and `$XDG_CONFIG_HOME/polyscribe/models.json` for the manifest; override with `POLYSCRIBE_CACHE_DIR` / `POLYSCRIBE_CONFIG_DIR`.

## Tests

- Run all tests:
  - `cargo test --workspace`
- The suite needs no network, models, or GPU. It uses the deterministic fixtures in `crates/polyscribe-cli/tests/fixtures/`.
- Run with `--workspace` (not `-p polyscribe-cli`) so the plugin end-to-end tests run; they are skipped with a note when the plugin binary is not built.
- One test exercises the ffmpeg failure path and is skipped (with a note) when ffmpeg is not on PATH.
- Some unit tests simulate GPU detection using env vars (`POLYSCRIBE_TEST_FORCE_*`). Do not rely on these flags in production code.
- Tests that mutate environment variables take a shared lock (`ENV_LOCK`) so they can run in parallel safely.
- The suite is feature-independent: run it with default features and, on hosts with ROCm, also with `cargo test --workspace --features polyscribe-cli/gpu-hip`.
- `cargo test --workspace --release` additionally compiles and runs the release-only tests.
- The boundary and recovery suites need a model and skip themselves without one:
  `POLYSCRIBE_E2E_MODEL=/path/to/ggml-tiny.en.bin cargo test -p polyscribe-cli --test integration_boundary --test integration_recovery`.
  The fixtures are in `crates/polyscribe-cli/tests/fixtures/boundary/` and
  `crates/polyscribe-cli/tests/fixtures/recovery/`.
- `POLYSCRIBE_WINDOW_TRACE=1` makes a transcription write the sample range of
  every window, its segments and every region that whisper did not report to
  stderr. It is a diagnostic, not part of the normal output.
- The streaming tests feed a recording to the engine in blocks while the
  recording is still being delivered:
  `POLYSCRIBE_E2E_MODEL=/path/to/ggml-tiny.en.bin cargo test -p polyscribe-cli
  --test integration_stream`. They need no device, no desktop session, no
  network and no key.
- The streaming measurement is an ignored test:
  `cargo test -p polyscribe-cli --test integration_stream -- --ignored
  --nocapture` with `POLYSCRIBE_E2E_MODEL` set. It reports the time to the
  first window, the time to the first finalized segment, the largest amount
  of buffered audio and the peak resident memory.

## Clippy and formatting

- Run lint checks and treat warnings as errors:
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Format:
  - `cargo fmt --all -- --check`
- The workspace lints (`[workspace.lints]` in the root `Cargo.toml`) deny unused imports and warn on dead code; each crate opts in with `[lints] workspace = true`.

## Adding a feature

- Find the closest existing module (core: backend/models/model_manager/output; cli: transcribe/cli) and add a small, focused unit test.
- Keep user-facing changes documented in README/docs and update `CHANGELOG.md`.
- Prefer small functions with clear responsibilities; avoid exposing unnecessary items publicly.
- Follow the existing logging style (`elog!`/`ilog!`/`dlog!` macros in core; `ui::*` helpers for interactive output).

## Plugin development

- A plugin is an executable named `polyscribe-plugin-<name>` on `PATH` or in `$XDG_DATA_HOME/polyscribe/plugins`.
- Interface: `<plugin> info` prints capabilities JSON (`polyscribe_protocol::Capabilities`); `<plugin> run` reads one JSON-RPC request line from stdin and writes NDJSON `StreamItem`s (progress/result/error) to stdout.
- `plugins/polyscribe-plugin-tubescribe` is the reference implementation; `crates/polyscribe-cli/tests/integration_plugins.rs` exercises it end to end.

## Env var examples for local testing

- Use a local copy of models and a specific model file:
  - `export POLYSCRIBE_MODELS_DIR=$PWD/models`
  - `export WHISPER_MODEL=$PWD/models/ggml-tiny.en.bin`
- For full offline runs of the `models` subcommands, pre-populate the cache dir or use `models add` with a repo reachable on your network.
