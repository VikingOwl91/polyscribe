# Development

This document describes how to build, test, and run PolyScribe locally, and how to work with models during development.

Prerequisites
- Rust 1.88 or newer (the MSRV; `rustup` recommended).
- A C/C++ toolchain, `cmake`, and `libclang` — the `whisper-rs-sys` build script compiles whisper.cpp and generates Rust bindings with bindgen.
- `ffmpeg` installed and on `PATH` (needed at runtime for media transcription, and by one integration test).
- For GPU builds: the matching toolchains and libraries (CUDA, ROCm).

Rust toolchain
- Target stable Rust; the MSRV is 1.88 (`rust-version` in `Cargo.toml`).
- Recommended:
  - `rustup install stable`
  - `rustup default stable`
- Verify the MSRV after dependency changes:
  - `cargo +1.88 check --all-targets`

Build
- CPU-only (default):
  - `cargo build`
  - `cargo test`
- Enable GPU features at build time (availability is a build-time property):
  - CUDA: `cargo build --features gpu-cuda`
  - HIP: `cargo build --features gpu-hip`
- There is no `gpu-vulkan` feature: the Vulkan backend is not wired up.

Run locally
- Development model directory defaults to `./models` when built in debug mode.
- You can override:
  - `POLYSCRIBE_MODELS_DIR=/path/to/models`
  - `WHISPER_MODEL=/path/to/model.bin` (forces a specific file)
- Example run:
  - `cargo run -- -v -o output --language en samples/example.mp3`

Models during development
- Interactive downloader:
  - `cargo run -- --download-models`
- Non-interactive update (checks sizes/hashes, downloads if missing):
  - `cargo run -- --update-models --no-interaction -q`

Tests
- Run all tests:
  - `cargo test`
- The test suite includes CLI-oriented integration tests and unit tests. Integration tests use the deterministic fixtures in `tests/fixtures/` and need no network, models, or GPU. One test exercises the ffmpeg failure path and is skipped (with a note) when ffmpeg is not on PATH.
- Some unit tests simulate GPU detection using env vars (`POLYSCRIBE_TEST_FORCE_*`). Do not rely on these flags in production code.
- Tests that mutate environment variables take a shared lock (`ENV_LOCK`) so they can run in parallel safely.

Clippy and formatting
- Run lint checks and treat warnings as errors:
  - `cargo clippy --all-targets -- -D warnings`
- Format:
  - `cargo fmt --all -- --check`

Code layout
- `src/lib.rs`: core library surface: runtime flags, logging macros, model discovery (`find_model_file`), audio decoding (`decode_audio_to_pcm_f32_ffmpeg`), input classification, and module re-exports.
- `src/output.rs`: transcript types (`OutputEntry`, `OutputRoot`), SRT rendering, sorting/id reassignment, output writing (JSON/TOML/SRT, stdout), and JSON transcript loading.
- `src/backend.rs`: backend selection (`select_backend`), the `TranscribeBackend` trait and CPU/CUDA/HIP implementations, per-input entry collection (`collect_entries_for_input`), and the whisper-rs transcription glue.
- `src/models.rs`: model discovery, manifest, download/update logic (Hugging Face).
- `src/main.rs`: CLI parsing, prompting, and the three output modes (separate, merge, merge-and-separate).
- `tests/`: integration tests (`integration_cli.rs`, `integration_aux.rs`) and fixtures (`tests/fixtures/`).

Adding a feature
- Find the closest existing module (output/backend/models/main) and add a small, focused unit test.
- Keep user-facing changes documented in README/docs.
- Prefer small functions with clear responsibilities; avoid exposing unnecessary items publicly.
- Follow the existing logging style (`elog!`/`wlog!`/`ilog!`/`dlog!`).

Running the model downloader
- Interactive:
  - `cargo run -- --download-models`
- Non-interactive suggestions for CI:
  - `POLYSCRIBE_MODELS_DIR=$PWD/models cargo run -- --update-models --no-interaction -q`

Env var examples for local testing
- Use a local copy of models and a specific model file:
  - `export POLYSCRIBE_MODELS_DIR=$PWD/models`
  - `export WHISPER_MODEL=$PWD/models/large-v3-turbo-q8_0.bin`
- Test manifests/offline copies are handled internally (`POLYSCRIBE_MODELS_MANIFEST`, `POLYSCRIBE_MODELS_BASE_COPY_DIR`). For full offline runs, pre-populate `models/` with the desired `.bin` files.
