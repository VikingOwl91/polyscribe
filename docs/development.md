# Development

This document describes how to build, test, and run PolyScribe locally, and how to work with models during development.

Prerequisites
- Rust toolchain via rustup (recommended).
- ffmpeg installed and on PATH.
- For GPU builds: appropriate toolkits and libraries installed (CUDA/ROCm/Vulkan).

Rust toolchain
- Target stable Rust.
- Recommended:
  - rustup install stable
  - rustup default stable

Dependency pinning
- We pin whisper-rs (git dependency) to a known-good commit in Cargo.toml for reproducibility.
- To bump it, resolve/test the desired commit locally, then run:
  - cargo update -p whisper-rs --precise 135b60b85a15714862806b6ea9f76abec38156f1
  Replace the SHA with the desired commit and update the rev in Cargo.toml accordingly.

Build
- CPU-only (default):
  - cargo build
  - cargo test
- Enable GPU features at build time:
  - CUDA: cargo build --features gpu-cuda
  - HIP: cargo build --features gpu-hip
  - Vulkan: cargo build --features gpu-vulkan

Run locally
- Development model directory defaults to ./models when built in debug mode.
- You can override:
  - POLYSCRIBE_MODELS_DIR=/path/to/models
  - WHISPER_MODEL=/path/to/model.bin (forces a specific file)
- Example run:
  - cargo run -- -v -o output samples/example.mp3

Models during development
- Interactive downloader:
  - cargo run -- --download-models
- Non-interactive update (checks sizes/hashes, downloads if missing):
  - cargo run -- --update-models --no-interaction -q

Tests
- Run all tests:
  - cargo test
- The test suite includes CLI-oriented integration tests and unit tests. Some tests simulate GPU detection using env vars (POLYSCRIBE_TEST_FORCE_*). Do not rely on these flags in production code.

Clippy
- Run lint checks and treat warnings as errors:
  - cargo clippy --all-targets -- -D warnings
- Common warnings can often be fixed by simplifying code, removing unused imports, and following idiomatic patterns.

Code layout
- src/lib.rs: core library surface and re-exports
  - backend: runtime backend selection and transcription glue
  - models: model discovery, manifest, download/update logic
- src/main.rs: CLI parsing, I/O orchestration, logging, and workflows
- tests/: integration tests

Adding a feature
- Find the closest existing module (backend/models/main) and add a small, focused unit test.
- Keep user-facing changes documented in README/docs and update CHANGELOG.md.
- Prefer small functions with clear responsibilities; avoid exposing unnecessary items publicly.
- Follow existing logging style (elog!/wlog!/ilog!/dlog!).

Running the model downloader
- Interactive:
  - cargo run -- --download-models
- Non-interactive suggestions for CI:
  - POLYSCRIBE_MODELS_DIR=$PWD/models \
    cargo run -- --update-models --no-interaction -q

Env var examples for local testing
- Use a local copy of models and a specific model file:
  - export POLYSCRIBE_MODELS_DIR=$PWD/models
  - export WHISPER_MODEL=$PWD/models/large-v3-turbo-q8_0.bin
- Test manifests/offline copies are handled internally. For full offline runs, pre-populate models/ with the desired .bin files.
