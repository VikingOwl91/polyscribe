# Contributing

Thank you for your interest in contributing!

Development setup
- Install Rust via rustup.
- Ensure ffmpeg is installed and available on PATH.
- For GPU builds, install the appropriate runtime (CUDA/ROCm/Vulkan) and enable the matching features.

Coding guidelines
- Prefer small, focused changes.
- Add tests where reasonable.
- Keep user-facing changes documented in README/docs.
- Run clippy and fix warnings.

CI checklist
- Build: cargo build --all-targets --locked
- Tests: cargo test --all --locked
- Lints: cargo clippy --all-targets -- -D warnings
- Optional: smoke-run examples inline (from README):
  - ./target/release/polyscribe --update-models --no-interaction -q
  - ./target/release/polyscribe -o output samples/podcast_clip.mp3

Notes
- For GPU features, use --features gpu-cuda|gpu-hip|gpu-vulkan as needed in your local runs.
- For docs-only changes, please still ensure the project builds.
