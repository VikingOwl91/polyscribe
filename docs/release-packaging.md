# Release packaging

Model directory layout
- Recommended system path for models:
  - `$XDG_DATA_HOME/polyscribe/models`
  - Fallback: `~/.local/share/polyscribe/models`
- Allow override with `POLYSCRIBE_MODELS_DIR`.
- Keep models outside of `/usr` so users can update without root.

Binary naming
- Install binary as `polyscribe`.

Build requirements
- Rust 1.88+ (MSRV), a C/C++ toolchain, `cmake`, and `libclang` (whisper.cpp is compiled from source and its bindings generated with bindgen).
- HTTPS uses rustls (pure Rust); there is no OpenSSL build or runtime dependency.

Linux distribution tips (Arch example)
- Package name: `polyscribe`
- Runtime dependencies:
  - `ffmpeg`
  - (optional) CUDA/ROCm runtimes matching enabled features
- Build:
  - CPU-only: `cargo build --release`
  - CUDA: `cargo build --release --features gpu-cuda`
  - HIP: `cargo build --release --features gpu-hip`
- Place README and docs under `/usr/share/doc/polyscribe`.
- Do not place models under `/usr/share` directly; prefer the XDG data path resolved at runtime.

Man page and completions
- Generate at install time and store under conventional locations:
  - `polyscribe man > /usr/share/man/man1/polyscribe.1`
  - `polyscribe completions bash > /usr/share/bash-completion/completions/polyscribe`
  - `polyscribe completions zsh > /usr/share/zsh/site-functions/_polyscribe`

Non-interactive behavior for CI and packaging
- Use `--no-interaction` and set `POLYSCRIBE_MODELS_DIR` to a writable location during build/test steps.
