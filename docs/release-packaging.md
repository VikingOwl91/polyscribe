# Release packaging

## Workspace

The repository is a Cargo workspace. Package the CLI crate; the engine and plugin crates are libraries/build artifacts of it:

- CLI crate: `crates/polyscribe-cli` (binary `polyscribe`)
- Install from source: `cargo install --path crates/polyscribe-cli`
- Reference plugin: `plugins/polyscribe-plugin-tubescribe` (binary `polyscribe-plugin-tubescribe`); ship it on `PATH` or document installing it into `$XDG_DATA_HOME/polyscribe/plugins`.

## Build requirements

- Rust 1.88+ (MSRV), a C/C++ toolchain, `cmake`, and `libclang` (whisper.cpp is compiled from source and its bindings generated with bindgen).
- HTTPS uses rustls (pure Rust, bundled Mozilla roots); there is no OpenSSL build or runtime dependency.

## Model directory layout

- Transcription models (used by `transcribe`):
  - `$XDG_DATA_HOME/polyscribe/models`
  - Fallback: `~/.local/share/polyscribe/models`
  - Debug builds: `./models` next to the project.
- Override with `POLYSCRIBE_MODELS_DIR`; force a file with `WHISPER_MODEL`.
- `models` subcommands: files in `$XDG_CACHE_HOME/polyscribe/models`, manifest at `$XDG_CONFIG_HOME/polyscribe/models.json`.
- Keep models outside of `/usr` so users can update without root.

## Binary naming

- Install the CLI binary as `polyscribe`.
- Install the reference plugin as `polyscribe-plugin-tubescribe`.

## Linux distribution tips (Arch example)

- Package name: `polyscribe`
- Runtime dependencies:
  - `ffmpeg`
  - (optional) CUDA/ROCm runtimes matching enabled features
- Build:
  - CPU-only: `cargo build --release -p polyscribe-cli`
  - CUDA: `cargo build --release -p polyscribe-cli --features gpu-cuda`
  - HIP: `cargo build --release -p polyscribe-cli --features gpu-hip`
- Place README and docs under `/usr/share/doc/polyscribe`.
- Do not place models under `/usr/share` directly; prefer the XDG data path resolved at runtime.

## Man page and completions

- Generate at install time and store under conventional locations:
  - `polyscribe man > /usr/share/man/man1/polyscribe.1`
  - `polyscribe completions bash > /usr/share/bash-completion/completions/polyscribe`
  - `polyscribe completions zsh > /usr/share/zsh/site-functions/_polyscribe`

## Non-interactive behavior for CI and packaging

- Use `--no-interaction` (top-level) and the `models add` / `models update` subcommands; set `POLYSCRIBE_MODELS_DIR` (and `POLYSCRIBE_CACHE_DIR` / `POLYSCRIBE_CONFIG_DIR` for the `models` subcommands) to writable locations during build/test steps.
