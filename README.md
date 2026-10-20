# PolyScribe

PolyScribe is a fast, local-first CLI for transcribing audio/video and merging existing JSON transcripts. It uses whisper.cpp (via the `whisper-rs` crate) under the hood, can discover and download Whisper models automatically, and supports CPU and optional GPU builds (CUDA, ROCm/HIP).

Key features
- Transcribe audio and common video files; audio extraction is done with ffmpeg.
- Merge multiple JSON transcripts, or merge and also keep per-file outputs.
- Model management: interactive downloader and non-interactive updater with SHA-256 verification.
- GPU backend selection at runtime among the backends compiled into the binary (CUDA, ROCm/HIP); CPU-only by default.
- Outputs JSON, TOML, and SRT; speaker labels via per-file prompts (manual labeling only — there is no automatic speaker diarization).
- Useful logging controls (`-v`/`-q`, `--no-interaction`).

Prerequisites
- Rust 1.88 or newer (MSRV; `rustup` recommended)
- A C/C++ toolchain, `cmake`, and `libclang` (needed to build whisper.cpp and its Rust bindings)
- `ffmpeg` available on `PATH` at runtime
- Optional for GPU builds at build and run time: CUDA toolkit/driver (NVIDIA) or ROCm (AMD)

Installation
- Build from source (CPU-only by default):
  - `rustup install stable`
  - `cargo build --release`
- Binary path: `./target/release/polyscribe`
- GPU builds (optional):
  - CUDA: `cargo build --release --features gpu-cuda`
  - HIP (ROCm): `cargo build --release --features gpu-hip`

Quickstart
1) Download a model (first run can prompt you):
- `./target/release/polyscribe --download-models`

2) Transcribe a file (requires `--language`):
- `./target/release/polyscribe -v -o output my_audio.mp3 --language en`
- This writes JSON, TOML, and SRT into the output directory with a date prefix.

Shell completions and man page
- Completions: `./target/release/polyscribe completions <bash|zsh|fish|powershell|elvish> > polyscribe.<ext>`
  - Then install into your shell's completion directory.
- Man page: `./target/release/polyscribe man > polyscribe.1` (then copy to your manpath)

Model locations
- Development (debug builds): `./models` next to the project.
- Packaged/release builds: `$XDG_DATA_HOME/polyscribe/models` or `~/.local/share/polyscribe/models`.
- Override via env var: `POLYSCRIBE_MODELS_DIR=/path/to/models`.
- Force a specific model file via env var: `WHISPER_MODEL=/path/to/model.bin`.

Most-used CLI flags
- `-o, --output FILE_OR_DIR`: Output path. In the default (separate) mode this is a directory; with `-m` it is a file base name. A date prefix is added to file names. If omitted, JSON prints to stdout.
- `-m, --merge`: Merge all inputs into one output; otherwise one output per input.
- `--merge-and-separate`: Write both merged output and separate per-input outputs (requires `-o` dir).
- `--set-speaker-names`: Prompt for a speaker label per input file (manual labeling; no automatic diarization).
- `--update-models`: Verify/update local models by size/hash against the upstream manifest.
- `--download-models`: Interactive model list + multi-select download.
- `--language LANG`: Language code hint (e.g., `en`, `de`). Required when transcribing audio; English-only models reject non-`en` hints.
- `--gpu-backend [auto|cpu|cuda|hip]`: Select backend among those compiled in (default: `auto`, prefers CUDA, then HIP, then CPU).
- `-v/--verbose` (repeatable): Increase log verbosity. `-vv` shows very detailed logs.
- `-q/--quiet`: Suppress non-error logs (stderr); does not silence stdout results.
- `--no-interaction`: Never prompt; suitable for CI.

Minimal usage examples
- Transcribe an audio file to JSON/TOML/SRT:
  - `./target/release/polyscribe -o output --language en samples/podcast_clip.mp3`
- Merge multiple transcripts into one:
  - `./target/release/polyscribe -m -o output merged input/a.json input/b.json`
- Update local models non-interactively (good for CI):
  - `./target/release/polyscribe --update-models --no-interaction -q`

Notes and limitations
- Vulkan is not currently supported: PolyScribe does not wire up whisper.cpp's Vulkan backend, and `--gpu-backend vulkan` is rejected. CUDA and HIP builds are supported.
- Speaker labels are assigned manually per input file (`--set-speaker-names`); PolyScribe does not perform automatic speaker diarization.
- Network access is only used by `--download-models` and `--update-models` (Hugging Face API and model downloads over HTTPS). Everything else runs fully offline.

Troubleshooting & docs
- `docs/faq.md` – common issues and solutions (missing ffmpeg, GPU selection, model paths)
- `docs/usage.md` – complete CLI reference and workflows
- `docs/development.md` – build, test, and contribute locally
- `docs/design.md` – architecture overview and decisions
- `docs/release-packaging.md` – packaging notes for distributions
- `docs/ci.md` – CI checklist and job outline
- `CONTRIBUTING.md` – PR checklist and workflow

Examples
See the `examples/` directory for copy-paste scripts:
- `examples/transcribe_file.sh`
- `examples/update_models.sh`
- `examples/download_models_interactive.sh`

License
-------
This project is licensed under the MIT License — see the LICENSE file for details.
