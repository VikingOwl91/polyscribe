# PolyScribe

PolyScribe is a fast, local-first CLI for transcribing audio/video and merging existing JSON transcripts. It uses whisper-rs under the hood, can discover and download Whisper models automatically, and supports CPU and optional GPU backends (CUDA, ROCm/HIP, Vulkan).

Key features
- Transcribe audio and common video files using ffmpeg for audio extraction.
- Merge multiple JSON transcripts, or merge and also keep per-file outputs.
- Model management: interactive downloader and non-interactive updater with hash verification.
- GPU backend selection at runtime; auto-detects available accelerators.
- Clean outputs (JSON and SRT), speaker naming prompts, and useful logging controls.

Prerequisites
- Rust toolchain (rustup recommended)
- ffmpeg available on PATH
- Optional for GPU acceleration at runtime: CUDA, ROCm/HIP, or Vulkan drivers (match your build features)

Installation
- Build from source (CPU-only by default):
  - rustup install stable
  - rustup default stable
  - cargo build --release
- Binary path: ./target/release/polyscribe
- GPU builds (optional): build with features
  - CUDA: cargo build --release --features gpu-cuda
  - HIP: cargo build --release --features gpu-hip
  - Vulkan: cargo build --release --features gpu-vulkan

Quickstart
1) Download a model (first run can prompt you):
- ./target/release/polyscribe --download-models

2) Transcribe a file:
- ./target/release/polyscribe -v -o output my_audio.mp3
This writes JSON and SRT into the output directory with a date prefix.

Shell completions and man page
- Completions: ./target/release/polyscribe completions <bash|zsh|fish|powershell|elvish> > polyscribe.<ext>
  - Then install into your shell’s completion directory.
- Man page: ./target/release/polyscribe man > polyscribe.1 (then copy to your manpath)

Model locations
- Development (debug builds): ./models next to the project.
- Packaged/release builds: $XDG_DATA_HOME/polyscribe/models or ~/.local/share/polyscribe/models.
- Override via env var: POLYSCRIBE_MODELS_DIR=/path/to/models.
- Force a specific model file via env var: WHISPER_MODEL=/path/to/model.bin.

Most-used CLI flags
- -o, --output FILE_OR_DIR: Output path base (date prefix added). If omitted, JSON prints to stdout.
- -m, --merge: Merge all inputs into one output; otherwise one output per input.
- --merge-and-separate: Write both merged output and separate per-input outputs (requires -o dir).
- --set-speaker-names: Prompt for a speaker label per input file.
- --update-models: Verify/update local models by size/hash against the upstream manifest.
- --download-models: Interactive model list + multi-select download.
- --language LANG: Language code hint (e.g., en, de). English-only models reject non-en hints.
- --gpu-backend [auto|cpu|cuda|hip|vulkan]: Select backend (auto by default).
- --gpu-layers N: Offload N layers to GPU when supported.
- -v/--verbose (repeatable): Increase log verbosity. -vv shows very detailed logs.
- -q/--quiet: Suppress non-error logs (stderr); does not silence stdout results.
- --no-interaction: Never prompt; suitable for CI.

Minimal usage examples
- Transcribe an audio file to JSON/SRT:
  - ./target/release/polyscribe -o output samples/podcast_clip.mp3
- Merge multiple transcripts into one:
  - ./target/release/polyscribe -m -o output merged input/a.json input/b.json
- Update local models non-interactively (good for CI):
  - ./target/release/polyscribe --update-models --no-interaction -q

Troubleshooting & docs
- docs/faq.md – common issues and solutions (missing ffmpeg, GPU selection, model paths)
- docs/usage.md – complete CLI reference and workflows
- docs/development.md – build, run, and contribute locally
- docs/design.md – architecture overview and decisions
- docs/release-packaging.md – packaging notes for distributions
- docs/ci.md – minimal CI checklist and job outline
- CONTRIBUTING.md – PR checklist and workflow

CI status: [CI badge placeholder]

Examples
See the examples/ directory for copy-paste scripts:
- examples/transcribe_file.sh
- examples/update_models.sh
- examples/download_models_interactive.sh

License
-------
This project is licensed under the MIT License — see the LICENSE file for details.
