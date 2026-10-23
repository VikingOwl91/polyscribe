# PolyScribe

PolyScribe is a fast, local-first CLI for transcribing audio/video and merging existing JSON transcripts, with a plugin system for extensibility. It uses whisper.cpp (via the `whisper-rs` crate) under the hood, can discover and download Whisper models automatically (with caching and integrity verification), and supports CPU and optional, experimental GPU builds (CUDA, ROCm/HIP).

## Features

- **Live microphone transcription**: `polyscribe record` captures the microphone through PipeWire and transcribes it while the recording runs. The audio is decoded incrementally, the memory does not grow with the length of the recording, and multilingual models detect the language by default.
- **Local-first**: Whisper inference always runs locally. Network access is used for model management and the opt-in YouTube downloader.
- **YouTube transcripts**: captions are preferred and normalized; when a video has no usable captions, PolyScribe can download audio and use the installed local Whisper model. An explicitly requested caption language is never replaced by another language. HTTP 403/429, network failures, and malformed caption files fail instead of starting audio transcription.
- **Transcription**: audio and common video files are decoded to 16 kHz mono PCM via ffmpeg and transcribed with whisper.cpp.
- **Transcript merging**: merge multiple JSON transcripts, or merge and also keep per-file outputs.
- **Model management**: `polyscribe models` subcommands to list, add, remove, verify, update, search, and download models, with a manifest cache, concurrent ranged downloads, and SHA-256 verification.
- **Plugin system**: extensible via JSON-RPC (NDJSON) plugins discovered on `PATH` or in the XDG plugin directory.
- **GPU backends**: CPU, CUDA, and ROCm/HIP; availability is a build-time property (`--features gpu-cuda` / `--features gpu-hip`). GPU support is experimental: no GPU runtime is exercised in CI. The HIP backend is validated manually on an AMD Radeon RX 9070 XT with ROCm 7.2.
- **Bounded memory for long recordings**: the audio is decoded and transcribed in windows of 30 seconds, so the peak memory of a transcription does not grow with the length of the recording.
- **Missing-speech recovery**: whisper does not always report a segment for the beginning of the audio it is given. PolyScribe decodes such a region again from a slightly different position, and warns on stderr when the region still holds no text.
- **Outputs**: JSON, TOML, and SRT; speaker labels via per-file prompts (manual labeling only — there is no automatic speaker diarization).
- **Machine-readable mode**: global `--json` flag for scripts.

## Workspace layout

- `crates/polyscribe-core`: engine — backends, transcription, models, model manager, output rendering, UI helpers.
- `crates/polyscribe-cli`: the `polyscribe` binary — CLI parsing and command orchestration.
- `crates/polyscribe-host`: plugin discovery and process management.
- `crates/polyscribe-protocol`: the PSP/1 plugin protocol types.
- `plugins/polyscribe-plugin-tubescribe`: reference plugin (builds with the workspace).

## Prerequisites

- Rust 1.88 or newer (MSRV; `rustup` recommended; `rust-toolchain.toml` pins 1.89 for development)
- A C/C++ toolchain, `cmake`, and `libclang` (needed to build whisper.cpp and its Rust bindings)
- `ffmpeg` available on `PATH` at runtime
- `yt-dlp` available on `PATH` for `polyscribe youtube`
- For `polyscribe record`: a running PipeWire session and `pw-record` on `PATH` (the Arch package `pipewire-audio`)
- A writable `TMPDIR` with a few GB free for GPU builds and GPU runs: the HIP and CUDA runtimes compile device code at runtime, and a full `/tmp` makes the run fail with an LLVM I/O error
- Optional for GPU builds at build and run time: CUDA toolkit/driver (NVIDIA) or ROCm (AMD)

## Supported platforms

- **Linux x86_64**: the tested platform. CI builds the workspace, runs the test suite and transcribes real audio on it, on CPU. The live capture was validated on it against a real PipeWire session and a USB microphone.
- **macOS**: expected to work on CPU, but untested in CI.
- **Windows**: not supported. The workspace does not compile on Windows yet.

## Installation

```bash
# Build the CLI from source (CPU-only by default)
cargo build --release
# Binary: ./target/release/polyscribe

# GPU builds (optional)
cargo build --release -p polyscribe-cli --features gpu-cuda   # NVIDIA
cargo build --release -p polyscribe-cli --features gpu-hip    # AMD ROCm

# Install
cargo install --path crates/polyscribe-cli
```

## Usage

```bash
# Transcribe audio/video (requires --language)
polyscribe transcribe --language en -o output input.mp4

# Retrieve and normalize captions; local Whisper is used only when captions are absent
polyscribe youtube 'https://www.youtube.com/watch?v=VIDEO_ID' --language en --format srt
polyscribe youtube 'https://www.youtube.com/watch?v=VIDEO_ID' --captions-only
polyscribe youtube 'https://www.youtube.com/watch?v=VIDEO_ID' --audio-only --language en
polyscribe youtube 'https://www.youtube.com/watch?v=VIDEO_ID' --refresh

# Record the microphone and transcribe it while the recording runs
polyscribe record                            # the language is detected
polyscribe record --language de --device voice_arctis_mono

# Merge multiple transcripts
polyscribe transcribe -m -o output merged input1.json input2.json

# Merge and also keep per-file outputs
polyscribe transcribe --merge-and-separate -o output input1.json input2.json

# Model management
polyscribe models ls
polyscribe models search ggerganov/whisper.cpp --query tiny
polyscribe models add ggerganov/whisper.cpp ggml-tiny.en.bin
polyscribe models verify ggml-tiny.en
polyscribe models update
polyscribe models download        # interactive downloader
polyscribe models clear-cache     # force a fresh manifest fetch

# Plugins
polyscribe plugins list
polyscribe plugins info tubescribe
polyscribe plugins run tubescribe generate_metadata --payload '{"url":"..."}'

# Shell completions and man page
polyscribe completions bash > polyscribe.bash
polyscribe man > polyscribe.1
```

Run `polyscribe -h` or `polyscribe <subcommand> -h` for the full reference; `docs/usage.md` has the details.

## Model locations and caching

- Transcription models: `$XDG_DATA_HOME/polyscribe/models` (or `~/.local/share/polyscribe/models`), the same in debug and release builds; models installed with `models add` in the cache dir are found automatically. Override with `POLYSCRIBE_MODELS_DIR`; force a specific file with `WHISPER_MODEL`.
- `models` subcommands use a separate cache: `$XDG_CACHE_HOME/polyscribe/models` for model files and `$XDG_CONFIG_HOME/polyscribe/models.json` for the manifest. Override with `POLYSCRIBE_CACHE_DIR` / `POLYSCRIBE_CONFIG_DIR`.
- The Hugging Face manifest is cached (`$XDG_CACHE_HOME/polyscribe/manifest/`, 24 h TTL); `POLYSCRIBE_NO_CACHE_MANIFEST=1` disables it, `POLYSCRIBE_MANIFEST_TTL_SECONDS` changes the TTL.
- YouTube transcripts are cached under `$XDG_CACHE_HOME/polyscribe/youtube/v1/` (or `~/.cache/polyscribe/youtube/v1/`). The cache is versioned, private, and shared across TXT/SRT/JSON renders. Use `youtube --refresh` to bypass it.
- `HF_TOKEN` is used for authenticated Hugging Face API/download requests when set.

## Notes and limitations

- **No Vulkan backend**: PolyScribe does not wire up whisper.cpp's Vulkan backend; `--gpu-backend vulkan` is rejected.
- **GPU support is experimental**: the CUDA and HIP backends are wired up at build time, but no GPU runtime is exercised in CI. Availability is a build-time property: build with `--features gpu-cuda` / `--features gpu-hip` and install the matching runtime; `--gpu-backend auto` picks CUDA → HIP → CPU among the backends compiled in.
- **No automatic speaker diarization**: speaker labels are assigned manually per input file (`--set-speaker-names`); the downloader lists `tinydiarize` models but diarization is not wired into transcription.
- **YouTube fallback is conservative**: an explicitly requested caption language is never replaced. HTTP 403/429, network failures, and malformed caption files fail instead of starting audio transcription. Audio downloads are size- and time-limited; the ffmpeg decoder has a deadline, but Whisper inference runs in-process and has no hard timeout.
- **A transcript can be incomplete at a window boundary**: whisper does not always report a segment for the beginning of the audio it is given. PolyScribe decodes such a region again from a slightly different position, and warns on stderr when the region still holds no text. The check that decides whether a region carries speech is a level and a share of frames, not a voice activity detector, so the pipeline cannot promise that every word of a recording is in the transcript.

## Development

```bash
cargo build                 # build the workspace
cargo test --workspace      # run all tests (no network/models/GPU needed)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

See `CONTRIBUTING.md` for the full checklist and `docs/development.md` for details.

## Troubleshooting & docs

- `docs/faq.md` — common issues and solutions
- `docs/releases.md` — branch model and release procedure
- `docs/usage.md` — complete CLI reference and workflows
- `docs/development.md` — build, test, and contribute locally
- `docs/design.md` — architecture overview and decisions
- `docs/release-packaging.md` — packaging notes for distributions
- `docs/ci.md` — CI checklist and job outline
- `CONTRIBUTING.md` — PR checklist and workflow
- `CHANGELOG.md` — notable changes
- `docs/benchmarks/` — chunked transcription, live-window quality, and backend measurements

## License

This project is licensed under the MIT License — see the LICENSE file for details.
