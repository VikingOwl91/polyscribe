# PolyScribe

PolyScribe is a command-line tool for transcribing local audio and video, recording live audio, merging JSON transcripts, and transcribing YouTube audio. Whisper inference runs on your machine. YouTube and model downloads use external network services, and the optional Mistral cloud engine uploads audio to Mistral.

## Key capabilities

- Transcribe audio and video with a local Whisper model.
- Record from a PipeWire microphone and transcribe while recording.
- Fetch YouTube audio without requesting subtitles, then transcribe it locally; text output carries video metadata as YAML frontmatter.
- Optionally transcribe files with Mistral's Voxtral through the cloud API (`--engine voxtral-cloud`).
- Optionally run a locally installed Voxtral wrapper before the Whisper fallback.
- Write JSON, TOML, and SRT files, or pipe JSON and transcript text through stdout.
- Manage Whisper models and run plugins that implement the PolyScribe protocol.

## Requirements

- Rust 1.88 or newer to build from source.
- A C/C++ toolchain, CMake, and libclang to build whisper.cpp bindings.
- `ffmpeg` on `PATH` for local audio/video transcription.
- `yt-dlp` on `PATH` for YouTube audio acquisition.
- A Whisper model for transcription. Models are not bundled.
- For `polyscribe record`, a running PipeWire session and `pw-record` on `PATH`.

## Installation

Build the CPU-default CLI from the repository:

```bash
cargo install --path crates/polyscribe-cli --locked
```

Or build it without installing:

```bash
cargo build --release --locked -p polyscribe-cli
# Binary: target/release/polyscribe
```

The build above is CPU-only. For an AMD ROCm/HIP build use:

```bash
cargo build --release --locked -p polyscribe-cli --features gpu-hip
```

For NVIDIA CUDA use `--features gpu-cuda` instead. GPU support is a build-time
feature; a CPU-only binary reports `Compiled-in backends: []` at `-vv` and uses
the CPU even when ROCm is installed.

## Quick start

Install a small English Whisper model. This command downloads it from Hugging Face:

```bash
polyscribe models add ggerganov/whisper.cpp ggml-tiny.en.bin
```

Transcribe a local audio file to JSON on stdout:

```bash
polyscribe transcribe --language en interview.wav > transcript.json
```

Write JSON, TOML, and SRT files into a directory:

```bash
polyscribe transcribe --language en -o transcripts interview.wav
```

Transcribe YouTube audio as SRT, or record from the default PipeWire source:

```bash
polyscribe youtube 'https://www.youtube.com/watch?v=VIDEO_ID' --language en --format srt
polyscribe record
```

## YouTube audio transcription

`polyscribe youtube URL` checks its local transcript cache first. On a cache miss, `yt-dlp` fetches video metadata and the best available audio. PolyScribe never requests or downloads subtitles. The audio is transcribed locally and the transcript is cached independently of its output format. Use `--refresh` to bypass the cache and acquire a new transcript.

`txt` output starts with a YAML frontmatter block (title, source URL, video ID, channel, upload date, duration, language, transcription time and engine); `--no-frontmatter` writes plain text, and `json` carries the same fields. See [YouTube output](docs/usage.md#output-and-errors).

YouTube transcription needs network access to YouTube and a working `yt-dlp` installation. HTTP 403/429 responses and other download failures stop before transcription. See the [CLI usage guide](docs/usage.md#youtube) for cache identity, limits, errors, and options.

If YouTube requests bot verification, install a Netscape-format export as `~/.config/polyscribe/youtube/cookies.txt` (or below `$XDG_CONFIG_HOME`). Use directory mode `0700` and file mode `0600`. PolyScribe automatically uses this file through a temporary copy. On Unix, transcripts are cached separately for each cookie-file version; repeat requests reuse the transcript without downloading or transcribing again. `--cookies /path/to/cookies.txt` overrides the stored file; `--no-cookies` selects the anonymous cache. See [cookie export and setup](docs/usage.md#set-up-youtube-cookies-once). PolyScribe never reads browser cookies automatically.

## Optional Mistral cloud engine

`polyscribe transcribe --engine voxtral-cloud` sends the audio to Mistral's Voxtral transcription API. It is part of every build (CPU, CUDA, and HIP), needs no GPU, and is only used when you select it. `--engine auto` uses it only if the `mistral_token` file exists (a bare `MISTRAL_API_KEY` does not count) and falls back to local Whisper with a warning if the cloud fails; the default engine stays `whisper`. Audio leaves your machine, so PolyScribe prints a one-line notice before the first upload (silence it with `--yes`). The API key is read from `MISTRAL_API_KEY` or, if that is unset, from `~/.config/polyscribe/mistral_token` (keep it mode `0600`); it is never logged. See [transcription engines](docs/usage.md#transcription-engines) for options, retries, and the AMD note.

## Optional Voxtral wrapper and Whisper fallback

Voxtral is not bundled or installed by PolyScribe. To use it, set `POLYSCRIBE_VOXTRAL` to an absolute path to a local executable that implements PolyScribe's protocol-v1 wrapper contract. When configured, PolyScribe tries that executable first. If it is unset, times out, exits unsuccessfully, or returns invalid output, PolyScribe makes one local Whisper attempt with the same downloaded audio.

Whisper fallback requires an installed local model. Without a configured Voxtral executable, local Whisper is used directly. See [Voxtral and fallback configuration](docs/usage.md#optional-voxtral-wrapper) for the wrapper interface and model lookup rules.

## Supported output formats

- `transcribe`: JSON, TOML, and SRT files when writing to a directory; pretty JSON on stdout when no output directory is selected.
- `youtube`: choose one format per run with `--format txt|srt|json` (default `txt`).
- `record`: finalized transcript segments as plain-text lines, or JSON lines with the global `--json` flag.

The commands have different output schemas. See [output behavior and piping](docs/usage.md#stdout-stderr-and-piping) before consuming output in a script.

## Configuration overview

- `WHISPER_MODEL` selects a specific local model. Otherwise, PolyScribe searches its transcription model directory and the model-download cache.
- `POLYSCRIBE_MODELS_DIR` overrides the transcription model directory.
- `POLYSCRIBE_YTDLP` selects a specific `yt-dlp` file; otherwise it is found on `PATH`.
- `POLYSCRIBE_VOXTRAL` selects the optional Voxtral wrapper executable.
- `MISTRAL_API_KEY` or the `mistral_token` file in the config directory provides the key for `--engine voxtral-cloud`.
- `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and `XDG_CONFIG_HOME` determine default model, transcript-cache, and configuration locations.
- `HF_TOKEN` is used for authenticated Hugging Face model requests when set.

The full environment-variable reference and defaults are in [usage](docs/usage.md#configuration-and-environment).

## Documentation

- [CLI usage and command reference](docs/usage.md)
- [Architecture and trust boundaries](docs/design.md)
- [FAQ](docs/faq.md)
- [Development and testing](docs/development.md)
- [CI checks](docs/ci.md)
- [Release and branch process](docs/releases.md)
- [Packaging notes](docs/release-packaging.md)
- [Changelog](CHANGELOG.md)
- [Backlog](TODO.md)

## Limitations and project maturity

PolyScribe is alpha software. Linux x86_64 is the CI-tested platform. The workspace does not compile on Windows yet; macOS is not covered by CI. CUDA and ROCm/HIP builds are optional and experimental, and CI does not verify GPU runtime behavior.

The Mistral cloud engine is covered by mock-server tests and one live run, not by CI. There is no automatic speaker diarization. The live microphone command requires PipeWire. YouTube transcription is audio-only, and a failed download is not retried through a caption path. Available commands are listed in `polyscribe --help`; older `stt` and `ytt` command names are not provided.

## License

PolyScribe is licensed under the MIT License. See [LICENSE](LICENSE).
