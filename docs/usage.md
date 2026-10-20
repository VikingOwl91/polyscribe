# Usage

PolyScribe is a command-line tool. Run `polyscribe -h` at any time to see the latest help.

Common patterns
- Single file to transcript (JSON + TOML + SRT):
  - `polyscribe -o output --language en path/to/audio_or_video.mp3`
- Multiple files → merged transcript:
  - `polyscribe -m -o output merged path/a.mp3 path/b.mp4`
- Multiple files → both merged and separate outputs:
  - `polyscribe --merge-and-separate -o output path/a.json path/b.json`
- Prompt for speaker names per input (manual labeling; no automatic diarization):
  - `polyscribe --set-speaker-names -o output path/a.mp3 path/b.mp4`

CLI reference
- Positional arguments:
  - `inputs`: One or more `.json` transcripts or media files (audio/video). Media files are decoded to 16 kHz mono PCM via ffmpeg.
  - With at least two inputs and no `-o`, a trailing argument that does not exist on disk is treated as the output path.
- Flags:
  - `-o, --output FILE_OR_DIR`
    - Output path. In the default (separate) mode and in `--merge-and-separate` mode this is a directory. In `-m` (merge) mode it is a file base name. File names get a `YYYY-MM-DD_` date prefix. If omitted, JSON prints to stdout.
  - `-m, --merge`
    - Merge all inputs into a single output instead of one output per input.
  - `--merge-and-separate`
    - Write both a merged output and separate outputs per input (requires `-o` directory).
  - `--set-speaker-names`
    - Prompt for a speaker label per input (manual labeling; PolyScribe does not perform automatic speaker diarization).
  - `--language LANG`
    - Language hint (e.g., `en`, `de`). Required when transcribing audio. English-only models reject non-`en` hints.
  - `--gpu-backend [auto|cpu|cuda|hip]`
    - Choose the backend among those compiled into the binary. Default is `auto` (prefers CUDA → HIP → CPU). Requesting a GPU backend that was not compiled in is an error that explains how to rebuild.
  - `--download-models`
    - Launch the interactive model downloader (lists Hugging Face models; multi-select to download).
  - `--update-models`
    - Verify/update local models by comparing sizes and SHA-256 hashes with the upstream manifest.
  - `-v, --verbose` (repeatable)
    - Increase log verbosity; use `-vv` for very detailed logs.
  - `-q, --quiet`
    - Suppress non-error logs to stderr; does not affect stdout outputs.
  - `--no-interaction`
    - Disable all interactive prompts (for CI). Combine with env vars to control behavior.
  - Subcommands:
    - `completions <shell>`: Write a shell completion script to stdout.
    - `man`: Write a man page to stdout.

Expected outputs
- For each processed input or merged group, PolyScribe produces:
  - A JSON transcript file with segments (`id`, `speaker`, `start`, `end`, `text`).
  - A TOML transcript file with the same content.
  - An SRT subtitle file with timestamps and text (`speaker:`-prefixed when a label is set).
- When `-o` is used with a directory, outputs are written into that directory with a `YYYY-MM-DD` prefix.
- When no `-o` is given, pretty-printed JSON is written to stdout (JSON only).

Typical workflows
1) Single file → transcript:
- `polyscribe -o output --language en media/example.mp3`

2) Multiple files → merged transcript:
- `polyscribe -m -o output merged --language en media/a.mp3 media/b.mp4`

3) Multiple files → both merged and individual transcripts:
- `polyscribe --merge-and-separate -o output media/a.json media/b.json`

4) Video → extract audio automatically:
- `polyscribe -o output --language en videos/talk.mp4`
(Requires ffmpeg on PATH.)

Model locations
- Development builds (debug): `./models` is used by default.
- Packaged releases: `$XDG_DATA_HOME/polyscribe/models` or `~/.local/share/polyscribe/models`.
- Override:
  - `POLYSCRIBE_MODELS_DIR=/path/to/models`
  - `WHISPER_MODEL=/path/to/specific_model.bin` (forces exact model file).

Environment variables
- `POLYSCRIBE_MODELS_DIR`: Override the default models directory.
- `WHISPER_MODEL`: Point directly to a model file.
- `XDG_DATA_HOME`/`HOME`: Used to resolve the default model path for release builds.
- `CI`/`GITHUB_ACTIONS`: When set, PolyScribe assumes non-TTY in some paths and may avoid prompts.
- Test-only toggles (used by our tests; not recommended in production):
  - `POLYSCRIBE_TEST_FORCE_CUDA=1`
  - `POLYSCRIBE_TEST_FORCE_HIP=1`

Notes
- GPU support is a build-time property: build with `--features gpu-cuda` or `--features gpu-hip` (see development.md). A Vulkan backend is not currently wired up; `--gpu-backend vulkan` is rejected.
- English-only models cannot be used with non-English language hints.
- Network access happens only for `--download-models` and `--update-models` (Hugging Face over HTTPS). All other operations are fully offline.
