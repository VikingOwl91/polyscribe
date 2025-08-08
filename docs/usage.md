# Usage

PolyScribe is a command-line tool. Run `polyscribe -h` at any time to see the latest help.

Common patterns
- Single file to transcript (JSON + SRT):
  - polyscribe -o output path/to/audio_or_video.mp3
- Multiple files → merged transcript:
  - polyscribe -m -o output merged path/a.mp3 path/b.mp4
- Multiple files → both merged and separate outputs:
  - polyscribe --merge-and-separate -o output path/a.json path/b.json
- Prompt for speaker names per input:
  - polyscribe --set-speaker-names -o output path/a.mp3 path/b.mp4

CLI reference
- Positional arguments:
  - inputs: One or more .json transcripts or media files (audio/video). When media files are given, PolyScribe extracts audio via ffmpeg.
- Flags:
  - -o, --output FILE_OR_DIR
    - Output base path. For directories, date prefix is added and both .json and .srt are created. If omitted, JSON prints to stdout.
  - -m, --merge
    - Merge all inputs into a single output instead of one output per input.
  - --merge-and-separate
    - Write both a merged output and separate outputs (requires -o directory).
  - --set-speaker-names
    - Prompt for a speaker label per input (useful for multi-speaker datasets).
  - --language LANG
    - Language hint (e.g., en, de). English-only models reject non-en hints.
  - --gpu-backend [auto|cpu|cuda|hip|vulkan]
    - Choose runtime backend. Default is auto (prefers CUDA → HIP → Vulkan → CPU), depending on detection.
  - --gpu-layers N
    - Number of layers to offload to the GPU when supported.
  - --download-models
    - Launch interactive model downloader (lists Hugging Face models; multi-select to download).
  - --update-models
    - Verify/update local models by comparing sizes and hashes with the upstream manifest.
  - -v, --verbose (repeatable)
    - Increase log verbosity; use -vv for very detailed logs.
  - -q, --quiet
    - Suppress non-error logs to stderr; does not affect stdout outputs.
  - --no-interaction
    - Disable all interactive prompts (for CI). Combine with env vars to control behavior.
  - Subcommands:
    - completions <shell>: Write shell completion script to stdout.
    - man: Write a man page to stdout.

Expected outputs
- For each processed input or merged group, PolyScribe produces:
  - A JSON transcript file with segments (id, speaker, start, end, text).
  - An SRT subtitle file with timestamps and text (speaker: prefixed when provided).
- When -o is used with a directory, outputs are written into that directory with a YYYY-MM-DD prefix.

Typical workflows
1) Single file → transcript:
- polyscribe -o output media/example.mp3

2) Multiple files → merged transcript:
- polyscribe -m -o output merged media/a.mp3 media/b.mp4 media/c.wav

3) Multiple files → both merged and individual transcripts:
- polyscribe --merge-and-separate -o output media/a.json media/b.json

4) Video → extract audio automatically:
- polyscribe -o output videos/talk.mp4
(Requires ffmpeg on PATH.)

Model locations
- Development builds (debug): ./models is used by default.
- Packaged releases: $XDG_DATA_HOME/polyscribe/models or ~/.local/share/polyscribe/models.
- Override:
  - POLYSCRIBE_MODELS_DIR=/path/to/models
  - WHISPER_MODEL=/path/to/specific_model.bin (forces exact model file).

Environment variables
- POLYSCRIBE_MODELS_DIR: Override default models directory.
- WHISPER_MODEL: Point directly to a model file.
- XDG_DATA_HOME/HOME: Used to resolve default model path for release builds.
- CI/GITHUB_ACTIONS: When set, PolyScribe assumes non-TTY in some paths and may avoid prompts.
- Test-only toggles (used by our tests; not recommended in production):
  - POLYSCRIBE_TEST_FORCE_CUDA=1
  - POLYSCRIBE_TEST_FORCE_HIP=1
  - POLYSCRIBE_TEST_FORCE_VULKAN=1

Notes
- GPU selection depends on both build features and runtime detection. Build with the corresponding cargo features (see development.md) for CUDA/HIP/Vulkan support.
- English-only models cannot be used with non-English language hints.
