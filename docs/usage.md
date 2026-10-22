# Usage

PolyScribe is a command-line tool. Run `polyscribe -h` at any time to see the latest help.

The CLI is organized into subcommands: `transcribe`, `record`, `models`, `plugins`, `completions`, and `man`. Global flags `--json` and `--quiet` are accepted before or after the subcommand; `-v/--verbose`, `--no-interaction`, and `--no-progress` are top-level flags (before the subcommand).

## Common patterns

- Single file to transcript (JSON + TOML + SRT):
  - `polyscribe transcribe --language en -o output path/to/audio_or_video.mp3`
- Multiple files → merged transcript:
  - `polyscribe transcribe -m -o output merged path/a.mp3 path/b.mp4`
- Multiple files → both merged and separate outputs:
  - `polyscribe transcribe --merge-and-separate -o output path/a.json path/b.json`
- Prompt for speaker names per input (manual labeling; no automatic diarization):
  - `polyscribe transcribe --set-speaker-names -o output path/a.mp3 path/b.mp4`

## transcribe

- Positional arguments:
  - `inputs`: One or more `.json` transcripts or media files (audio/video), required. Media files are decoded to 16 kHz mono PCM via ffmpeg.
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
    - Choose the backend. Default is `auto` (prefers CUDA → HIP → CPU).
    - `cpu` always runs on the CPU.
    - `cuda` and `hip` require the matching backend to be compiled in **and** a
      matching GPU device to be present. If the backend is not compiled in, the
      error says how to rebuild; if it is compiled in but no device is available,
      the error says how to check the runtime and the device. An explicit GPU
      request is never silently continued on the CPU.
    - `auto` falls back to the CPU when no GPU device is available and says so.
    - The device list comes from the same registry ggml and whisper.cpp use, so
      the check reflects what the runtime actually found. Run with `-vv` to see
      the detected backends and devices.
- Warnings:
  - whisper does not always report a segment for the beginning of the audio it
    is given. PolyScribe decodes such a region again from a slightly
    different position. If the region still holds no text, the command writes a
    warning to stderr that names the file and the time range and says that the
    transcript may be incomplete. The exit status stays the same: the
    transcription succeeded, and the warning is the only hint that something is
    missing.
  - The per-window diagnostics are opt-in: set `POLYSCRIBE_WINDOW_TRACE=1` to
    get the sample range of every window and every region that whisper did not
    report on stderr.

## record

Record the microphone and transcribe it while the recording runs. The audio
is captured from PipeWire and handed to the transcription pipeline while it
arrives, so the transcript grows during the recording.

```bash
# Record the default source of the session, detect the language
polyscribe record

# A language, a device and a shorter window
polyscribe record --language de --device voice_arctis_mono --window 5
```

- Flags:
  - `--model MODEL` — a path to a Whisper model file, or the name of a model
    in the models directory. Without it the model is found the way
    `transcribe` finds it (`WHISPER_MODEL`, the models directory, the model
    cache).
  - `--language LANG` — the language of the recording, e.g. `en`, `de`.
    Without it the language is detected from the audio.
  - `--device NODE` — the PipeWire node to record from, e.g.
    `voice_arctis_mono`. Without it the default source of the session is
    used. A node that does not exist is refused: `pw-record` would record
    from the default source instead and would not say so.
  - `--window SECONDS` — the length of a live window. The default is 5.
    Shorter windows show the text sooner and need more decoding.
  - `-o, --output FILE` — write the transcript to this file as well. The
    file is written while the recording runs.
  - `--gpu-backend [auto|cpu|cuda|hip]` — the backend, as in `transcribe`.
- The capture needs a running PipeWire session and the program `pw-record`
  (the Arch package `pipewire-audio`).
- The output contract:
  - The standard output carries the transcript: one finalized segment per
    line, as the segments become final. With `--json` the lines are JSON
    objects with the fields `speaker`, `start`, `end` and `text`.
  - The standard error carries the readiness of the capture, the text that
    is still provisional (prefixed with `… `), and the diagnostics of the
    run. `--quiet` and `--json` suppress the status messages.
- A segment that is still provisional may still change: the next window
  decodes the speech at the end of the window again, with the context it
  needs. Only a finalized segment is final, and it is never reported twice.
- Ctrl+C stops the recording. The segments that were finalized are kept and
  the exit status is 0. A capture that failed exits with a nonzero status.
- The language detection: the first windows of a run are longer, so that
  the detection has enough speech to work with. A language is only pinned
  when the window carries speech; a language that was detected in silence
  is a guess, and the guess is named on stderr.

## Model loading

The Whisper model is loaded once per invocation and reused for every input
file. On a GPU build the weights are uploaded into VRAM once, which matters
when several files are transcribed in a single run. Each file still gets its
own inference state, so the outputs, timestamps and error handling of the
files stay independent. The model is released when the command finishes.

## models

Manage Whisper models (downloads come from Hugging Face, verified by SHA-256 when the hash is known):

- `models ls` — list installed models (from the manifest).
- `models add <repo> <file>` — add or update a model (e.g., `models add ggerganov/whisper.cpp ggml-tiny.en.bin`); the alias is derived from repo and file.
- `models rm <alias>` — remove a model by alias.
- `models verify <alias>` — verify a model file's SHA-256.
- `models update` — update all installed models (HEAD + ETag; skips unchanged).
- `models download` — launch the interactive model downloader (lists Hugging Face models; two-stage select; skipped in `--no-interaction`).
- `models clear-cache` — clear the cached Hugging Face manifest (force a fresh fetch).
- `models gc` — garbage-collect unreferenced files and stale manifest entries.
- `models search <repo> [--query <substr>]` — search a repo for `.bin`/`.gguf` files.
- `models` subcommands accept `--concurrency N` and `--limit-rate BYTES_PER_SEC`.
- Exit codes: `0` ok, `2` not found, `3` network error, `4` verification failed.

Model files live in `$XDG_CACHE_HOME/polyscribe/models` (override `POLYSCRIBE_CACHE_DIR`); the manifest is `$XDG_CONFIG_HOME/polyscribe/models.json` (override `POLYSCRIBE_CONFIG_DIR`). Set `HF_TOKEN` for authenticated requests.

## plugins

- `plugins list` — discover plugins on `PATH` and in `$XDG_DATA_HOME/polyscribe/plugins`.
- `plugins info <name>` — show a plugin's capabilities (JSON).
- `plugins run <name> <command> [--payload '<json>']` — run a plugin command: the host sends one JSON-RPC (PSP/1) request over the plugin's stdin and forwards its NDJSON stdout stream.

## completions / man

- `completions <bash|zsh|fish|powershell|elvish>` — write a shell completion script to stdout.
- `man` — write a man page to stdout.

## Global flags

- `--json` — emit machine-readable JSON on stdout; suppress decorative logs.
- `--quiet` — reduce log chatter (errors only).
- `-v, --verbose` (repeatable, top-level) — increase log verbosity; `-vv` shows very detailed logs.
- `--no-interaction` (top-level) — never prompt; suitable for CI.
- `--no-progress` (top-level) — disable progress bars/spinners.

## Expected outputs

- For each processed input or merged group, `transcribe` produces:
  - A JSON transcript file with segments (`id`, `speaker`, `start`, `end`, `text`).
  - A TOML transcript file with the same content.
  - An SRT subtitle file with timestamps and text (`speaker:`-prefixed when a label is set).
- When `-o` is used with a directory, outputs are written into that directory with a `YYYY-MM-DD` prefix.
- When no `-o` is given, pretty-printed JSON is written to stdout (JSON only).

## Typical workflows

1) Single file → transcript:
- `polyscribe transcribe --language en -o output media/example.mp3`

2) Multiple files → merged transcript:
- `polyscribe transcribe -m -o output merged --language en media/a.mp3 media/b.mp4`

3) Multiple files → both merged and individual transcripts:
- `polyscribe transcribe --merge-and-separate -o output media/a.json media/b.json`

4) Video → extract audio automatically:
- `polyscribe transcribe --language en -o output videos/talk.mp4`
(Requires ffmpeg on PATH.)

5) Prepare a model non-interactively (good for CI):
- `polyscribe models add ggerganov/whisper.cpp ggml-tiny.en.bin`

## Model locations

- Transcription models (used by `transcribe`): `$XDG_DATA_HOME/polyscribe/models` or `~/.local/share/polyscribe/models`, the same in debug and release builds. Models installed with `models add` live in the cache dir (`$XDG_CACHE_HOME/polyscribe/models`) and are found automatically as a fallback.
- Override:
  - `POLYSCRIBE_MODELS_DIR=/path/to/models`
  - `WHISPER_MODEL=/path/to/specific_model.bin` (forces exact model file).

## Environment variables

- `POLYSCRIBE_MODELS_DIR`: Override the transcription models directory.
- `WHISPER_MODEL`: Point directly to a model file.
- `POLYSCRIBE_CACHE_DIR` / `POLYSCRIBE_CONFIG_DIR`: Override the `models` subcommand cache/config locations.
- `POLYSCRIBE_NO_CACHE_MANIFEST=1`: Disable the manifest cache.
- `POLYSCRIBE_MANIFEST_TTL_SECONDS`: Manifest cache TTL (default 24 h).
- `POLYSCRIBE_HF_REPO`: Override the default Hugging Face repo (`ggerganov/whisper.cpp`).
- `HF_TOKEN`: Hugging Face access token for authenticated requests.
- `XDG_DATA_HOME` / `XDG_CACHE_HOME` / `XDG_CONFIG_HOME` / `HOME`: Used to resolve default paths.
- Test-only toggles (used by our tests; not recommended in production):
  - `POLYSCRIBE_TEST_FORCE_CUDA=1` / `=0`
  - `POLYSCRIBE_TEST_FORCE_HIP=1` / `=0`

## Notes

- Supported platforms: Linux x86_64 is the tested platform (CI builds, tests and transcribes on CPU there). macOS is expected to work but untested. Windows is not supported and does not compile.
- GPU support is a build-time property: build with `--features gpu-cuda` or `--features gpu-hip` (see development.md). A Vulkan backend is not wired up; `--gpu-backend vulkan` is rejected. GPU support is experimental and no GPU runtime is exercised in CI.
- English-only models cannot be used with non-English language hints.
- Network access happens only for the `models` subcommands (Hugging Face over HTTPS). `transcribe`, `plugins`, `completions`, and `man` are fully offline.
