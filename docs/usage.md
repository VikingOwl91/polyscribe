# CLI usage

This is the user-facing reference for the `polyscribe` command. The executable's `--help` output is generated from the same argument definitions. Run `polyscribe --help` or `polyscribe <command> --help` for a short summary.

## Commands and global options

```text
polyscribe [OPTIONS] <COMMAND>
polyscribe transcribe [OPTIONS] <INPUTS>...
polyscribe youtube [OPTIONS] <URL>
polyscribe record [OPTIONS]
polyscribe models <COMMAND>
polyscribe plugins <COMMAND>
polyscribe completions <bash|zsh|fish|powershell|elvish>
polyscribe man
```

Commands are `transcribe`, `youtube`, `record`, `models`, `plugins`, `completions`, and `man`. Clap also provides `help` and `--version`.

Global options:

| Option | Default | Effect |
|---|---|---|
| `--json` | off | Selects machine-readable output where a command provides it and suppresses decorative logs. It does not select YouTube's output format. |
| `--quiet` | off | Reduces informational output. Errors and warnings remain visible. |
| `-v`, `--verbose` | off | Increase diagnostic detail; repeat for more detail. These top-level options go before the subcommand. |
| `--no-interaction` | off | Do not prompt. This top-level option goes before the subcommand. |
| `--no-progress` | off | Disable progress bars and spinners. This top-level option goes before the subcommand. |

`--json` and `--quiet` may appear before or after a subcommand. A command's transcript or data stays on stdout; diagnostics go to stderr. See [stdout, stderr, and piping](#stdout-stderr-and-piping).

When stderr is a terminal, `transcribe` and `youtube` show a progress bar for the current transcription with elapsed time and an ETA derived from completed inference windows. `--quiet` and `--no-progress` disable it. The transcript stays on stdout.

## `transcribe`

Transcribe one or more local audio/video files, load JSON transcript inputs, and optionally merge entries. Audio/video is decoded with `ffmpeg`; Whisper inference is local. This command does not download a model or contact a remote service.

### Syntax and options

```text
polyscribe transcribe [OPTIONS] <INPUTS>...
```

Accepted media extensions are `mp3`, `wav`, `m4a`, `mp4`, `aac`, `flac`, `ogg`, `wma`, `webm`, `mkv`, `mov`, `avi`, `m4b`, `3gp`, `opus`, `aiff`, and `alac`, case-insensitive. JSON inputs use the shape `{"segments":[{"start":0.0,"end":1.2,"text":"spoken words"}]}`. Other JSON fields are ignored.

| Option | Default | Effect |
|---|---|---|
| `-o`, `--output <OUTPUT>` | stdout or error, depending on mode | In separate mode, names an output directory. In merge mode, names an output base path. In merge-and-separate mode, names the required output directory. |
| `-m`, `--merge` | off | Merge all inputs into one transcript. |
| `--merge-and-separate` | off | Write one merged transcript and a separate transcript for each input. Requires `-o OUTPUT_DIR`. |
| `--language <LANGUAGE>` | unset | Required if any input is audio/video. Pass a Whisper language hint such as `en` or `de`. JSON-only input does not need it. File transcription does not auto-detect language. |
| `--set-speaker-names` | off | Prompt for one manual speaker label per input. This is labeling, not diarization. |
| `--gpu-backend <GPU_BACKEND>` | `auto` | Choose `auto`, `cpu`, `cuda`, or `hip`. GPU values require a matching compiled backend and available device. |
| `--gpu-layers <GPU_LAYERS>` | `0` | Hidden and reserved. It currently has no supported effect; omit it. |

### Examples and output

One audio input with no `-o` writes pretty JSON to stdout:

```bash
polyscribe transcribe --language en interview.wav > interview.json
```

With `-o` in the default separate mode, PolyScribe creates JSON, TOML, and SRT files in that directory. Names include a `YYYY-MM-DD_` date prefix:

```bash
polyscribe transcribe --language en -o transcripts interview.wav
```

Merge JSON transcripts and write all three formats using `merged` as the output base name:

```bash
polyscribe transcribe --merge -o transcripts/merged input-a.json input-b.json
```

Write both individual outputs and a merged output:

```bash
polyscribe transcribe --merge-and-separate -o transcripts input-a.json input-b.json
```

With multiple inputs and no `-o`, separate mode fails because there is no directory for the files. In merge mode, omitting `-o` writes the merged JSON transcript to stdout. A historical shorthand treats a final, non-existent positional path as the output path when there are at least two inputs and `-o` is absent; explicit `-o` is clearer and avoids ambiguity.

Common errors:

- Audio/video input without `--language`: add a language hint. Automatic language detection is available in `record` and `youtube`, not this command.
- `ffmpeg` missing or unable to decode the file: install `ffmpeg`, put it on `PATH`, and check that the input is readable media.
- No Whisper model found: install one with `polyscribe models add ...` or set `WHISPER_MODEL` to an existing model file.
- Explicit CUDA/HIP request unavailable: use `--gpu-backend cpu` or build with the matching feature and check the runtime/device. `auto` uses a detected compiled backend when available and otherwise uses CPU.
- `--merge-and-separate` without `-o`: provide an output directory.

## `youtube`

Transcribe the audio of a YouTube video. The default is audio-only. PolyScribe checks the local transcript cache first. On a cache miss, it runs `yt-dlp` to fetch metadata and download audio, then uses a configured Voxtral wrapper or local Whisper. It never requests or downloads captions/subtitles.

### Syntax and options

```text
polyscribe youtube [OPTIONS] <URL>
```

| Option | Default | Effect |
|---|---|---|
| `--language <LANGUAGE>` | unset, meaning automatic selection | Hint passed to Voxtral and Whisper. Accepted values are at most 64 ASCII letters, digits, hyphens, or underscores. `auto` is not accepted; omit the option for automatic selection. |
| `--format <FORMAT>` | `txt` | Select one output format: `txt`, `srt`, or `json` (lowercase). |
| `-o`, `--output <OUTPUT>` | stdout | Write the selected format to this file instead of stdout. |
| `--refresh` | off | Ignore a cached transcript, fetch audio, and transcribe again. The new result is cached when caching is available. |
| `--prefer-manual` | on | Deprecated compatibility flag. No effect; captions are not used. |
| `--audio-only` | off | Compatibility alias. Audio is already the default. |
| `--captions-only` | off | Unsupported. Fails before any network or process invocation. It conflicts with `--audio-only`. |

Supported URLs use HTTPS YouTube forms such as `youtube.com/watch`, `youtu.be`, `youtube.com/embed`, and `youtube.com/shorts`.

### Examples

```bash
# Transcript text to stdout, automatic language selection
polyscribe youtube 'https://www.youtube.com/watch?v=VIDEO_ID'

# SRT file, with a language hint
polyscribe youtube 'https://youtu.be/VIDEO_ID' --language de --format srt -o talk.srt

# JSON to stdout, ignoring the cached transcript
polyscribe youtube 'https://www.youtube.com/watch?v=VIDEO_ID' --format json --refresh
```

`--json` is a global output/logging option and does not change `--format`. For JSON transcript content, use `--format json`.

### Output and errors

When no output file is given, the selected transcript is written to stdout with a trailing newline. `txt` contains cue text, `srt` contains numbered timed cues, and `json` contains the source URL, video ID, available title/uploader/duration, requested language, Voxtral-reported language when available (or the Whisper hint), engine/model, cue count, normalization statistics, and timed cues. The audio-only JSON also retains `caption_language`, `is_automatic`, and `is_translated` fields as `null`. The command writes a short source summary to stderr, including when stdout is JSON or `--quiet` is used. `-o` writes the transcript file and reports the destination on stderr.

Common failures are reported on stderr with a category such as `invalid_url`, `invalid_language`, `unsupported_format`, `tool_unavailable`, `rate_limited`, `forbidden`, `network`, `audio_download`, or `whisper`:

- Invalid URL or language: use a supported HTTPS YouTube URL and omit `--language` for automatic selection.
- `yt-dlp` unavailable: install it and put it on `PATH`, or set `POLYSCRIBE_YTDLP` to its file path.
- HTTP 403/429 or another download error: check that the video is available to your network/account and try again later. PolyScribe does not start transcription after a download failure and does not retry through captions.
- Voxtral failure followed by a Whisper error: install a Whisper model or point `WHISPER_MODEL` at a valid local model. See [Whisper model lookup](#whisper-model-lookup).
- Unsupported caption mode: remove `--captions-only`; caption transcription is not implemented.
- Invalid output path or unwritable destination: use a path without `..` components or control characters and ensure its parent directory exists and is writable.

### Cache behavior and invalidation

YouTube transcripts are stored under `$XDG_CACHE_HOME/polyscribe/youtube/v3/`, or `~/.cache/polyscribe/youtube/v3/` when `XDG_CACHE_HOME` is unset and `HOME` is available. The cached cues are reused across TXT, SRT, and JSON renders.

Cache identity includes the cache/schema, normalization and audio-policy versions, video ID, requested language, and configured Voxtral executable identity. Voxtral identity includes its configured path and resolved executable/symlink metadata. A Whisper-backed result also records the model identity and is rejected when the selected model's file metadata changes. Changing the language, Voxtral executable, relevant version, or Whisper model invalidates the matching entry. Unsafe, unreadable, invalid, or unavailable cache storage is ignored and the command continues without caching. `--refresh` bypasses lookup; there is no separate YouTube cache-clear subcommand.

### Network, limits, and temporary files

YouTube processing is not fully offline: `yt-dlp` contacts YouTube and related hosts to retrieve metadata and audio. Transcription itself runs locally. Metadata retrieval is limited to 120 seconds and 50 MiB of stdout. Audio download is limited to 15 minutes and 250 MiB. Voxtral and the ffmpeg audio decoder each have a 60-minute process deadline. The decoder deadline does not interrupt a Whisper inference call already running in-process. Temporary downloads use a private temporary directory and are removed on normal completion and handled errors. The YouTube command does not install `SIGINT` or `SIGTERM` handlers; abrupt signal termination can bypass cleanup and leave temporary files. A subprocess descendant that leaves its process group may also survive termination.

## Optional Voxtral wrapper

Voxtral is not bundled, downloaded, or configured automatically. Set `POLYSCRIBE_VOXTRAL` to an absolute path to a local executable to enable it:

```bash
export POLYSCRIBE_VOXTRAL="$HOME/.local/bin/polyscribe-voxtral"
polyscribe youtube 'https://youtu.be/VIDEO_ID' --language en
```

PolyScribe invokes the file directly, without a shell, with these arguments:

```text
--protocol-version 1 --input /absolute/path/to/audio
[--language en]
```

The wrapper must write one JSON object to stdout:

```json
{"protocol_version":1,"engine":"voxtral","model":"local-model","language":"en","cues":[{"start_ms":0,"end_ms":1200,"text":"Hello."}]}
```

`language` is a language code or `null`; `cues` must be a non-empty array of timed text cues. The wrapper may write diagnostics to stderr, but PolyScribe does not forward child stderr to the terminal. It reports bounded typed failures such as a nonzero exit code, timeout, oversized output, or invalid protocol response without printing arbitrary child output.

If the variable is unset or the wrapper fails, PolyScribe makes exactly one local Whisper attempt with the same audio. It does not install the wrapper or Whisper model. Local Whisper requires a valid model file; see the next section.

## `record`

Record from the current PipeWire session and transcribe while audio arrives. It uses local Whisper and does not contact a remote service.

### Syntax and options

```text
polyscribe record [OPTIONS]
```

| Option | Default | Effect |
|---|---|---|
| `--model <MODEL>` | automatic model lookup | Use a model file path or a model name in the transcription model directory/cache. |
| `--language <LANGUAGE>` | auto-detect for multilingual models | Set the language, for example `en` or `de`. English-only models use English and reject non-English hints. |
| `--device <NODE>` | PipeWire session's default source | Select a named PipeWire node. PolyScribe checks it against the running session when it can list sources. |
| `--window <WINDOW>` | 5 seconds | Live window duration in seconds. Values must be finite and at least 2 seconds. |
| `-o`, `--output <OUTPUT>` | no file | Also write transcript lines to this file, truncating an existing file. |
| `--gpu-backend <GPU_BACKEND>` | `auto` | Choose `auto`, `cpu`, `cuda`, or `hip`; GPU choices need a matching build and device. |

Examples:

```bash
# Detect language and use the default PipeWire source
polyscribe record

# Set language and device; save transcript lines as well as writing stdout
polyscribe record --language de --device voice_arctis_mono --output recording.txt
```

The output on stdout is one finalized segment per line. With global `--json`, each line is a JSON object with `speaker`, `start`, `end`, and `text`. Provisional text and diagnostics go to stderr. `--quiet` suppresses informational status and provisional text, not errors or warnings. A segment is printed once, after it is finalized. Ctrl+C asks the run to stop at a safe boundary; a Whisper call already running is not interrupted. A second Ctrl+C ends the process immediately.

Common errors include missing PipeWire/`pw-record`, a missing model, an unknown device, and a window shorter than two seconds. If source enumeration fails, PolyScribe warns and continues with the requested device because it cannot verify it.

## `models`

Manage Whisper models from Hugging Face. These commands use the network for model searches and downloads, except local operations such as listing the manifest, removing, verifying, garbage collection, and clearing the manifest cache.

All commands except `download` and `clear-cache` accept `--concurrency <N>` (default `4`) and `--limit-rate <BYTES_PER_SEC>` (default unlimited). They control transfer behavior where downloads occur; they do not make local operations use the network. `--json` selects structured output where available. Model commands return exit code `2` for not found, `3` for network failure, and `4` for failed verification; success is `0`.

| Command | Purpose and example |
|---|---|
| `models ls` | List the installed-model manifest and discover local `*.bin` / `*.gguf` files: `polyscribe models ls` |
| `models add <REPO> <FILE>` | Download or update a model and derive its alias from the repo/file: `polyscribe models add ggerganov/whisper.cpp ggml-tiny.en.bin` |
| `models rm <ALIAS_OR_FILE>` | Remove an installed model by alias or file name: `polyscribe models rm ggml-tiny.en.bin` |
| `models verify <ALIAS_OR_FILE>` | Check the installed file against its known SHA-256: `polyscribe models verify ggml-tiny.en.bin` |
| `models update` | Update Hugging Face models; discovered local files are skipped: `polyscribe models update` |
| `models download` | Open the interactive Hugging Face model selector: `polyscribe models download` |
| `models clear-cache` | Remove the cached Hugging Face manifest so the next request fetches it again: `polyscribe models clear-cache` |
| `models gc` | Remove unreferenced files and stale manifest entries: `polyscribe models gc` |
| `models search <REPO> [--query <SUBSTRING>]` | Search a Hugging Face repo for `.bin` and `.gguf` model files: `polyscribe models search ggerganov/whisper.cpp --query tiny` |

`models download` is interactive and is skipped when stdin is not a terminal or `--no-interaction` is set. In scripts, prefer `models add`. A missing repo/file, an unavailable Hugging Face service, or rate limiting can fail a network command. Use a valid repo and filename, check connectivity, and set `HF_TOKEN` only when authenticated access is required.

## `plugins`

Discover and run local plugin executables. PolyScribe searches `PATH` and the platform's PolyScribe data directory under `plugins/`. These commands do not contact remote services themselves, but a plugin runs with your user privileges and may perform its own network access.

| Syntax | Output and example |
|---|---|
| `plugins list` | Print discovered plugin names, one per line: `polyscribe plugins list` |
| `plugins info <NAME>` | Run the plugin's `info` operation and print its capabilities JSON: `polyscribe plugins info tubescribe` |
| `plugins run <NAME> <COMMAND> [--payload <JSON>]` | Send one request and forward the plugin's output stream: `polyscribe plugins run tubescribe generate_metadata --payload '{"url":"https://example.com"}'` |

If omitted, `--payload` defaults to `{}`. Valid JSON text is sent as a JSON value; non-JSON text is sent as a JSON string. Child stderr is inherited by the terminal. A missing plugin, invalid JSON from `info`, or nonzero `run` exit is an error. `plugins run` forwards stdout lines and does not validate their contents. The `--payload` option avoids a name collision with global `--json`.

## `completions` and `man`

Both commands write generated content to stdout and can be redirected to files:

```bash
polyscribe completions bash > polyscribe.bash
polyscribe man > polyscribe.1
```

Supported completion targets are `bash`, `zsh`, `fish`, `powershell`, and `elvish`. An unsupported target is rejected by argument parsing.

## Whisper model lookup

`WHISPER_MODEL=/path/to/model.bin` selects an exact existing file. A bare file name such as `WHISPER_MODEL=ggml-small.bin` is looked up in `POLYSCRIBE_MODELS_DIR`, then `$XDG_DATA_HOME/polyscribe/models` or `~/.local/share/polyscribe/models`. Otherwise PolyScribe searches the same directories and uses the largest `.bin` model it finds. Model downloads and local discovery use the same directory by default.

To download a model non-interactively:

```bash
polyscribe models add ggerganov/whisper.cpp ggml-tiny.en.bin
```

An English-only model can only be used with English. A multilingual model can use language detection where the command supports it, or an explicit language hint.

## Configuration and environment

| Variable | Purpose and default |
|---|---|
| `POLYSCRIBE_MODELS_DIR` | Override the directory searched for transcription models. |
| `WHISPER_MODEL` | Select one exact local Whisper model file. |
| `POLYSCRIBE_YTDLP` | Select the `yt-dlp` executable file instead of searching `PATH`. |
| `POLYSCRIBE_VOXTRAL` | Absolute path to the optional local Voxtral wrapper. Unset by default. |
| `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `HOME` | Resolve model, cache, config, and plugin paths. |
| `POLYSCRIBE_CACHE_DIR` | Override the model-download cache base; model files are stored below its `models/` child. |
| `POLYSCRIBE_CONFIG_DIR` | Override the model manifest config directory when `POLYSCRIBE_CACHE_DIR` is also set. Otherwise the XDG/HOME default is used. |
| `HF_TOKEN` | Optional token for authenticated Hugging Face requests. |
| `POLYSCRIBE_NO_CACHE_MANIFEST` | Any set value disables the Hugging Face manifest cache. |
| `POLYSCRIBE_MANIFEST_TTL_SECONDS` | Manifest cache lifetime; default `86400` seconds. |
| `POLYSCRIBE_HF_REPO` | Default Hugging Face repository for the interactive downloader; default `ggerganov/whisper.cpp`. |
| `POLYSCRIBE_USER_AGENT` | Override the HTTP user agent. |
| `POLYSCRIBE_HTTP_TIMEOUT_SECS` | HTTP timeout for model operations; default `8` seconds. |
| `POLYSCRIBE_MANIFEST_CACHE_FILENAME` | Manifest cache filename; default `hf_manifest_whisper_cpp.json`. |
| `POLYSCRIBE_WINDOW_TRACE` | Set to `1` to emit detailed transcription window diagnostics to stderr. |

The model-download cache and transcript cache are different locations. YouTube transcripts use `$XDG_CACHE_HOME/polyscribe/youtube/v3/`; there is no separate environment-variable override for that path.

## Network and local processing

- `transcribe` and `record` use local Whisper inference and do not contact remote services. `record` needs PipeWire and `pw-record`; `transcribe` needs `ffmpeg` for media decoding.
- `youtube` contacts YouTube through `yt-dlp` to fetch metadata and audio. Voxtral and Whisper processing then happen locally.
- `models add`, `update`, `search`, and the interactive downloader contact Hugging Face. Other model commands work on the local manifest and files.
- `plugins`, `completions`, and `man` have no built-in network activity. A plugin may make its own requests.

## stdout, stderr, and piping

Transcript/data output belongs on stdout; informational messages and errors belong on stderr. This permits shell redirection and piping without mixing most diagnostics into the data stream:

```bash
polyscribe transcribe --language en input.wav > transcript.json
polyscribe youtube 'https://youtu.be/VIDEO_ID' --format json > youtube-transcript.json
polyscribe record --json > live-transcript.jsonl
```

Use `2>errors.log` to capture stderr separately. YouTube always writes a short source summary to stderr after a successful transcript. `--quiet` and global `--json` reduce decorative logs but do not remove errors or that YouTube summary. For `record --json`, stdout is JSON Lines, with one finalized segment object per line. YouTube's JSON is one object with a `cues` array. `transcribe` stdout is a JSON transcript root with an `items` array. These schemas are not interchangeable.

## Current limitations

- YouTube captions are unsupported; `--captions-only` fails. `--prefer-manual` is deprecated and ignored; `--audio-only` is a no-op alias.
- YouTube HTTP 403/429 and other acquisition failures stop before transcription. No caption or network retry fallback is attempted.
- A configured Voxtral wrapper and Whisper fallback execute locally but are not sandboxed. Whisper inference runs in-process and cannot be forcibly stopped by the decoder timeout.
- File transcription requires `--language`; microphone recording and YouTube can use language detection.
- Automatic speaker diarization is not implemented.
- Linux x86_64 is the CI-tested platform. The workspace does not compile on Windows yet; macOS is not covered by CI. GPU runtime behavior is not verified by CI.
