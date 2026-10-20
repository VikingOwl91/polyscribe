# FAQ

Models are missing — what do I do?
- For transcription: put a `ggml-*.bin` model in the models dir (debug: `./models`; release: `$XDG_DATA_HOME/polyscribe/models`), set `WHISPER_MODEL=/path/to/model.bin`, or override the dir with `POLYSCRIBE_MODELS_DIR`.
- For the `models` subcommands: `polyscribe models add ggerganov/whisper.cpp ggml-tiny.en.bin`, or `polyscribe models download` for the interactive downloader.
- For CI: `polyscribe models add ggerganov/whisper.cpp ggml-tiny.en.bin` (non-interactive, verified download).

I get timeouts or slow downloads
- Try again later, or download manually into the models/cache dir from `https://huggingface.co/ggerganov/whisper.cpp`.
- Ensure your network allows Hugging Face downloads (HTTPS). Set `HF_TOKEN` if you hit rate limits on private/gated repos.
- Use `models add --limit-rate 1000000` to throttle, or `--concurrency 1` to serialize ranged downloads.

Non-interactive CI runs hang or fail
- Add `--no-interaction` (top-level flag) to disable prompts.
- Use `models add` / `models update` (non-interactive by design) instead of `models download`.
- Use `--quiet` to reduce noise in logs; use `-v` or `-vv` when debugging failures.

GPU was not detected / which backends exist?
- GPU support is a build-time property. Build with `--features gpu-cuda` (NVIDIA) or `--features gpu-hip` (AMD ROCm) and install the matching runtime.
- `transcribe --gpu-backend auto` picks CUDA → HIP → CPU among the backends compiled into the binary; `--gpu-backend cpu` forces CPU.
- There is no Vulkan backend; `--gpu-backend vulkan` is rejected by the CLI.

Is there automatic speaker diarization?
- No. Speaker labels are assigned manually per input file with `--set-speaker-names` (or default to the sanitized file stem). Diarization is on the backlog (TODO.md).

Which directories are used?
- Transcription models: `./models` (debug) or `$XDG_DATA_HOME/polyscribe/models` (release); override `POLYSCRIBE_MODELS_DIR`.
- `models` subcommands: files in `$XDG_CACHE_HOME/polyscribe/models`, manifest at `$XDG_CONFIG_HOME/polyscribe/models.json`; override `POLYSCRIBE_CACHE_DIR` / `POLYSCRIBE_CONFIG_DIR`.
- Manifest cache: `$XDG_CACHE_HOME/polyscribe/manifest/` (24 h TTL).

SRT timestamps look wrong
- SRT times are derived from model timestamps. If your input has variable sample rate or corrupted timestamps, ensure ffmpeg can decode it; consider re-encoding the audio.

Why does building require cmake/libclang?
- The `whisper-rs-sys` crate compiles whisper.cpp from source and generates Rust bindings with bindgen, which needs libclang. Install `cmake`, a C/C++ toolchain, and `libclang` (e.g. `libclang-dev`).

How do I write a plugin?
- See `docs/development.md` ("Plugin development") and the reference plugin `plugins/polyscribe-plugin-tubescribe`. A plugin is a `polyscribe-plugin-<name>` executable implementing `info` (capabilities JSON) and `run` (one JSON-RPC request on stdin, NDJSON stream on stdout).
