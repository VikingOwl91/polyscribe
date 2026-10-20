# FAQ

Models are missing — what do I do?
- Run `polyscribe --download-models` to pick and download models interactively.
- For CI/non-interactive, set `POLYSCRIBE_MODELS_DIR` to a writable dir and run `polyscribe --update-models --no-interaction`.
- You can also point to a specific file via `WHISPER_MODEL=/path/to/model.bin`.

I get timeouts or slow downloads
- Try again later, or download manually into the models dir (`ggml-<name>.bin` from `https://huggingface.co/ggerganov/whisper.cpp`).
- Ensure your network allows Hugging Face downloads (HTTPS).

Non-interactive CI runs hang or fail
- Add `--no-interaction` to disable prompts.
- Set `POLYSCRIBE_MODELS_DIR` to a known location and pre-populate models or run `--update-models`.
- Use `-q` to reduce noise in logs; use `-v` or `-vv` when debugging failures.

GPU was not detected / which backends exist?
- GPU support is a build-time property. Build with `--features gpu-cuda` (NVIDIA) or `--features gpu-hip` (AMD ROCm) and install the matching runtime.
- `--gpu-backend auto` picks CUDA → HIP → CPU among the backends compiled into the binary; `--gpu-backend cpu` forces CPU.
- There is no Vulkan backend; `--gpu-backend vulkan` is rejected by the CLI.

Is there automatic speaker diarization?
- No. Speaker labels are assigned manually per input file with `--set-speaker-names` (or default to the sanitized file stem). Diarization is on the backlog (TODO.md, F1).

Which model directory is used in releases?
- For packaged binaries, PolyScribe uses `$XDG_DATA_HOME/polyscribe/models` or `~/.local/share/polyscribe/models` by default. Override with `POLYSCRIBE_MODELS_DIR`.

SRT timestamps look wrong
- SRT times are derived from model timestamps. If your input has variable sample rate or corrupted timestamps, ensure ffmpeg can decode it; consider re-encoding the audio.

Why does building require cmake/libclang?
- The `whisper-rs-sys` crate compiles whisper.cpp from source and generates Rust bindings with bindgen, which needs libclang. Install `cmake`, a C/C++ toolchain, and `libclang` (e.g. `libclang-dev`).
