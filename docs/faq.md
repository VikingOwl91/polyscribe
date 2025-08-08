# FAQ

Models are missing — what do I do?
- Run `polyscribe --download-models` to pick and download models interactively.
- For CI/non-interactive, set POLYSCRIBE_MODELS_DIR to a writable dir and run `polyscribe --update-models --no-interaction`.
- You can also point to a specific file via `WHISPER_MODEL=/path/to/model.bin`.

I get timeouts or slow downloads
- Try again later or use a closer mirror (if available by setting upstream env vars or downloading manually into the models dir).
- Ensure your network allows Hugging Face downloads.

Non-interactive CI runs hang or fail
- Add `--no-interaction` to disable prompts.
- Set `POLYSCRIBE_MODELS_DIR` to a known location and pre-populate models or run `--update-models`.
- Use `-q` to reduce noise in logs; use `-v` or `-vv` when debugging failures.

GPU was not detected
- Ensure you built with the matching feature (`gpu-cuda`, `gpu-hip`, or `gpu-vulkan`).
- Install the relevant runtime (CUDA toolkit/driver, ROCm libraries, Vulkan loader/SDK) and ensure libraries are on the loader path.
- Force CPU backend with `--gpu-backend cpu` to verify the rest of the pipeline.

Which model directory is used in releases?
- For packaged binaries, PolyScribe uses `$XDG_DATA_HOME/polyscribe/models` or `~/.local/share/polyscribe/models` by default. Override with `POLYSCRIBE_MODELS_DIR`.

SRT timestamps look wrong
- SRT times are derived from model timestamps. If your input has variable sample rate or corrupted timestamps, ensure ffmpeg can decode it; consider re-encoding the audio.
