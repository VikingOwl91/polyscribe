# Design

Overview
PolyScribe is a CLI that orchestrates:
- CLI parsing and prompting (main.rs)
- Core library (lib.rs) exposing reusable logic: logging, runtime flags, model discovery, audio decoding
- Output module (output.rs): transcript types, sorting, and JSON/TOML/SRT rendering
- Backends for transcription (backend.rs) that bind to whisper-rs
- Model management (models.rs) that discovers/downloads/verifies models

Data flow
1) CLI collects inputs (media or JSON), options (merge, speaker names, language, backend), and output path.
2) For media, audio is streamed from ffmpeg and decoded to 16 kHz mono PCM f32 in memory.
3) A Whisper model is selected (env var override, last-used, interactive download, or non-interactive default).
4) The selected backend transcribes via whisper-rs, producing segments.
5) Segments are sorted, reindexed, and written to JSON + TOML + SRT (or JSON to stdout).

Key decisions
- Local-first: default to local models in `./models` (debug) or the XDG data dir (release) for predictable behavior. Network access happens only for `--download-models` / `--update-models` (Hugging Face over HTTPS, rustls).
- Whisper model selection: last-used cache (`.last_model`) provides a stable default across runs.
- Non-interactive mode: avoid prompts for CI; download a sensible default if needed.
- Logging: simple macros (`elog!`/`wlog!`/`ilog!`/`dlog!`) with quiet/verbose controls; stderr is used for diagnostics, stdout for results.
- GPU selection: availability is a build-time property. A binary built with `gpu-cuda`/`gpu-hip` links the corresponding runtime and can use it; without the feature the backend is unavailable and requesting it produces an actionable error. Runtime dlopen probing was tried and removed (it was unstable and never detected anything). A Vulkan backend is not wired up.
- Speaker labels are manual (per-input prompts). There is no automatic speaker diarization.

Model discovery & verification (conceptual)
- Remote model list pulled from Hugging Face repositories (tree API, with per-file size/SHA-256 metadata; HEAD requests fill gaps).
- Downloads go to a `.part` file and are verified by SHA-256 when the hash is known; the file is renamed into place only after verification.
- Updates compare local files against the manifest (hash first, size as fallback).
- Best local model is chosen by largest file size (tie-break: lexicographic filename) to balance quality and speed.

Extensibility
- New transcription engines: implement `TranscribeBackend` and register them in `select_backend` (backend.rs).
- New output formats: extend `output.rs` (`render_srt` is the pattern for text formats).
- New model sources: extend `models.rs` to read additional manifests or repositories.
- Packaging: respect `XDG_DATA_HOME`/`HOME`; allow `POLYSCRIBE_MODELS_DIR` override; avoid hard-coding system paths.

Binary naming and CLI surface
- Binary is `polyscribe`.
- Keep CLI flags stable and documented; add new flags conservatively.
