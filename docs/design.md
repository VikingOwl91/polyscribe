# Design

Overview
PolyScribe is a CLI that orchestrates:
- CLI parsing and I/O (main.rs)
- Core library (lib.rs) exposing reusable logic
- Backends for transcription (backend.rs) that bind to whisper-rs
- Model management (models.rs) that discovers/downloads/verifies models

Data flow
1) CLI collects inputs (media or JSON), options (merge, speaker names, language, GPU backend), and output path.
2) For media, audio is extracted via ffmpeg to PCM f32 in-memory.
3) A Whisper model is selected (env var override, last-used, interactive download, or non-interactive default).
4) The selected backend performs transcription via whisper-rs producing segments.
5) Segments are merged/organized and written to JSON and SRT as requested.

Key decisions
- Local-first: default to local models in ./models (debug) or XDG data dir (release) for predictable behavior.
- Whisper model selection: last-used cache (.last_model) provides stable default across runs.
- Non-interactive mode: avoid prompts for CI; download a sensible default if needed.
- Logging: simple macros (elog!/wlog!/ilog!/dlog!) with quiet/verbose controls; stderr used for diagnostics.
- GPU selection: runtime auto-detect with compile-time feature gates per backend.

Model discovery & verification (conceptual)
- Remote model list pulled from Hugging Face repositories.
- For each model entry we track name, size, and optionally SHA-256.
- Downloads verify size and hash when available; updates compare local files against the manifest.
- Best local model is chosen based on reasonable heuristics (e.g., prefer larger quantized variants when available) to balance quality and speed.

Extensibility
- New backends: implement TranscribeBackend and add selection wiring in select_backend.
- New model sources: extend models.rs to read additional manifests or repositories.
- Packaging: respect XDG_DATA_HOME/HOME; allow POLYSCRIBE_MODELS_DIR override; avoid hard-coding system paths.

Binary naming and CLI surface
- Binary is `polyscribe`.
- Keep CLI flags stable and documented; add new flags conservatively.
