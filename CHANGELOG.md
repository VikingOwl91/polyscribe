# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## Unreleased

### Added
- YouTube: `polyscribe youtube` now uses an audio-first, cache-first pipeline and never requests or downloads subtitles. It tries an optional local Voxtral executable first, then makes one local Whisper attempt with the same audio when Voxtral is unavailable or fails. `--captions-only` fails before network access, `--prefer-manual` is a no-op, and `--audio-only` aliases the default.
- YouTube: transcript cache schema and path move to v3 so caption-era cache entries cannot satisfy audio requests. Cache identity includes the configured Voxtral executable and its resolved symlink chain. JSON output retains the source URL, video ID, available video metadata, language, engine, and model. Audio downloads, subprocess output, temporary files, and process duration are bounded; download failures including HTTP 429 stop before transcription.
- Tests: offline fake-process coverage verifies no subtitle arguments, Voxtral success/fallback, cache hits across output formats, refresh, HTTP 429 classification, temporary-file cleanup, and captions-only behavior. An optional model-backed e2e test uses a local audio fixture and fake downloader; it is not a live YouTube test and requires a model and ffmpeg.
- Record: `polyscribe record` captures the microphone through PipeWire and transcribes it while the recording runs. The audio is captured with `pw-record` in raw mode, so the samples reach the pipeline while the recording is still going, and the recording is never written to a file. The capture runs in a supervised process group; its standard error is read by its own thread, and shutdown stops the group before joining the reader threads. The transcript goes to standard output, one finalized segment per line, and everything else goes to standard error.
- Record: the language is selected automatically by default and `--language` overrides that selection. Multilingual models detect it on a longer startup window, because a short window carries too little speech for a reliable detection. English-only models select English directly and reject non-English hints. A multilingual model only pins the detection when the window carries speech; a language detected in silence is a guess, and the guess is named on stderr. The file path still requires `--language` and is unchanged.
- Record: `--device` selects the PipeWire node to record from. The node is checked against the nodes of the running session before the capture starts, because `pw-record` records from the default source when it is given a node that does not exist, and does not say so.
- Record: `--window` sets the length of a live window and `--output` writes the transcript to a file while the recording runs, replacing any existing contents.
- Record: a second Ctrl+C ends the run at once. The first signal asks the run to stop at its next safe boundary, and a whisper call that is running is not interrupted, so a user that waits for nothing presses the key again. The second signal stops the capture and ends the process.
- Record: a live segment is dropped when its own audio interval holds less than two 20 ms frames above the speech-energy threshold. The phrases that Whisper makes up for a silent window stay out of the transcript, and a short utterance is kept even when it occupies a small share of a longer window.
- Core: a live window policy for the streaming path. The batch policy keeps its windows of 30 seconds and its behaviour; the live policy uses shorter windows, so that the text of a recording appears while it is still running, and drops the segments whose own audio interval holds no speech-level energy. The window handling, the reconciliation of the segments and the recovery of a region that whisper did not transcribe are the ones of the batch path.
- Core: the streaming path reports the segments that may still change as provisional, separately from the segments that are final. A provisional segment is reported only when its text changed, a final segment is never reported twice, and the cancellation of a run does not present an unconfirmed segment as final.
- Docs: initial live-window quality and HIP latency measurements, including the limits of the fixture set, in `docs/benchmarks/live-transcription.md`.
- Core: the live path drops a segment when Whisper reports a no-speech probability above 0.6. This extra filter is limited to live transcription; the file path keeps the segments Whisper reports, matching its pre-Sprint-12 behavior.
- Core: a streaming input for the transcription pipeline. A source delivers decoded 16 kHz mono samples in blocks while the transcription runs; the pipeline turns them into the same windows the file path produces, so the window handling, the reconciliation of the segments and the recovery of a region that whisper did not report are shared. A source error ends the run with that error, and a cancellation token stops the run before the next window is submitted. The finalized segments are the ones the file path reports; the segments that may still change are reported separately as provisional and are never presented as final. The command line is unchanged.

### Added
- Transcribe/YouTube: a stderr progress bar for terminal sessions with the current stage, elapsed time, and an ETA derived from completed inference windows. `--quiet` and `--no-progress` disable it, and stdout remains transcript-only.

### Fixed
- Audio decoding: ffmpeg now receives a null stdin. It runs in its own process group, so inheriting an interactive terminal let it stop with SIGTTIN/SIGTTOU and hang transcription at 0% until interrupted.
- Models: `models ls` now discovers local `*.bin` / `*.gguf` files and persists them in the same manifest used by the other model commands. `models rm` and `models verify` accept either an alias or a file name, and `models update` skips discovered local files instead of trying to update them from a Hugging Face repo named `local`.
- Transcribe: `WHISPER_MODEL` accepts a bare model file name and looks it up in the models directory, while an explicit path still takes precedence.
- Transcribe/YouTube: `-vv` now reports monotonic timings for ffmpeg decoding, Whisper context loading, state creation, and every `state.full()` call, plus the selected model path, audio duration, sample count, and inference-window count. A CPU-only binary also says that no GPU backend is compiled in and names the feature needed for GPU support.
- Transcribe: escape control characters in displayed input and output paths so a filename cannot emit terminal control sequences. Filesystem paths and transcript text are unchanged.
- Record: escape PipeWire source names in readiness messages and child-reported audio formats in mismatch errors.
- YouTube: stop subprocess readers promptly after cancellation, timeout, or output overflow. A 200 ms post-exit drain preserves the child status when only stderr stays open and rejects incomplete stdout.
- YouTube: report bounded, typed Voxtral process failures with actionable configuration and protocol details; arbitrary child stderr remains private.
- Tests: cover a short-timeout Voxtral-to-Whisper fallback path without network access or a model download.
- Transcribe: a segment that was cut short by the end of a window is no longer decoded again and again in a window that holds the same audio. The number of the attempts is bounded, so that the pipeline cannot spend an endless series of decodes on a segment that stays cut short. The output of the file path is unchanged.
- Core: a segment that covers a whole window is no longer re-decoded. It is complete, and a window of the same length cannot give it more context. This prevents a hallucinated phrase (or a silence) that spans a whole window from being decoded again and again, advancing by one sample per attempt.
- Core: the minimum window advance after a recovery is measured from the main window, not the recovery window. A recovery window starts earlier, so measuring from it could stall the main window at its original position.
- Core: a recoverable speech hole in the final partial window can be decoded again from the retained samples. If recovery cannot get a window for a speech-bearing hole, the run now warns that the transcript may be incomplete.

### Removed
- Core: remove the unused public WebVTT parser, caption-track selector, rolling-caption normalization, and filename sanitizer, along with their exclusive tests and VTT fixture. YouTube transcription no longer uses captions. This removes alpha-stage core APIs; live URL validation, cue rendering, and cache version fields remain.
- Core: remove unused empty `VideoMetadata.subtitles` and `automatic_captions` fields.
- Dependencies: remove unused direct dependencies from the CLI, plugin host, and reference plugin, plus the unused workspace `tracing` dependency.

## v0.1.0-alpha.2 - 2026-10-21

### Changed
- Transcribe: long recordings are decoded and transcribed in windows of 30 seconds instead of being loaded into memory as a whole. The peak memory of a transcription no longer grows with the length of the recording: an hour long recording needs about 235 MB instead of 670 MB (CPU) and 780 MB (HIP). The output contracts do not change; a recording shorter than a window is transcribed exactly as before, a longer one keeps the same text with segment boundaries that can move slightly at a window boundary.
- Transcribe: a window now advances by at least half a window. whisper reports short segments for silence, so a window that started where the last segment ended decoded the same stretch of silence again and again: a minute of silence took 16 windows and produced a hallucinated segment per window, it now takes 2 windows. The speech at a window boundary is also recognized better, because the window no longer starts in the middle of a sentence.
- Backends: an explicit `--gpu-backend cuda` or `--gpu-backend hip` now requires a matching GPU device to be present. The device list comes from ggml's backend registry, the same registry whisper.cpp selects its GPU from. Without a device the request fails with an actionable error instead of silently continuing on the CPU; `--gpu-backend auto` falls back to the CPU and reports it.
- Transcribe: the Whisper model is loaded once per invocation and reused for every input file, instead of being loaded again for each file. Every file still gets its own inference state, so outputs, timestamps and error handling stay independent.

### Fixed
- Transcribe: a region that whisper did not transcribe is no longer lost. whisper does not always report a segment for the beginning of the audio it is given: for some positions it reports its first segment many seconds into the window, and for others it reports nothing at all. The pipeline advanced to the last segment of the window, so the speech in front of it was never transcribed and never offered again. Such a region is decoded again from a slightly different position, which gives the speech back: on the recording that reproduces it, the word error rate goes from 0.679 to 0.277, and the transcript is better than the one of the implementation that hands the whole recording to whisper. A region of silence is not decoded again, and the attempts are bounded.
- Transcribe: a GPU-enabled build ran every transcription on the GPU, also the ones that asked for the CPU, while the log still reported the selected backend. The selected backend now reaches the model parameters, so a CPU request keeps the model in RAM and a GPU request offloads it.
- Transcribe: a segment that was cut short by the end of a window is no longer dropped before it is decoded again. The re-decode can start a little before the dropped segment and was then mistaken for a duplicate, so a whole sentence could be missing from the transcript. The cut segment is kept until the re-decode replaces it.
- Transcribe: segments without a duration are no longer written to the transcript. whisper reports such a segment at the end of a window, with the text of the speech around it; it showed up as a subtitle without a duration, on top of the segment next to it.

### Added
- Transcribe: a warning on stderr when a region that carries speech was not transcribed and could not be decoded again. It names the file and the time range and says that the transcript may be incomplete. The exit status does not change.
- Core: `chunked` module that decodes a media file incrementally and hands it to the transcriber in windows, so the memory of a transcription is independent of the length of the recording
- Docs: `docs/benchmarks/chunked-transcription.md` describes the pipeline, the memory bounds, the window and reconciliation strategy, the boundary quality measurement, the recovery of a region that whisper did not transcribe and the measured before/after results
- Core: `gpu` module with a device probe for the GPU backends, and unit tests for the backend selection contract (explicit CPU, explicit GPU with and without a device, and the automatic fallback).
- Tests: CLI integration tests for the backend selection contract.
- Tests: CLI integration tests for the window boundaries, with speech fixtures that cross a boundary. They check that every stretch of speech is covered by a segment, that the timestamps are ordered and do not overlap, and that a minute of silence does not need more than four windows.
- Tests: CLI integration tests for the recovery of a region that whisper did not transcribe, with the recording that reproduces it and a recording with varied speech.
- Tests: speech fixtures for the boundary and the recovery tests, generated with espeak-ng and resampled to 16 kHz mono, with the reference transcript of every clip.
- Scripts: `scripts/benchmarks/boundary_quality.py` composes the boundary fixtures, transcribes them with two binaries and reports the word error rate, the missing and the duplicated words against the reference transcript.
- Scripts: `scripts/benchmarks/recovery_quality.py` does the same for the boundary and the recovery fixtures and reports the number of windows, the recovery attempts and the regions with speech that were not recovered.
- Docs: the backend selection contract and the model context lifecycle in `docs/usage.md` and `docs/design.md`, plus the before/after measurements of the model context reuse in `docs/benchmarks/amd-rx-9070-xt.md`.

## v0.1.0-alpha.1 - 2026-10-21

### Added
- Tests (audit F-05): end-to-end transcription smoke test (`crates/polyscribe-cli/tests/e2e_transcribe.rs`) that runs the built CLI on the committed `speech.wav` fixture with a real Whisper model on CPU, asserting successful execution, valid JSON/TOML/SRT output, and sane timestamps. It skips itself with a note unless `POLYSCRIBE_E2E_MODEL` points at a model file, so the workspace suite still needs no model.
- CI: new `e2e-transcribe` job that downloads a SHA-256-pinned `ggml-tiny.en.bin` (cached; the checksum is verified on every run so unverified cache contents are never trusted) and executes the e2e smoke test against the compiled CLI.
- CI: the `test` and `e2e-transcribe` jobs install ffmpeg when it is missing (current ubuntu-latest runners no longer ship it), so ffmpeg-dependent tests run instead of skipping; the first-run model-discovery tests skip themselves with a note when ffmpeg is absent locally.

### Changed
- Docs: Replace `--download-models`/`--update-models` flags with `models download`/`models update` subcommands in `README.md`, `docs/usage.md`, and `docs/development.md`.
- Host: Plugin discovery now scans `$XDG_DATA_HOME/polyscribe/plugins` (platform equivalent via `directories`) in addition to `PATH`.
- CI: Add GitHub Actions workflow to run fmt, clippy (warnings as errors), and tests for pushes and PRs.
- Models (audit F-02): the transcription models directory is resolved by one canonical policy in every build mode: `POLYSCRIBE_MODELS_DIR`, then `$XDG_DATA_HOME/polyscribe/models`, then `~/.local/share/polyscribe/models`. The debug-only `./models` default is gone, so `cargo run -- models download` and `cargo run -- transcribe` agree on the directory. `ConfigService::models_dir` delegates to the shared resolver; the divergent `directories`-crate-based `default_models_dir` was removed. Docs (README, usage, development, release-packaging, faq) updated to match.

### Integrated (merge of the modernization line into the workspace)
- Dependency: `whisper-rs` upgraded from 0.14.3 (git) to the versioned crates.io release 0.16; segment access migrated to the 0.16 API. MSRV declared as 1.88 (`rust-version`), verified with `cargo +1.88 check --workspace`.
- Backends: removed the non-functional Vulkan stub (`gpu-vulkan` feature, `VulkanBackend`, `vulkan` CLI value); GPU availability is now a build-time property (`--features gpu-cuda` / `--features gpu-hip` on `polyscribe-cli`), with actionable errors when a backend was not compiled in.
- HTTPS: `reqwest` uses `rustls-tls` (pure-Rust TLS, bundled Mozilla roots) with charset/http2/system-proxy enabled; no system OpenSSL requirement.
- Transcribe: the `transcribe` subcommand now runs the full pipeline (backend selection, ffmpeg decode, whisper inference, JSON/TOML/SRT output) instead of a stub; output writing is deduplicated in `polyscribe-core`'s `output` module.
- Audio: ffmpeg decoding streams PCM instead of using a fixed-name temp file (no cross-run collisions, half the peak memory).
- Logging: quiet mode now consistently suppresses info/success/progress output (ProgressReporter, Spinner, BytesProgress, ui::info/success/println_above_bars); errors and warnings remain visible.
- Security: model file names from remote listings/manifests are validated against path traversal (`ModelManager::add_or_update`, manifest file paths).
- Plugins: fixed the PSP/1 interface — the protocol crate gained `Capabilities`/`StreamItem`/`JsonRpcRequest`, the reference plugin now implements `info`/`run` subcommands, the host resolves plugins from the XDG plugin dir as well as `PATH`, and `plugins run --payload` no longer clashes with the global `--json` flag (renamed from `--json`).
- Tests: deterministic fixtures live in `crates/polyscribe-cli/tests/fixtures/`; the transcribe integration suite (20 tests), models smoke tests, and plugin end-to-end tests run without network, models, or GPU; env-mutating tests are serialized with a shared lock.
- CI: workflow runs fmt, locked workspace build, locked workspace tests, clippy with warnings as errors, plus an MSRV (1.88) job; triggers on all pushes and PRs.
- Docs: README, CONTRIBUTING, and docs/ rewritten to match the workspace CLI (subcommands, models/plugins, caching, GPU behavior, limitations).

### Fixed
- `plugins run --json` collided with the global `--json` output flag, making the subcommand unparseable; the payload flag is now `--payload`.
- `transcribe` previously validated inputs and reported success without transcribing anything.
- First run (audit F-01): with no model installed and no model cache dir, `transcribe` now shows the actionable "No Whisper model files found ... download a model (polyscribe models add) or set WHISPER_MODEL" message instead of `Failed to read models dir: ...`. A missing cache dir counts as "no cached models"; genuine I/O errors on existing directories still propagate.
- Tests: model-discovery unit tests (empty/missing dirs, cache-dir fallback, `WHISPER_MODEL` overrides, unreadable cache dir) and CLI first-run integration tests running the binary with an isolated environment.
