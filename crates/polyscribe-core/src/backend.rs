// SPDX-License-Identifier: MIT

use crate::OutputEntry;
use crate::chunked::AudioWindow;
use crate::find_model_file;
use crate::gpu::{GpuBackend, GpuDevice, gpu_devices};
use crate::output::{is_audio_file, is_json_file, load_json_transcript};
use crate::prelude::*;
use anyhow::{Context, anyhow};
use std::env;
use std::path::{Path, PathBuf};

/// Test-only override for the GPU device probe.
///
/// "1" pretends that a device is present, "0" that none is. The backend
/// availability overrides only control what is compiled in, so the device
/// probe needs its own switch to cover the "compiled in but no device" case.
const ENV_TEST_GPU_DEVICES: &str = "POLYSCRIBE_TEST_FORCE_GPU_DEVICES";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Automatically select the best backend compiled into this binary (CUDA > HIP > CPU).
    Auto,
    /// Pure CPU backend using whisper-rs.
    Cpu,
    /// NVIDIA CUDA backend (requires a build with the `gpu-cuda` feature and the CUDA runtime).
    Cuda,
    /// AMD ROCm/HIP backend (requires a build with the `gpu-hip` feature and the ROCm runtime).
    Hip,
}

pub trait TranscribeBackend {
    fn kind(&self) -> BackendKind;
    fn transcribe(
        &mut self,
        audio_path: &Path,
        speaker: &str,
        language: Option<&str>,
        gpu_layers: Option<u32>,
        progress: Option<&(dyn Fn(i32) + Send + Sync)>,
    ) -> Result<Vec<OutputEntry>>;
}

/// Whether the CUDA backend is usable in this build.
///
/// GPU support is a build-time property: a binary built with the `gpu-cuda`
/// feature links the CUDA runtime, so the backend is available. Runtime
/// library probing was removed on purpose (dlopen probing proved unstable);
/// availability therefore follows the compiled-in features. The
/// `POLYSCRIBE_TEST_FORCE_CUDA` override exists for tests only (set it to
/// "0" to force the backend off).
fn cuda_available() -> bool {
    if let Ok(x) = env::var("POLYSCRIBE_TEST_FORCE_CUDA") {
        return x == "1";
    }
    #[cfg(not(test))]
    return cfg!(feature = "gpu-cuda");
    #[cfg(test)]
    return false;
}

/// Whether the ROCm/HIP backend is usable in this build (see [`cuda_available`]).
fn hip_available() -> bool {
    if let Ok(x) = env::var("POLYSCRIBE_TEST_FORCE_HIP") {
        return x == "1";
    }
    #[cfg(not(test))]
    return cfg!(feature = "gpu-hip");
    #[cfg(test)]
    return false;
}

/// A Whisper model that is loaded once and reused for every input of one
/// invocation.
///
/// Loading the model is the expensive part of a transcription run. On the GPU
/// path the weights are uploaded into VRAM, so loading them again for every
/// input file multiplied that cost by the number of files.
#[derive(Default)]
struct ModelSession {
    model_path: Option<PathBuf>,
    english_only: bool,
    context: Option<whisper_rs::WhisperContext>,
}

impl ModelSession {
    /// Locate the model, without loading it yet.
    ///
    /// This keeps the order of the original implementation: the input is
    /// decoded first, then the model is located, then the language hint is
    /// checked, and only then the model is loaded.
    fn resolve_model_path(&mut self) -> Result<()> {
        if self.model_path.is_some() {
            return Ok(());
        }
        let model_path = find_model_file()?;
        self.english_only = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.contains(".en.") || s.ends_with(".en.bin"))
            .unwrap_or(false);
        self.model_path = Some(model_path);
        Ok(())
    }

    /// Load the model, unless it is loaded already.
    fn ensure_context(&mut self, backend: BackendKind) -> Result<()> {
        if self.context.is_some() {
            return Ok(());
        }
        let model_path = self
            .model_path
            .clone()
            .expect("the model path was resolved before the context is loaded");
        let context = crate::with_suppressed_stderr(|| {
            let mut params = whisper_rs::WhisperContextParameters::default();
            // Only a GPU backend may use the GPU. whisper-rs defaults
            // `use_gpu` to the compiled-in GPU feature, so in a GPU-enabled
            // build a "cpu" request would still run on the GPU.
            params.use_gpu(matches!(backend, BackendKind::Cuda | BackendKind::Hip));
            whisper_rs::WhisperContext::new_with_params(&model_path, params).with_context(|| {
                format!("Failed to load Whisper model at {}", model_path.display())
            })
        })?;
        self.context = Some(context);
        Ok(())
    }
}

/// A backend that keeps the loaded model for the whole invocation.
#[derive(Default)]
pub struct CpuBackend {
    session: ModelSession,
}

/// A backend that keeps the loaded model for the whole invocation.
#[derive(Default)]
pub struct CudaBackend {
    session: ModelSession,
}

/// A backend that keeps the loaded model for the whole invocation.
#[derive(Default)]
pub struct HipBackend {
    session: ModelSession,
}

macro_rules! impl_whisper_backend {
    ($ty:ty, $kind:expr) => {
        impl TranscribeBackend for $ty {
            fn kind(&self) -> BackendKind {
                $kind
            }
            fn transcribe(
                &mut self,
                audio_path: &Path,
                speaker: &str,
                language: Option<&str>,
                _gpu_layers: Option<u32>,
                progress: Option<&(dyn Fn(i32) + Send + Sync)>,
            ) -> Result<Vec<OutputEntry>> {
                transcribe_with_whisper_rs(
                    &mut self.session,
                    $kind,
                    audio_path,
                    speaker,
                    language,
                    progress,
                )
            }
        }
    };
}

impl_whisper_backend!(CpuBackend, BackendKind::Cpu);
impl_whisper_backend!(CudaBackend, BackendKind::Cuda);
impl_whisper_backend!(HipBackend, BackendKind::Hip);

pub struct BackendSelection {
    pub backend: Box<dyn TranscribeBackend + Send + Sync>,
    pub chosen: BackendKind,
    /// The backends that are usable right now: compiled into this binary and
    /// backed by a device that ggml reports.
    pub detected: Vec<BackendKind>,
}

/// The GPU devices that are available for a backend, as reported by ggml's
/// backend registry.
///
/// The probe initializes the backend, so its diagnostics are kept out of the
/// normal output the same way the native libraries are.
fn gpu_devices_for(kind: BackendKind) -> Vec<GpuDevice> {
    if let Ok(x) = env::var(ENV_TEST_GPU_DEVICES) {
        // Test override: pretend that a device is (not) present.
        return if x == "1" {
            vec![GpuDevice {
                name: "test0".to_string(),
                description: "test device".to_string(),
            }]
        } else {
            Vec::new()
        };
    }
    let backend = match kind {
        BackendKind::Cuda => GpuBackend::Cuda,
        BackendKind::Hip => GpuBackend::Hip,
        _ => return Vec::new(),
    };
    crate::with_suppressed_stderr(|| gpu_devices(backend))
}

/// Select a backend according to the documented contract.
///
/// * `cpu` always runs on the CPU.
/// * `cuda` and `hip` require the backend to be compiled in **and** a
///   matching device to be present, otherwise the call fails with an
///   actionable error. A missing device is never silently replaced by the
///   CPU.
/// * `auto` picks CUDA, then HIP, and falls back to the CPU when no GPU
///   device is available.
///
/// The device list comes from ggml's backend registry, which is the same
/// registry whisper.cpp selects its GPU from. A device being present means
/// the runtime found it; it does not claim that every kernel of a
/// transcription ran on the GPU.
pub fn select_backend(requested: BackendKind, verbose: bool) -> Result<BackendSelection> {
    // The GPU backends that are compiled into this binary.
    let mut compiled = Vec::new();
    if cuda_available() {
        compiled.push(BackendKind::Cuda);
    }
    if hip_available() {
        compiled.push(BackendKind::Hip);
    }

    // The backends of those that have a device right now.
    let mut detected = Vec::new();
    let mut device_names: Vec<String> = Vec::new();
    for kind in &compiled {
        let devices = gpu_devices_for(*kind);
        if !devices.is_empty() {
            detected.push(*kind);
            device_names.extend(devices.iter().map(|d| d.name.clone()));
        }
    }

    let instantiate_backend = |k: BackendKind| -> Box<dyn TranscribeBackend + Send + Sync> {
        match k {
            BackendKind::Cpu => Box::new(CpuBackend::default()),
            BackendKind::Cuda => Box::new(CudaBackend::default()),
            BackendKind::Hip => Box::new(HipBackend::default()),
            BackendKind::Auto => Box::new(CpuBackend::default()), // resolved before construction
        }
    };

    let chosen = match requested {
        BackendKind::Auto => {
            if detected.contains(&BackendKind::Cuda) {
                BackendKind::Cuda
            } else if detected.contains(&BackendKind::Hip) {
                BackendKind::Hip
            } else {
                if !compiled.is_empty() {
                    crate::ilog!("No GPU device available, using the CPU backend.");
                }
                BackendKind::Cpu
            }
        }
        BackendKind::Cuda => {
            if !compiled.contains(&BackendKind::Cuda) {
                return Err(anyhow!(
                    "Requested CUDA backend but this binary was built without CUDA support. How to fix: install the NVIDIA driver + CUDA runtime and rebuild with --features gpu-cuda."
                ).into());
            }
            if !detected.contains(&BackendKind::Cuda) {
                return Err(anyhow!(
                    "Requested the CUDA backend but no CUDA device is available. How to fix: check that the NVIDIA driver and the CUDA runtime are working (nvidia-smi) and that the GPU is visible to this user."
                ).into());
            }
            BackendKind::Cuda
        }
        BackendKind::Hip => {
            if !compiled.contains(&BackendKind::Hip) {
                return Err(anyhow!(
                    "Requested ROCm/HIP backend but this binary was built without HIP support. How to fix: install the ROCm runtime and rebuild with --features gpu-hip."
                ).into());
            }
            if !detected.contains(&BackendKind::Hip) {
                return Err(anyhow!(
                    "Requested the ROCm/HIP backend but no compatible GPU device is available. How to fix: check that the ROCm runtime works (rocminfo), that the GPU is visible, and that this user can access /dev/kfd."
                ).into());
            }
            BackendKind::Hip
        }
        BackendKind::Cpu => BackendKind::Cpu,
    };

    if verbose {
        crate::dlog!(1, "Compiled-in backends: {:?}", compiled);
        crate::dlog!(1, "Detected backends: {:?}", detected);
        if !device_names.is_empty() {
            crate::dlog!(1, "GPU devices: {}", device_names.join(", "));
        }
        crate::dlog!(1, "Selected backend: {:?}", chosen);
    }

    Ok(BackendSelection {
        backend: instantiate_backend(chosen),
        chosen,
        detected,
    })
}

/// Collect transcript entries for a single input file.
///
/// Audio/video inputs are transcribed through `backend`; JSON transcripts are
/// loaded from disk. The `speaker` label is attached to every produced entry.
/// Progress is reported through the backend's progress callback and logged to
/// stderr (suppressed by quiet mode); a transcription failure is logged when
/// running interactively and then propagated.
pub fn collect_entries_for_input(
    backend: &mut dyn TranscribeBackend,
    path: &Path,
    speaker: &str,
    lang_opt: Option<&str>,
    gpu_layers: Option<u32>,
    progress: Option<&(dyn Fn(i32) + Send + Sync)>,
) -> Result<Vec<OutputEntry>> {
    if is_audio_file(path) {
        crate::ilog!("Processing file: {} ...", path.display());
        match backend.transcribe(path, speaker, lang_opt, gpu_layers, progress) {
            Ok(items) => {
                crate::ilog!("done");
                Ok(items)
            }
            Err(e) => {
                if !crate::is_no_interaction() && crate::stdin_is_tty() {
                    crate::elog!("{:#}", e);
                }
                Err(e)
            }
        }
    } else if is_json_file(path) {
        let mut entries = load_json_transcript(path)?;
        for e in &mut entries {
            e.speaker = speaker.to_string();
        }
        Ok(entries)
    } else {
        Err(anyhow!(
            "Unsupported input type (expected .json or audio media): {}",
            path.display()
        )
        .into())
    }
}

#[allow(clippy::too_many_arguments)]
fn transcribe_with_whisper_rs(
    session: &mut ModelSession,
    backend: BackendKind,
    audio_path: &Path,
    speaker: &str,
    language: Option<&str>,
    progress: Option<&(dyn Fn(i32) + Send + Sync)>,
) -> Result<Vec<OutputEntry>> {
    let report = |p: i32| {
        if let Some(cb) = progress {
            cb(p);
        }
    };
    report(0);

    // The audio is decoded in windows, so only one window of samples is in
    // memory at a time. The first window is decoded before the model is
    // touched, which keeps the order of the errors: a broken input file is
    // reported before a missing model.
    let mut windows =
        crate::chunked::ChunkedAudio::new(audio_path, crate::chunked::ChunkConfig::default())?;
    let first_window = match windows.next_window()? {
        Some(window) => window,
        None => {
            return Err(anyhow!(
                "The input contains no audio samples: {}",
                audio_path.display()
            )
            .into());
        }
    };
    report(5);

    // The model is located and loaded once per invocation; every input file
    // gets its own inference state.
    session.resolve_model_path()?;
    let model_path = session
        .model_path
        .clone()
        .expect("the model path was just resolved");
    if let Some(lang) = language
        && session.english_only
        && lang != "en"
    {
        return Err(anyhow!(
            "Selected model is English-only ({}), but a non-English language hint '{}' was provided. Please use a multilingual model or set WHISPER_MODEL.",
            model_path.display(),
            lang
        ).into());
    }

    // Try to reduce native library logging via environment variables when not super-verbose.
    if crate::verbose_level() < 2 {
        // These env vars are recognized by ggml/whisper in many builds; harmless if unknown.
        unsafe {
            std::env::set_var("GGML_LOG_LEVEL", "0");
            std::env::set_var("WHISPER_PRINT_PROGRESS", "0");
        }
    }

    // The model is loaded once per invocation; every input file gets its own
    // inference state. The state is created with the native libraries quiet,
    // the same way the model load is.
    session.ensure_context(backend)?;
    let mut state = crate::with_suppressed_stderr(|| {
        session
            .context
            .as_ref()
            .expect("the context was just loaded")
            .create_state()
            .map_err(|e| anyhow!("Failed to create Whisper state: {:?}", e))
    })?;
    report(20);

    let mut full_params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1);
    full_params.set_n_threads(threads);
    full_params.set_translate(false);
    if let Some(lang) = language {
        full_params.set_language(Some(lang));
    }
    report(30);

    // Every window is transcribed on its own. whisper reports the
    // timestamps of a window relative to the start of that window, so the
    // window's position in the recording is added to them.
    let mut entries: Vec<OutputEntry> = Vec::new();
    let mut window = Some(first_window);
    let mut emitted_end = 0.0f64;
    // The end of the transcribed timeline, in samples.
    let mut frontier = 0u64;
    let mut window_count = 0u64;
    // The index of a segment that may have been cut short by the end of the
    // window. The next window decodes it again and replaces it.
    let mut provisional: Option<usize> = None;
    // The recovery of a region that whisper did not transcribe.
    let mut recovery = Recovery::default();
    // The regions that whisper did not transcribe and that the recovery
    // could not fix. The transcript may be missing them, so they are
    // reported to the user.
    let mut unresolved = UnresolvedRegions::default();
    report(40);
    while let Some(mut current) = window.take() {
        window_count += 1;
        let mut segments = decode_window(&mut state, &full_params, &current)?;
        // Whether the window was decoded again to recover a region that
        // whisper did not transcribe.
        let mut recovered = false;
        trace_window(&current, &segments);

        // whisper does not always report a segment for the beginning of the
        // audio it is given. When it does not, the region in front of the
        // first segment was not transcribed, and the window has to be
        // decoded again from a different position. The attempts are
        // bounded, and the region that could not be recovered is reported.
        loop {
            let Some(hole) = detect_hole(&current, &segments, frontier) else {
                break;
            };
            if !recovery_enabled() {
                // The recovery is switched off, so the region stays
                // untranscribed and the user has to know.
                if region_may_carry_speech(&current, &hole) {
                    unresolved.add(UnresolvedRegion {
                        start: hole.start,
                        end: hole.end,
                    });
                }
                break;
            }
            let attempt = recovery.next_attempt(&hole, &current, windows.retained_from());
            trace_hole(&current, &hole, attempt);
            let rewind = match attempt {
                RecoveryAttempt::Decode(rewind) => rewind,
                RecoveryAttempt::NoSpeech => break,
                RecoveryAttempt::Exhausted => {
                    // The region carries speech that whisper did not
                    // report and that could not be decoded again. The
                    // transcript may be missing it, so the user has to know.
                    unresolved.add(UnresolvedRegion {
                        start: hole.start,
                        end: hole.end,
                    });
                    break;
                }
            };
            windows.seek_to(rewind);
            let Some(next) = windows.next_window()? else {
                break;
            };
            window_count += 1;
            current = next;
            segments = decode_window(&mut state, &full_params, &current)?;
            recovered = true;
            trace_window(&current, &segments);
        }
        recovery.reset();

        let window_end = current.end_seconds();
        // Whether the segments come from a window that was decoded again to
        // recover a region that whisper did not transcribe.
        let filling_hole = recovered;
        // The index of the last segment this window reported.
        let mut last_index: Option<usize> = None;
        for segment in &segments {
            let start = segment.start as f64 / crate::chunked::WHISPER_SAMPLE_RATE as f64;
            // whisper reports the end of a segment behind the audio it has
            // decoded when the mel is padded with silence. The segment ends
            // with the window, so that the timestamps stay inside the
            // recording.
            let end =
                (segment.end as f64 / crate::chunked::WHISPER_SAMPLE_RATE as f64).min(window_end);

            // A segment that reaches into the region that whisper did not
            // transcribe is kept even when it starts inside the transcribed
            // part: it is the only text for that region. Its start is
            // moved to the end of the transcribed timeline, so that the
            // segments stay ordered and do not overlap.
            let fills_hole = filling_hole && start < emitted_end && end > emitted_end;
            let action = segment_action(&entries, provisional, start, end, emitted_end, fills_hole);

            // The re-decode of the provisional segment replaces it: the
            // version of the previous window may have been cut short by
            // the end of that window. It is kept, because dropping it
            // would lose the words that the cut version did not hear.
            match action {
                SegmentAction::Replace(i) => {
                    entries[i] = OutputEntry {
                        id: 0,
                        speaker: speaker.to_string(),
                        start,
                        end,
                        text: segment.text.clone(),
                    };
                    emitted_end = end;
                    frontier = segment
                        .end
                        .min(current.start_sample + current.samples.len() as u64);
                    last_index = Some(i);
                    // The re-decode is the complete version of the segment.
                    provisional = None;
                }
                SegmentAction::Skip => {}
                SegmentAction::Emit => {
                    entries.push(OutputEntry {
                        id: 0,
                        speaker: speaker.to_string(),
                        start: if fills_hole { emitted_end } else { start },
                        end,
                        text: segment.text.clone(),
                    });
                    emitted_end = end;
                    frontier = segment
                        .end
                        .min(current.start_sample + current.samples.len() as u64);
                    last_index = Some(entries.len() - 1);
                }
            }
        }

        // Where the next window starts:
        //
        // * A segment that was cut short by the end of the window is
        //   decoded again, with context, in the next window. That window
        //   starts at the beginning of the segment, so the text is
        //   complete.
        // * Otherwise the next window starts where the last segment ended,
        //   so the audio in between is decoded again and no part of the
        //   recording is skipped. The window still advances by at least half
        //   a window, so that a stretch of silence, where whisper reports
        //   short segments, is not decoded over and over.
        // * A window without any segment advances by at least half a window.
        let rate = crate::chunked::WHISPER_SAMPLE_RATE as f64;
        let min_advance = (MIN_WINDOW_ADVANCE_SECONDS * rate) as u64;
        let window_start_sample = current.start_sample;

        // The last segment of the window may have been cut short by the end
        // of the window. It is kept, and the next window replaces it.
        provisional = match last_index {
            Some(i) if segment_is_cut(entries[i].end, window_end) => Some(i),
            _ => None,
        };

        let next_start = match provisional {
            // Decode the cut segment again, with the context it needs.
            Some(i) => (entries[i].start * rate).round() as u64,
            None => match entries.last() {
                Some(last) => {
                    ((last.end * rate).round() as u64).max(window_start_sample + min_advance)
                }
                None => (window_end * rate).round() as u64,
            },
        };
        // The pipeline always moves forward: a window that would start at
        // the position of the current one would be decoded again and again.
        let next_start = next_start.max(window_start_sample + 1);

        windows.seek_to(next_start);
        window = windows.next_window()?;
    }

    // A region that whisper did not transcribe and that the recovery could
    // not fix means that the transcript may be missing speech. The warning
    // goes to stderr and does not change the exit status: the transcription
    // itself succeeded.
    if let Some(warning) = unresolved.warning(audio_path) {
        crate::ui::warn(warning);
    }

    crate::dlog!(1, "Transcribed {window_count} window(s) of audio");
    report(90);
    report(100);
    Ok(entries)
}

/// A segment that whisper reported for a window.
#[derive(Debug, Clone, PartialEq)]
struct RawSegment {
    /// The start of the segment, in samples of the whole recording.
    start: u64,
    /// The end of the segment, in samples of the whole recording.
    end: u64,
    /// The text of the segment.
    text: String,
}

/// The region at the beginning of a window that whisper did not transcribe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Hole {
    /// The first sample of the region.
    start: u64,
    /// The sample after the region.
    end: u64,
}

impl Hole {
    /// The length of the region in samples.
    fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// The region at the beginning of a window that whisper did not transcribe.
///
/// whisper reports the segments of a window relative to the window. When its
/// first segment starts well after the beginning of the window, the audio in
/// between was not transcribed: whisper skipped it. The pipeline advances to
/// the last segment of the window, so the region would be lost.
fn detect_hole(window: &AudioWindow, segments: &[RawSegment], frontier: u64) -> Option<Hole> {
    let start = window.start_sample.max(frontier);
    let end = segments
        .first()
        .map(|s| s.start)
        .unwrap_or_else(|| window.start_sample + window.samples.len() as u64);
    if end.saturating_sub(start) >= HOLE_MIN_SAMPLES {
        Some(Hole { start, end })
    } else {
        None
    }
}

/// The level of a range of samples, in dBFS.
fn rms_dbfs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    let rms = (sum / samples.len() as f64).sqrt();
    if rms <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * rms.log10()
    }
}

/// The share of the frames of a region that carry speech level energy.
///
/// The level of the whole region is not enough: a region of silence that
/// ends in the first words of a segment carries that speech, and the region
/// is not speech that was missed. The share of the frames that carry
/// energy tells the two apart.
fn speech_frame_ratio(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut loud = 0usize;
    let mut frames = 0usize;
    for frame in samples.chunks(ENERGY_FRAME_SAMPLES) {
        frames += 1;
        if rms_dbfs(frame) > HOLE_SPEECH_LEVEL_DBFS {
            loud += 1;
        }
    }
    loud as f64 / frames as f64
}

/// Whether the audio of a region carries enough speech to be speech that
/// whisper may have missed.
///
/// The check is a level and a share of frames, not a voice activity
/// detector: it tells silence from something that was heard, and it does
/// not claim that the speech in the region was understood. A region below
/// the share is not decoded again, so a recording with long silences does
/// not pay for a retry per silence.
fn region_may_carry_speech(window: &AudioWindow, hole: &Hole) -> bool {
    let start = hole.start.saturating_sub(window.start_sample) as usize;
    let end = (hole.end.saturating_sub(window.start_sample)) as usize;
    let end = end.min(window.samples.len());
    if start >= end {
        return false;
    }
    speech_frame_ratio(&window.samples[start..end]) > HOLE_SPEECH_RATIO
}

/// What the recovery does with a region that whisper did not transcribe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryAttempt {
    /// The window is decoded again from this position.
    Decode(u64),
    /// The region carries no speech, so it is not decoded again.
    NoSpeech,
    /// The attempts are used up, or the audio they need is not available.
    Exhausted,
}

/// The recovery of a region that whisper did not transcribe.
///
/// The window is decoded again from a slightly different position, because
/// whisper's behaviour depends on where the window starts: a window that
/// starts a fraction of a second earlier or later reports the speech that
/// the first window skipped. The attempts are bounded, and each one is
/// reported.
#[derive(Debug, Default)]
struct Recovery {
    /// The attempts that were made for the current region.
    attempts: u32,
}

impl Recovery {
    /// What to do with the region that whisper did not transcribe.
    fn next_attempt(
        &mut self,
        hole: &Hole,
        window: &AudioWindow,
        retained_from: u64,
    ) -> RecoveryAttempt {
        if !region_may_carry_speech(window, hole) {
            return RecoveryAttempt::NoSpeech;
        }
        while (self.attempts as usize) < RECOVERY_SHIFTS.len() {
            let shift = RECOVERY_SHIFTS[self.attempts as usize];
            self.attempts += 1;
            let target = if shift < 0 {
                hole.start.saturating_sub(shift.unsigned_abs())
            } else {
                hole.start.saturating_add(shift as u64)
            };
            if target == window.start_sample {
                // The window would be the same one, and whisper would skip
                // the region again.
                continue;
            }
            // The audio in front of the window has to be available, otherwise
            // the window cannot start there.
            if target < retained_from {
                continue;
            }
            return RecoveryAttempt::Decode(target);
        }
        RecoveryAttempt::Exhausted
    }

    /// Forget the attempts of a region that was not a problem.
    fn reset(&mut self) {
        self.attempts = 0;
    }
}

/// A region that whisper did not transcribe and that the recovery could not
/// fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnresolvedRegion {
    /// The first sample of the region.
    start: u64,
    /// The sample after the region.
    end: u64,
}

/// The regions of a recording that whisper did not transcribe and that the
/// recovery could not fix.
///
/// A region like that means that the transcript may be missing speech. The
/// user has to know that, so the regions are reported in a warning.
#[derive(Debug, Default)]
struct UnresolvedRegions {
    regions: Vec<UnresolvedRegion>,
}

impl UnresolvedRegions {
    fn add(&mut self, region: UnresolvedRegion) {
        self.regions.push(region);
    }

    /// The warning for the regions, if there are any.
    ///
    /// The warning says that speech *may* be missing: the check that decides
    /// whether a region carries speech is a level and a share of frames, not
    /// a voice activity detector.
    fn warning(&self, path: &Path) -> Option<String> {
        if self.regions.is_empty() {
            return None;
        }
        let rate = crate::chunked::WHISPER_SAMPLE_RATE as f64;
        let mut ranges = Vec::new();
        for region in self.regions.iter().take(MAX_REPORTED_REGIONS) {
            ranges.push(format!(
                "between {:.1} s and {:.1} s",
                region.start as f64 / rate,
                region.end as f64 / rate
            ));
        }
        let mut text = ranges.join(", ");
        if self.regions.len() > MAX_REPORTED_REGIONS {
            text.push_str(&format!(
                " and {} more",
                self.regions.len() - MAX_REPORTED_REGIONS
            ));
        }
        Some(format!(
            "{}: the transcript may be incomplete. whisper did not report the audio {}, and it could not be decoded again. Speech may be missing there.",
            path.display(),
            text
        ))
    }
}

/// How many regions the warning names.
const MAX_REPORTED_REGIONS: usize = 5;

/// Whether the recovery of a region that whisper did not transcribe is on.
///
/// The switch exists for the measurements: the pipeline without the
/// recovery is the one of the previous release, and the two are compared
/// with the same binary.
fn recovery_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env::var("POLYSCRIBE_DISABLE_RECOVERY").is_err())
}

/// Whether the window handling is traced.
///
/// The trace is off by default and goes to stderr, so it does not disturb
/// the output of the CLI.
fn trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env::var("POLYSCRIBE_WINDOW_TRACE").is_ok_and(|v| v != "0"))
}

/// Trace the window that is decoded.
fn trace_window(window: &AudioWindow, segments: &[RawSegment]) {
    if !trace_enabled() {
        return;
    }
    let rate = crate::chunked::WHISPER_SAMPLE_RATE as f64;
    let checksum: f64 = window.samples.iter().map(|s| *s as f64).sum();
    eprintln!(
        "TRACE window start_sample={} end_sample={} start={:.3}s segments={} checksum={:.6} first={:?}",
        window.start_sample,
        window.start_sample + window.samples.len() as u64,
        window.start_sample as f64 / rate,
        segments.len(),
        checksum,
        &window.samples[..window.samples.len().min(3)]
    );
    for segment in segments {
        eprintln!(
            "TRACE   segment start_sample={} end_sample={} rel=[{:.3},{:.3}) global=[{:.3},{:.3}) text={:?}",
            segment.start,
            segment.end,
            (segment.start - window.start_sample) as f64 / rate,
            (segment.end - window.start_sample) as f64 / rate,
            segment.start as f64 / rate,
            segment.end as f64 / rate,
            segment.text
        );
    }
}

/// Trace the decision about a region that whisper did not transcribe.
fn trace_hole(window: &AudioWindow, hole: &Hole, attempt: RecoveryAttempt) {
    if !trace_enabled() {
        return;
    }
    let rate = crate::chunked::WHISPER_SAMPLE_RATE as f64;
    let level = {
        let start = hole.start.saturating_sub(window.start_sample) as usize;
        let end = (hole.end.saturating_sub(window.start_sample) as usize).min(window.samples.len());
        rms_dbfs(&window.samples[start.min(end)..end])
    };
    let start = hole.start.saturating_sub(window.start_sample) as usize;
    let end = (hole.end.saturating_sub(window.start_sample) as usize).min(window.samples.len());
    let speech = speech_frame_ratio(&window.samples[start.min(end)..end]);
    match attempt {
        RecoveryAttempt::Decode(rewind) => eprintln!(
            "TRACE   hole start_sample={} end_sample={} len={:.3}s level={:.1}dBFS speech={:.3} action=recover rewind_sample={}",
            hole.start,
            hole.end,
            hole.len() as f64 / rate,
            level,
            speech,
            rewind
        ),
        RecoveryAttempt::NoSpeech => eprintln!(
            "TRACE   hole start_sample={} end_sample={} len={:.3}s level={:.1}dBFS speech={:.3} action=silence",
            hole.start,
            hole.end,
            hole.len() as f64 / rate,
            level,
            speech
        ),
        RecoveryAttempt::Exhausted => eprintln!(
            "TRACE   hole start_sample={} end_sample={} len={:.3}s level={:.1}dBFS speech={:.3} action=unresolved",
            hole.start,
            hole.end,
            hole.len() as f64 / rate,
            level,
            speech
        ),
    }
}

/// Decode a window and return the segments whisper reported for it.
///
/// The timestamps of the segments are relative to the window, so they are
/// moved to the position of the window in the recording. The end of a
/// segment is kept inside the window: whisper reports the end of a segment
/// behind the audio it has decoded when the mel is padded with silence.
fn decode_window(
    state: &mut whisper_rs::WhisperState,
    full_params: &whisper_rs::FullParams<'_, '_>,
    window: &AudioWindow,
) -> Result<Vec<RawSegment>> {
    crate::with_suppressed_stderr(|| {
        state
            .full(full_params.clone(), &window.samples)
            .map_err(|e| anyhow!("Whisper full() failed: {:?}", e))
    })?;

    let rate = crate::chunked::WHISPER_SAMPLE_RATE as u64;
    let window_end = window.start_sample + window.samples.len() as u64;
    let mut segments = Vec::new();
    for seg in state.as_iter() {
        let text = seg
            .to_str()
            .map_err(|e| anyhow!("Failed to get segment text: {:?}", e))?;
        let start = window.start_sample + (seg.start_timestamp() as u64) * rate / 100;
        let end = (window.start_sample + (seg.end_timestamp() as u64) * rate / 100).min(window_end);
        segments.push(RawSegment {
            start,
            end,
            text: text.trim().to_string(),
        });
    }
    Ok(segments)
}

/// What to do with a segment that a window reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentAction {
    /// Replace the segment at this index: it is the re-decode of a segment
    /// that may have been cut short by the end of the previous window.
    Replace(usize),
    /// The segment repeats speech that was already reported.
    Skip,
    /// The segment is new.
    Emit,
}

/// Decide what to do with a segment that a window reported.
///
/// A segment without a duration carries no time. whisper reports such a
/// segment at the end of a window, with the text of the speech around it,
/// so it is dropped: it would show up as a subtitle without a duration,
/// overlapping the segment next to it.
///
/// A segment that overlaps the provisional one is the re-decode of a
/// segment that may have been cut short by the end of the previous
/// window, and replaces it: the cut version can be missing words. A
/// segment that starts inside an already reported segment repeats speech
/// that was transcribed, and is dropped.
///
/// A segment of a recovery window that reaches beyond the transcribed
/// timeline fills the region that whisper did not transcribe. It is kept,
/// because it is the only text for that region.
fn segment_action(
    entries: &[OutputEntry],
    provisional: Option<usize>,
    segment_start: f64,
    segment_end: f64,
    emitted_end: f64,
    filling_hole: bool,
) -> SegmentAction {
    if segment_end <= segment_start {
        return SegmentAction::Skip;
    }
    if let Some(i) = provisional
        && segment_start < entries[i].end
        && segment_end > entries[i].start
    {
        return SegmentAction::Replace(i);
    }
    if filling_hole && segment_start < emitted_end && segment_end > emitted_end {
        return SegmentAction::Emit;
    }
    if segment_is_duplicate(entries, segment_start, emitted_end) {
        return SegmentAction::Skip;
    }
    SegmentAction::Emit
}

/// Whether a segment of a window repeats what an earlier window already
/// reported.
///
/// A segment that starts before the end of the last emitted segment was
/// transcribed already, usually because the window overlaps the previous
/// one.
fn segment_is_duplicate(entries: &[OutputEntry], segment_start: f64, emitted_end: f64) -> bool {
    if segment_start >= emitted_end - SEGMENT_OVERLAP_TOLERANCE {
        return false;
    }
    // The segment starts before the last emitted segment ended, so it
    // repeats speech that was already reported.
    entries
        .iter()
        .any(|e| e.start <= segment_start && segment_start < e.end)
}

/// Whether the last segment of a window can be cut short by the end of the
/// window.
///
/// The audio of a window ends with the window, so whisper has no context
/// beyond it and can only report the part of the speech it heard. Such a
/// segment is decoded again in the next window.
fn segment_is_cut(segment_end: f64, window_end: f64) -> bool {
    segment_end >= window_end - SEGMENT_CUT_TOLERANCE
}

/// How much a segment may start before the end of the last emitted segment
/// before it counts as a duplicate.
///
/// whisper works on 10 millisecond steps, so the start of a segment and the
/// end of the previous one can differ by a step.
const SEGMENT_OVERLAP_TOLERANCE: f64 = 0.05;

/// How close to the end of a window a segment has to reach to be treated as
/// cut short by it.
const SEGMENT_CUT_TOLERANCE: f64 = 0.02;

/// The smallest distance a window advances.
///
/// whisper reports short segments for silence, so a window that started
/// where the last segment ended would decode the same stretch of silence
/// again and again. Half a window is enough to keep the windows moving
/// while the audio at the boundary is still decoded with context.
const MIN_WINDOW_ADVANCE_SECONDS: f64 = 15.0;

/// The smallest region at the beginning of a window that is treated as
/// audio that whisper did not transcribe.
const HOLE_MIN_SAMPLES: u64 = crate::chunked::WHISPER_SAMPLE_RATE as u64;

/// The level below which a frame of audio is treated as silence.
///
/// It is low enough that quiet speech is still heard, and high enough that
/// the noise floor of a recording is not.
const HOLE_SPEECH_LEVEL_DBFS: f64 = -45.0;

/// The share of the frames of a region that have to carry speech level
/// energy before the region is decoded again.
const HOLE_SPEECH_RATIO: f64 = 0.05;

/// The length of the frames the energy of a region is measured in.
const ENERGY_FRAME_SAMPLES: usize = 320;

/// The positions a window is decoded again from, relative to the beginning
/// of the region that whisper did not transcribe.
///
/// whisper skips the beginning of a window for some alignments and not for
/// others, and the difference is a fraction of a second: a window that
/// starts a quarter of a second earlier reports the speech that the first
/// window skipped. The attempts therefore use small offsets in both
/// directions, so that one of them is very likely to report the region.
/// The attempts are bounded, so a region that cannot be recovered does not
/// cost more than this list.
///
/// A negative offset looks back: the segment that is decoded again covers
/// the region, and its text repeats the words of the overlap. A positive
/// offset looks ahead: the region is covered as well, but the first
/// fractions of a second of it are left out. The lookback is the fallback
/// for the case that the audio in front of the window is not available.
const RECOVERY_SHIFTS: [i64; 6] = [
    -(crate::chunked::WHISPER_SAMPLE_RATE as i64) / 4,
    crate::chunked::WHISPER_SAMPLE_RATE as i64 / 4,
    -(crate::chunked::WHISPER_SAMPLE_RATE as i64) / 2,
    crate::chunked::WHISPER_SAMPLE_RATE as i64 / 2,
    -(crate::chunked::WHISPER_SAMPLE_RATE as i64),
    crate::chunked::WHISPER_SAMPLE_RATE as i64,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// Serializes tests that mutate process environment variables.
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    /// Set the test-only backend detection overrides explicitly. Setting a
    /// variable (even to "0") forces the value, which keeps these tests
    /// independent of the cargo features the suite is compiled with.
    fn force_backend_env(cuda: bool, hip: bool) {
        unsafe {
            std::env::set_var("POLYSCRIBE_TEST_FORCE_CUDA", if cuda { "1" } else { "0" });
            std::env::set_var("POLYSCRIBE_TEST_FORCE_HIP", if hip { "1" } else { "0" });
        }
    }

    /// Pretend that a GPU device is (not) present.
    fn force_device_env(present: bool) {
        unsafe {
            std::env::set_var(ENV_TEST_GPU_DEVICES, if present { "1" } else { "0" });
        }
    }

    /// Remove all test overrides again.
    fn clear_overrides() {
        unsafe {
            std::env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std::env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
            std::env::remove_var(ENV_TEST_GPU_DEVICES);
        }
    }

    fn entry(start: f64, end: f64) -> OutputEntry {
        OutputEntry {
            id: 0,
            speaker: "speaker".to_string(),
            start,
            end,
            text: "text".to_string(),
        }
    }

    #[test]
    fn test_segment_action_replaces_the_provisional_segment() {
        let entries = vec![entry(10.0, 15.0), entry(20.0, 25.0)];
        // The provisional segment is the last one, which the next window
        // decoded again. The re-decode replaces it, because the earlier
        // version may have been cut short by the end of its window.
        assert_eq!(
            segment_action(&entries, Some(1), 20.1, 25.4, 25.0, false),
            SegmentAction::Replace(1)
        );
        // A re-decode that starts a little before the provisional segment
        // still replaces it.
        assert_eq!(
            segment_action(&entries, Some(1), 19.9, 25.2, 25.0, false),
            SegmentAction::Replace(1)
        );
    }

    #[test]
    fn test_segment_action_skips_a_repeated_segment() {
        let entries = vec![entry(10.0, 15.0), entry(20.0, 25.0)];
        // A segment that starts inside an already reported segment
        // repeats speech that was transcribed.
        assert_eq!(
            segment_action(&entries, None, 22.0, 23.0, 25.0, false),
            SegmentAction::Skip
        );
    }

    #[test]
    fn test_segment_action_emits_a_new_segment() {
        let entries = vec![entry(10.0, 15.0)];
        assert_eq!(
            segment_action(&entries, None, 20.0, 25.0, 15.0, false),
            SegmentAction::Emit
        );
    }

    #[test]
    fn test_segment_action_skips_a_segment_without_a_duration() {
        let entries = vec![entry(10.0, 15.0)];
        // whisper reports such a segment at the end of a window. It
        // carries no time and would overlap the segment next to it.
        assert_eq!(
            segment_action(&entries, None, 20.0, 20.0, 15.0, false),
            SegmentAction::Skip
        );
    }

    #[test]
    fn test_segment_action_keeps_a_segment_that_fills_a_hole() {
        let entries = vec![entry(10.0, 15.0)];
        // The segment of a recovery window reaches into the region that
        // whisper did not transcribe. It is the only text for that region,
        // so it is kept even though it starts inside the transcribed part.
        assert_eq!(
            segment_action(&entries, None, 14.0, 20.0, 15.0, true),
            SegmentAction::Emit
        );
        // Without the recovery the same segment repeats what was
        // transcribed already.
        assert_eq!(
            segment_action(&entries, None, 14.0, 20.0, 15.0, false),
            SegmentAction::Skip
        );
    }

    fn window(start_sample: u64, samples: usize) -> AudioWindow {
        AudioWindow {
            start_sample,
            samples: vec![0.0; samples],
        }
    }

    fn raw_segment(start: u64, end: u64) -> RawSegment {
        RawSegment {
            start,
            end,
            text: "text".to_string(),
        }
    }

    #[test]
    fn test_detect_hole_finds_a_late_first_segment() {
        let w = window(0, 480_000);
        // The first segment starts 22 seconds into the window: the audio in
        // front of it was not transcribed.
        let hole = detect_hole(&w, &[raw_segment(353_280, 480_000)], 0).unwrap();
        assert_eq!(hole.start, 0);
        assert_eq!(hole.end, 353_280);
        assert_eq!(hole.len(), 353_280);
    }

    #[test]
    fn test_detect_hole_ignores_a_window_that_starts_with_speech() {
        let w = window(0, 480_000);
        assert!(detect_hole(&w, &[raw_segment(0, 5_000)], 0).is_none());
    }

    #[test]
    fn test_detect_hole_covers_a_window_without_segments() {
        let w = window(0, 480_000);
        // whisper reported nothing at all: the whole window is the region
        // that was not transcribed.
        let hole = detect_hole(&w, &[], 0).unwrap();
        assert_eq!(hole.start, 0);
        assert_eq!(hole.end, 480_000);
    }

    #[test]
    fn test_detect_hole_starts_at_the_transcribed_timeline() {
        let w = window(480_000, 480_000);
        // The window starts before the end of the transcribed timeline, so
        // the region starts there and not at the start of the window.
        let hole = detect_hole(&w, &[raw_segment(600_000, 700_000)], 500_000).unwrap();
        assert_eq!(hole.start, 500_000);
        assert_eq!(hole.end, 600_000);
    }

    #[test]
    fn test_recovery_uses_bounded_attempts() {
        let mut w = window(100_000, 480_000);
        w.samples[0..100_000].iter_mut().for_each(|s| *s = 0.5);
        let hole = Hole {
            start: 100_000,
            end: 453_280,
        };
        let mut recovery = Recovery::default();
        let mut attempts = Vec::new();
        loop {
            match recovery.next_attempt(&hole, &w, 0) {
                RecoveryAttempt::Decode(target) => {
                    attempts.push(target);
                    assert!(
                        target != w.start_sample,
                        "the window must not be decoded again from the same position"
                    );
                    assert!(attempts.len() <= RECOVERY_SHIFTS.len());
                }
                RecoveryAttempt::NoSpeech => panic!("the region carries speech"),
                RecoveryAttempt::Exhausted => break,
            }
        }
        // Every shift of the list is used once, and the list is the bound.
        assert_eq!(attempts.len(), RECOVERY_SHIFTS.len());
        // The first attempt looks back, so that the segment that is decoded
        // again covers the region.
        assert!(attempts[0] < hole.start);
        // The attempts move away from the region in both directions.
        assert!(attempts.iter().any(|t| *t > hole.start));
    }

    #[test]
    fn test_recovery_skips_a_shift_without_audio_in_front() {
        let mut w = window(0, 480_000);
        w.samples[0..100_000].iter_mut().for_each(|s| *s = 0.5);
        let hole = Hole {
            start: 0,
            end: 353_280,
        };
        let mut recovery = Recovery::default();
        // The region starts at the beginning of the recording, so the shifts
        // that look back cannot be used.
        match recovery.next_attempt(&hole, &w, 0) {
            RecoveryAttempt::Decode(target) => assert!(target > hole.start),
            other => panic!("expected a decode attempt, got {other:?}"),
        }
    }

    #[test]
    fn test_recovery_skips_a_region_without_speech() {
        let w = window(0, 480_000);
        let hole = Hole {
            start: 0,
            end: 353_280,
        };
        let mut recovery = Recovery::default();
        // The region is silence, so it is not decoded again.
        assert_eq!(
            recovery.next_attempt(&hole, &w, 0),
            RecoveryAttempt::NoSpeech
        );
    }

    #[test]
    fn test_recovery_stops_when_the_lookback_is_not_available() {
        let mut w = window(100_000, 480_000);
        w.samples[0..100_000].iter_mut().for_each(|s| *s = 0.5);
        let hole = Hole {
            start: 100_000,
            end: 453_280,
        };
        let mut recovery = Recovery::default();
        // The audio in front of the window is gone, so a window that starts
        // before the hole cannot be decoded. The attempt that looks ahead
        // is used instead.
        match recovery.next_attempt(&hole, &w, 100_000) {
            RecoveryAttempt::Decode(target) => assert!(target > hole.start),
            other => panic!("expected a decode attempt, got {other:?}"),
        }
    }

    #[test]
    fn test_no_warning_without_an_unresolved_region() {
        let regions = UnresolvedRegions::default();
        assert_eq!(regions.warning(Path::new("recording.wav")), None);
    }

    #[test]
    fn test_the_warning_names_the_file_and_the_region() {
        let mut regions = UnresolvedRegions::default();
        regions.add(UnresolvedRegion {
            start: 1_244_800,
            end: 1_598_080,
        });
        let warning = regions
            .warning(Path::new("/tmp/recording.wav"))
            .expect("a region produces a warning");
        assert!(warning.contains("/tmp/recording.wav"), "{warning}");
        assert!(warning.contains("between 77.8 s and 99.9 s"), "{warning}");
        assert!(
            warning.contains("the transcript may be incomplete"),
            "{warning}"
        );
        assert!(
            warning.contains("Speech may be missing"),
            "the warning must not claim that speech was there: {warning}"
        );
    }

    #[test]
    fn test_the_warning_names_a_bounded_number_of_regions() {
        let mut regions = UnresolvedRegions::default();
        for i in 0..MAX_REPORTED_REGIONS + 3 {
            regions.add(UnresolvedRegion {
                start: i as u64 * 16_000,
                end: i as u64 * 16_000 + 8_000,
            });
        }
        let warning = regions.warning(Path::new("recording.wav")).unwrap();
        assert!(warning.contains("and 3 more"), "{warning}");
    }

    #[test]
    fn test_speech_frame_ratio_separates_silence_from_speech() {
        let silence = vec![0.0f32; 320_000];
        assert_eq!(speech_frame_ratio(&silence), 0.0);

        let mut speech = vec![0.0f32; 320_000];
        speech[0..32_000].iter_mut().for_each(|s| *s = 0.5);
        assert!(speech_frame_ratio(&speech) > HOLE_SPEECH_RATIO);
        // A region that is silence with a few words in it is silence: the
        // words belong to the segment that whisper reported.
        let mut mostly_silence = vec![0.0f32; 320_000];
        mostly_silence[319_000..320_000]
            .iter_mut()
            .for_each(|s| *s = 0.5);
        assert!(speech_frame_ratio(&mostly_silence) < HOLE_SPEECH_RATIO);
    }

    #[test]
    fn test_segment_duplicate_detection() {
        let entries = vec![entry(10.0, 15.0)];
        // A segment that starts before the end of the last emitted segment
        // was already reported.
        assert!(segment_is_duplicate(&entries, 14.0, 15.0));
        // A segment that starts at the end of the last one is new.
        assert!(!segment_is_duplicate(&entries, 15.0, 15.0));
        // A segment that starts a whisper step earlier is not a duplicate.
        assert!(!segment_is_duplicate(&entries, 14.98, 15.0));
        // Without any earlier segment nothing is a duplicate.
        assert!(!segment_is_duplicate(&[], 1.0, 5.0));
    }

    #[test]
    fn test_cut_detection() {
        // A segment that reaches the end of the window can be cut short by
        // it and is decoded again in the next window.
        assert!(segment_is_cut(30.0, 30.0));
        assert!(segment_is_cut(30.1, 30.0));
        assert!(segment_is_cut(29.99, 30.0));
        // A segment that ends well before the end of the window is complete.
        assert!(!segment_is_cut(25.0, 30.0));
    }

    #[test]
    fn test_backend_auto_order_prefers_cuda_then_hip_then_cpu() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // No GPU -> CPU
        force_backend_env(false, false);
        force_device_env(false);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cpu);
        // HIP only
        force_backend_env(false, true);
        force_device_env(true);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Hip);
        // CUDA preferred over HIP
        force_backend_env(true, true);
        force_device_env(true);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cuda);
        // Cleanup
        clear_overrides();
    }

    #[test]
    fn test_backend_explicit_missing_errors() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // Ensure all off
        force_backend_env(false, false);
        force_device_env(true);
        assert!(select_backend(BackendKind::Cuda, false).is_err());
        assert!(select_backend(BackendKind::Hip, false).is_err());
        // Turn on CUDA only
        force_backend_env(true, false);
        assert!(select_backend(BackendKind::Cuda, false).is_ok());
        // Turn on HIP only
        force_backend_env(false, true);
        assert!(select_backend(BackendKind::Hip, false).is_ok());
        // Cleanup
        clear_overrides();
    }

    #[test]
    fn test_backend_explicit_gpu_without_device_errors() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // The backend is compiled in, but no device is visible.
        force_backend_env(true, true);
        force_device_env(false);

        let err = match select_backend(BackendKind::Cuda, false) {
            Ok(_) => panic!("expected an error for a CUDA request without a device"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no CUDA device is available"),
            "unexpected error: {msg}"
        );
        assert!(
            !msg.contains("built without CUDA support"),
            "the backend is compiled in, only the device is missing: {msg}"
        );

        let err = match select_backend(BackendKind::Hip, false) {
            Ok(_) => panic!("expected an error for a HIP request without a device"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no compatible GPU device is available"),
            "unexpected error: {msg}"
        );
        assert!(
            !msg.contains("built without HIP support"),
            "the backend is compiled in, only the device is missing: {msg}"
        );
        clear_overrides();
    }

    #[test]
    fn test_backend_auto_falls_back_to_cpu_without_device() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // A GPU backend is compiled in, but no device is visible: auto has to
        // fall back to the CPU instead of failing.
        force_backend_env(true, true);
        force_device_env(false);

        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cpu);
        assert!(sel.detected.is_empty());
        clear_overrides();
    }

    #[test]
    fn test_backend_explicit_cpu_never_needs_a_device() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // The CPU backend works without any GPU, whatever the overrides say.
        force_backend_env(true, true);
        force_device_env(false);

        let sel = select_backend(BackendKind::Cpu, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cpu);
        clear_overrides();
    }
}
