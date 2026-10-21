// SPDX-License-Identifier: MIT

use crate::OutputEntry;
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
    let mut window_count = 0u64;
    report(40);
    while let Some(current) = window.take() {
        window_count += 1;
        crate::with_suppressed_stderr(|| {
            state
                .full(full_params.clone(), &current.samples)
                .map_err(|e| anyhow!("Whisper full() failed: {:?}", e))
        })?;

        let window_start = current.start_seconds();
        let window_end = current.end_seconds();
        let mut last: Option<(f64, f64)> = None;
        for seg in state.as_iter() {
            let text = seg
                .to_str()
                .map_err(|e| anyhow!("Failed to get segment text: {:?}", e))?;
            let start = window_start + (seg.start_timestamp() as f64) * 0.01;
            let end = window_start + (seg.end_timestamp() as f64) * 0.01;
            if segment_is_duplicate(&entries, start, emitted_end) {
                // Already transcribed by an earlier window.
                continue;
            }
            entries.push(OutputEntry {
                id: 0,
                speaker: speaker.to_string(),
                start,
                end,
                text: text.trim().to_string(),
            });
            last = Some((start, end));
        }

        // Where the next window starts:
        //
        // * A segment that was cut short by the end of the window is
        //   decoded again, with context, in the next window. That window
        //   starts at the beginning of the segment, so the text is
        //   complete.
        // * Otherwise the next window starts where the last segment ended,
        //   so the audio in between is decoded again and no part of the
        //   recording is skipped.
        // * A window without any segment advances by its own length.
        let rate = crate::chunked::WHISPER_SAMPLE_RATE as f64;
        let next_start = match last {
            Some((last_start, last_end)) if segment_is_cut(last_end, window_end) => {
                entries.pop();
                emitted_end = entries.last().map(|e| e.end).unwrap_or(emitted_end);
                (last_start * rate).round() as u64
            }
            Some((_, last_end)) => {
                emitted_end = last_end;
                (last_end * rate).round() as u64
            }
            None => (window_end * rate).round() as u64,
        };

        windows.seek_to(next_start);
        window = windows.next_window()?;
    }

    crate::dlog!(1, "Transcribed {window_count} window(s) of audio");
    report(90);
    report(100);
    Ok(entries)
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
