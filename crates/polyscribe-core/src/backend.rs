// SPDX-License-Identifier: MIT

use crate::OutputEntry;
use crate::output::{is_audio_file, is_json_file, load_json_transcript};
use crate::prelude::*;
use crate::{decode_audio_to_pcm_f32_ffmpeg, find_model_file};
use anyhow::{Context, anyhow};
use std::env;
use std::path::Path;

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
        &self,
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

#[derive(Default)]
pub struct CpuBackend;
#[derive(Default)]
pub struct CudaBackend;
#[derive(Default)]
pub struct HipBackend;

macro_rules! impl_whisper_backend {
    ($ty:ty, $kind:expr) => {
        impl TranscribeBackend for $ty {
            fn kind(&self) -> BackendKind {
                $kind
            }
            fn transcribe(
                &self,
                audio_path: &Path,
                speaker: &str,
                language: Option<&str>,
                _gpu_layers: Option<u32>,
                progress: Option<&(dyn Fn(i32) + Send + Sync)>,
            ) -> Result<Vec<OutputEntry>> {
                transcribe_with_whisper_rs(audio_path, speaker, language, progress, $kind)
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
    pub detected: Vec<BackendKind>,
}

/// Select an appropriate backend based on the user request and the backends
/// compiled into this binary.
///
/// If `requested` is `BackendKind::Auto`, the function prefers CUDA, then HIP,
/// falling back to CPU when no GPU backend was compiled in. When a specific
/// GPU backend is requested but this binary was built without the matching
/// feature, an error is returned with guidance on how to enable it.
pub fn select_backend(requested: BackendKind, verbose: bool) -> Result<BackendSelection> {
    let mut detected = Vec::new();
    if cuda_available() {
        detected.push(BackendKind::Cuda);
    }
    if hip_available() {
        detected.push(BackendKind::Hip);
    }

    let instantiate_backend = |k: BackendKind| -> Box<dyn TranscribeBackend + Send + Sync> {
        match k {
            BackendKind::Cpu => Box::new(CpuBackend),
            BackendKind::Cuda => Box::new(CudaBackend),
            BackendKind::Hip => Box::new(HipBackend),
            BackendKind::Auto => Box::new(CpuBackend), // resolved before construction
        }
    };

    let chosen = match requested {
        BackendKind::Auto => {
            if detected.contains(&BackendKind::Cuda) {
                BackendKind::Cuda
            } else if detected.contains(&BackendKind::Hip) {
                BackendKind::Hip
            } else {
                BackendKind::Cpu
            }
        }
        BackendKind::Cuda => {
            if detected.contains(&BackendKind::Cuda) {
                BackendKind::Cuda
            } else {
                return Err(anyhow!(
                    "Requested CUDA backend but this binary was built without CUDA support. How to fix: install the NVIDIA driver + CUDA runtime and rebuild with --features gpu-cuda."
                ).into());
            }
        }
        BackendKind::Hip => {
            if detected.contains(&BackendKind::Hip) {
                BackendKind::Hip
            } else {
                return Err(anyhow!(
                    "Requested ROCm/HIP backend but this binary was built without HIP support. How to fix: install the ROCm runtime and rebuild with --features gpu-hip."
                ).into());
            }
        }
        BackendKind::Cpu => BackendKind::Cpu,
    };

    if verbose {
        crate::dlog!(1, "Detected backends: {:?}", detected);
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
    backend: &dyn TranscribeBackend,
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
pub(crate) fn transcribe_with_whisper_rs(
    audio_path: &Path,
    speaker: &str,
    language: Option<&str>,
    progress: Option<&(dyn Fn(i32) + Send + Sync)>,
    backend: BackendKind,
) -> Result<Vec<OutputEntry>> {
    let report = |p: i32| {
        if let Some(cb) = progress {
            cb(p);
        }
    };
    report(0);

    let pcm_samples = decode_audio_to_pcm_f32_ffmpeg(audio_path)?;
    report(5);

    let model_path = find_model_file()?;
    let english_only_model = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains(".en.") || s.ends_with(".en.bin"))
        .unwrap_or(false);
    if let Some(lang) = language
        && english_only_model
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

    // Suppress stderr from whisper/ggml during model load and inference when quiet and not verbose.
    let (_context, mut state) = crate::with_suppressed_stderr(|| {
        let mut params = whisper_rs::WhisperContextParameters::default();
        // Only a GPU backend may use the GPU. whisper-rs defaults `use_gpu`
        // to the compiled-in GPU feature, so in a GPU-enabled build a
        // "cpu" request would still run on the GPU.
        params.use_gpu(matches!(backend, BackendKind::Cuda | BackendKind::Hip));
        let context = whisper_rs::WhisperContext::new_with_params(&model_path, params)
            .with_context(|| format!("Failed to load Whisper model at {}", model_path.display()))?;
        let state = context
            .create_state()
            .map_err(|e| anyhow!("Failed to create Whisper state: {:?}", e))?;
        Ok::<_, anyhow::Error>((context, state))
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

    crate::with_suppressed_stderr(|| {
        report(40);
        state
            .full(full_params, &pcm_samples)
            .map_err(|e| anyhow!("Whisper full() failed: {:?}", e))
    })?;

    report(90);
    let mut entries = Vec::new();
    for seg in state.as_iter() {
        let text = seg
            .to_str()
            .map_err(|e| anyhow!("Failed to get segment text: {:?}", e))?;
        let start = (seg.start_timestamp() as f64) * 0.01;
        let end = (seg.end_timestamp() as f64) * 0.01;
        entries.push(OutputEntry {
            id: 0,
            speaker: speaker.to_string(),
            start,
            end,
            text: text.trim().to_string(),
        });
    }
    report(100);
    Ok(entries)
}

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

    #[test]
    fn test_backend_auto_order_prefers_cuda_then_hip_then_cpu() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // No GPU -> CPU
        force_backend_env(false, false);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cpu);
        // HIP only
        force_backend_env(false, true);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Hip);
        // CUDA preferred over HIP
        force_backend_env(true, true);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cuda);
        // Cleanup
        unsafe {
            std::env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std::env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
        }
    }

    #[test]
    fn test_backend_explicit_missing_errors() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // Ensure all off
        force_backend_env(false, false);
        assert!(select_backend(BackendKind::Cuda, false).is_err());
        assert!(select_backend(BackendKind::Hip, false).is_err());
        // Turn on CUDA only
        force_backend_env(true, false);
        assert!(select_backend(BackendKind::Cuda, false).is_ok());
        // Turn on HIP only
        force_backend_env(false, true);
        assert!(select_backend(BackendKind::Hip, false).is_ok());
        // Cleanup
        unsafe {
            std::env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std::env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
        }
    }
}
