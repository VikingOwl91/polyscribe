use crate::OutputEntry;
use crate::{decode_audio_to_pcm_f32_ffmpeg, find_model_file};
use anyhow::{Context, Result, anyhow};
#[cfg(not(test))]
use libloading::Library;
use std::env;
use std::path::Path;

// Re-export a public enum for CLI parsing usage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Kind of transcription backend to use.
pub enum BackendKind {
    /// Automatically detect the best available backend (CUDA > HIP > Vulkan > CPU).
    Auto,
    /// Pure CPU backend using whisper-rs.
    Cpu,
    /// NVIDIA CUDA backend (requires CUDA runtime available at load time and proper feature build).
    Cuda,
    /// AMD ROCm/HIP backend (requires hip/rocBLAS libraries available and proper feature build).
    Hip,
    /// Vulkan backend (experimental; requires Vulkan loader/SDK and feature build).
    Vulkan,
}

/// Abstraction for a transcription backend implementation.
pub trait TranscribeBackend {
    /// Return the backend kind for this implementation.
    fn kind(&self) -> BackendKind;
    /// Transcribe the given audio file path and return transcript entries.
    ///
    /// Parameters:
    /// - audio_path: path to input media (audio or video) to be decoded/transcribed.
    /// - speaker: label to attach to all produced segments.
    /// - lang_opt: optional language hint (e.g., "en"); None means auto/multilingual model default.
    /// - gpu_layers: optional GPU layer count if applicable (ignored by some backends).
    fn transcribe(
        &self,
        audio_path: &Path,
        speaker: &str,
        lang_opt: Option<&str>,
        gpu_layers: Option<u32>,
    ) -> Result<Vec<OutputEntry>>;
}

fn check_lib(names: &[&str]) -> bool {
    #[cfg(test)]
    {
        // During unit tests, avoid touching system libs to prevent loader crashes in CI.
        // Mark parameter as used to silence warnings in test builds.
        let _ = names;
        false
    }
    #[cfg(not(test))]
    {
        if std::env::var("POLYSCRIBE_DISABLE_DLOPEN").ok().as_deref() == Some("1") {
            return false;
        }
        for n in names {
            // Attempt to dlopen; ignore errors
            if let Ok(_lib) = unsafe { Library::new(n) } {
                return true;
            }
        }
        false
    }
}

fn cuda_available() -> bool {
    if let Ok(x) = env::var("POLYSCRIBE_TEST_FORCE_CUDA") {
        return x == "1";
    }
    check_lib(&[
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11",
        "libcublas.so",
        "libcublas.so.12",
    ])
}

fn hip_available() -> bool {
    if let Ok(x) = env::var("POLYSCRIBE_TEST_FORCE_HIP") {
        return x == "1";
    }
    check_lib(&["libhipblas.so", "librocblas.so"])
}

fn vulkan_available() -> bool {
    if let Ok(x) = env::var("POLYSCRIBE_TEST_FORCE_VULKAN") {
        return x == "1";
    }
    check_lib(&["libvulkan.so.1", "libvulkan.so"])
}

/// CPU-based transcription backend using whisper-rs.
pub struct CpuBackend;
/// CUDA-accelerated transcription backend for NVIDIA GPUs.
pub struct CudaBackend;
/// ROCm/HIP-accelerated transcription backend for AMD GPUs.
pub struct HipBackend;
/// Vulkan-based transcription backend (experimental/incomplete).
pub struct VulkanBackend;

impl CpuBackend {
    /// Create a new CPU backend instance.
    pub fn new() -> Self {
        CpuBackend
    }
}
impl CudaBackend {
    /// Create a new CUDA backend instance.
    pub fn new() -> Self {
        CudaBackend
    }
}
impl HipBackend {
    /// Create a new HIP backend instance.
    pub fn new() -> Self {
        HipBackend
    }
}
impl VulkanBackend {
    /// Create a new Vulkan backend instance.
    pub fn new() -> Self {
        VulkanBackend
    }
}

impl TranscribeBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }
    fn transcribe(
        &self,
        audio_path: &Path,
        speaker: &str,
        lang_opt: Option<&str>,
        _gpu_layers: Option<u32>,
    ) -> Result<Vec<OutputEntry>> {
        transcribe_with_whisper_rs(audio_path, speaker, lang_opt)
    }
}

impl TranscribeBackend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }
    fn transcribe(
        &self,
        audio_path: &Path,
        speaker: &str,
        lang_opt: Option<&str>,
        _gpu_layers: Option<u32>,
    ) -> Result<Vec<OutputEntry>> {
        // whisper-rs uses enabled CUDA feature at build time; call same code path
        transcribe_with_whisper_rs(audio_path, speaker, lang_opt)
    }
}

impl TranscribeBackend for HipBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Hip
    }
    fn transcribe(
        &self,
        audio_path: &Path,
        speaker: &str,
        lang_opt: Option<&str>,
        _gpu_layers: Option<u32>,
    ) -> Result<Vec<OutputEntry>> {
        transcribe_with_whisper_rs(audio_path, speaker, lang_opt)
    }
}

impl TranscribeBackend for VulkanBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }
    fn transcribe(
        &self,
        _audio_path: &Path,
        _speaker: &str,
        _lang_opt: Option<&str>,
        _gpu_layers: Option<u32>,
    ) -> Result<Vec<OutputEntry>> {
        Err(anyhow!(
            "Vulkan backend not yet wired to whisper.cpp FFI. Build with --features gpu-vulkan and ensure Vulkan SDK is installed. How to fix: install Vulkan loader (libvulkan), set VULKAN_SDK, and run cargo build --features gpu-vulkan."
        ))
    }
}

/// Result of choosing a transcription backend.
pub struct SelectionResult {
    /// The constructed backend instance to perform transcription with.
    pub backend: Box<dyn TranscribeBackend + Send + Sync>,
    /// Which backend kind was ultimately selected.
    pub chosen: BackendKind,
    /// Which backend kinds were detected as available on this system.
    pub detected: Vec<BackendKind>,
}

/// Select an appropriate backend based on user request and system detection.
///
/// If `requested` is `BackendKind::Auto`, the function prefers CUDA, then HIP,
/// then Vulkan, falling back to CPU when no GPU backend is detected. When a
/// specific GPU backend is requested but unavailable, an error is returned with
/// guidance on how to enable it.
///
/// Set `verbose` to true to print detection/selection info to stderr.
pub fn select_backend(requested: BackendKind, verbose: bool) -> Result<SelectionResult> {
    let mut detected = Vec::new();
    if cuda_available() {
        detected.push(BackendKind::Cuda);
    }
    if hip_available() {
        detected.push(BackendKind::Hip);
    }
    if vulkan_available() {
        detected.push(BackendKind::Vulkan);
    }

    let mk = |k: BackendKind| -> Box<dyn TranscribeBackend + Send + Sync> {
        match k {
            BackendKind::Cpu => Box::new(CpuBackend::new()),
            BackendKind::Cuda => Box::new(CudaBackend::new()),
            BackendKind::Hip => Box::new(HipBackend::new()),
            BackendKind::Vulkan => Box::new(VulkanBackend::new()),
            BackendKind::Auto => Box::new(CpuBackend::new()), // will be replaced
        }
    };

    let chosen = match requested {
        BackendKind::Auto => {
            if detected.contains(&BackendKind::Cuda) {
                BackendKind::Cuda
            } else if detected.contains(&BackendKind::Hip) {
                BackendKind::Hip
            } else if detected.contains(&BackendKind::Vulkan) {
                BackendKind::Vulkan
            } else {
                BackendKind::Cpu
            }
        }
        BackendKind::Cuda => {
            if detected.contains(&BackendKind::Cuda) {
                BackendKind::Cuda
            } else {
                return Err(anyhow!(
                    "Requested CUDA backend but CUDA libraries/devices not detected. How to fix: install NVIDIA driver + CUDA toolkit, ensure libcudart/libcublas are in loader path, and build with --features gpu-cuda."
                ));
            }
        }
        BackendKind::Hip => {
            if detected.contains(&BackendKind::Hip) {
                BackendKind::Hip
            } else {
                return Err(anyhow!(
                    "Requested ROCm/HIP backend but libraries/devices not detected. How to fix: install ROCm hipBLAS/rocBLAS, ensure libs are in loader path, and build with --features gpu-hip."
                ));
            }
        }
        BackendKind::Vulkan => {
            if detected.contains(&BackendKind::Vulkan) {
                BackendKind::Vulkan
            } else {
                return Err(anyhow!(
                    "Requested Vulkan backend but libvulkan not detected. How to fix: install Vulkan loader/SDK and build with --features gpu-vulkan."
                ));
            }
        }
        BackendKind::Cpu => BackendKind::Cpu,
    };

    if verbose {
        eprintln!("INFO: Detected backends: {:?}", detected);
        eprintln!("INFO: Selected backend: {:?}", chosen);
    }

    Ok(SelectionResult {
        backend: mk(chosen),
        chosen,
        detected,
    })
}

// Internal helper: transcription using whisper-rs with CPU/GPU (depending on build features)
#[allow(clippy::too_many_arguments)]
pub(crate) fn transcribe_with_whisper_rs(
    audio_path: &Path,
    speaker: &str,
    lang_opt: Option<&str>,
) -> Result<Vec<OutputEntry>> {
    let pcm = decode_audio_to_pcm_f32_ffmpeg(audio_path)?;
    let model = find_model_file()?;
    let is_en_only = model
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains(".en.") || s.ends_with(".en.bin"))
        .unwrap_or(false);
    if let Some(lang) = lang_opt {
        if is_en_only && lang != "en" {
            return Err(anyhow!(
                "Selected model is English-only ({}), but a non-English language hint '{}' was provided. Please use a multilingual model or set WHISPER_MODEL.",
                model.display(),
                lang
            ));
        }
    }
    let model_str = model
        .to_str()
        .ok_or_else(|| anyhow!("Model path not valid UTF-8: {}", model.display()))?;

    let cparams = whisper_rs::WhisperContextParameters::default();
    let ctx = whisper_rs::WhisperContext::new_with_params(model_str, cparams)
        .with_context(|| format!("Failed to load Whisper model at {}", model.display()))?;
    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow!("Failed to create Whisper state: {:?}", e))?;

    let mut params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1);
    params.set_n_threads(n_threads);
    params.set_translate(false);
    if let Some(lang) = lang_opt {
        params.set_language(Some(lang));
    }

    state
        .full(params, &pcm)
        .map_err(|e| anyhow!("Whisper full() failed: {:?}", e))?;

    let num_segments = state
        .full_n_segments()
        .map_err(|e| anyhow!("Failed to get segments: {:?}", e))?;
    let mut items = Vec::new();
    for i in 0..num_segments {
        let text = state
            .full_get_segment_text(i)
            .map_err(|e| anyhow!("Failed to get segment text: {:?}", e))?;
        let t0 = state
            .full_get_segment_t0(i)
            .map_err(|e| anyhow!("Failed to get segment t0: {:?}", e))?;
        let t1 = state
            .full_get_segment_t1(i)
            .map_err(|e| anyhow!("Failed to get segment t1: {:?}", e))?;
        let start = (t0 as f64) * 0.01;
        let end = (t1 as f64) * 0.01;
        items.push(OutputEntry {
            id: 0,
            speaker: speaker.to_string(),
            start,
            end,
            text: text.trim().to_string(),
        });
    }
    Ok(items)
}
