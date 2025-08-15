// SPDX-License-Identifier: MIT

use crate::OutputEntry;
use crate::prelude::*;
use crate::{decode_audio_to_pcm_f32_ffmpeg, find_model_file};
use anyhow::{Context, anyhow};
use std::env;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Auto,
    Cpu,
    Cuda,
    Hip,
    Vulkan,
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

fn is_library_available(_names: &[&str]) -> bool {
    #[cfg(test)]
    {
        false
    }
    #[cfg(not(test))]
    {
        false
    }
}

fn cuda_available() -> bool {
    if let Ok(x) = env::var("POLYSCRIBE_TEST_FORCE_CUDA") {
        return x == "1";
    }
    is_library_available(&[
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
    is_library_available(&["libhipblas.so", "librocblas.so"])
}

fn vulkan_available() -> bool {
    if let Ok(x) = env::var("POLYSCRIBE_TEST_FORCE_VULKAN") {
        return x == "1";
    }
    is_library_available(&["libvulkan.so.1", "libvulkan.so"])
}

#[derive(Default)]
pub struct CpuBackend;
#[derive(Default)]
pub struct CudaBackend;
#[derive(Default)]
pub struct HipBackend;
#[derive(Default)]
pub struct VulkanBackend;

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
                transcribe_with_whisper_rs(audio_path, speaker, language, progress)
            }
        }
    };
}

impl_whisper_backend!(CpuBackend, BackendKind::Cpu);
impl_whisper_backend!(CudaBackend, BackendKind::Cuda);
impl_whisper_backend!(HipBackend, BackendKind::Hip);

impl TranscribeBackend for VulkanBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }
    fn transcribe(
        &self,
        _audio_path: &Path,
        _speaker: &str,
        _language: Option<&str>,
        _gpu_layers: Option<u32>,
        _progress: Option<&(dyn Fn(i32) + Send + Sync)>,
    ) -> Result<Vec<OutputEntry>> {
        Err(anyhow!(
            "Vulkan backend not yet wired to whisper.cpp FFI. Build with --features gpu-vulkan and ensure Vulkan SDK is installed. How to fix: install Vulkan loader (libvulkan), set VULKAN_SDK, and run cargo build --features gpu-vulkan."
        ).into())
    }
}

pub struct BackendSelection {
    pub backend: Box<dyn TranscribeBackend + Send + Sync>,
    pub chosen: BackendKind,
    pub detected: Vec<BackendKind>,
}

pub fn select_backend(requested: BackendKind, verbose: bool) -> Result<BackendSelection> {
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

    let instantiate_backend = |k: BackendKind| -> Box<dyn TranscribeBackend + Send + Sync> {
        match k {
            BackendKind::Cpu => Box::new(CpuBackend),
            BackendKind::Cuda => Box::new(CudaBackend),
            BackendKind::Hip => Box::new(HipBackend),
            BackendKind::Vulkan => Box::new(VulkanBackend),
            BackendKind::Auto => Box::new(CpuBackend),
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
                ).into());
            }
        }
        BackendKind::Hip => {
            if detected.contains(&BackendKind::Hip) {
                BackendKind::Hip
            } else {
                return Err(anyhow!(
                    "Requested ROCm/HIP backend but libraries/devices not detected. How to fix: install ROCm hipBLAS/rocBLAS, ensure libs are in loader path, and build with --features gpu-hip."
                ).into());
            }
        }
        BackendKind::Vulkan => {
            if detected.contains(&BackendKind::Vulkan) {
                BackendKind::Vulkan
            } else {
                return Err(anyhow!(
                    "Requested Vulkan backend but libvulkan not detected. How to fix: install Vulkan loader/SDK and build with --features gpu-vulkan."
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn transcribe_with_whisper_rs(
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
    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| anyhow!("Model path not valid UTF-8: {}", model_path.display()))?;

    if crate::verbose_level() < 2 {
        unsafe {
            std::env::set_var("GGML_LOG_LEVEL", "0");
            std::env::set_var("WHISPER_PRINT_PROGRESS", "0");
        }
    }

    let (_context, mut state) = crate::with_suppressed_stderr(|| {
        let params = whisper_rs::WhisperContextParameters::default();
        let context = whisper_rs::WhisperContext::new_with_params(model_path_str, params)
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
    let num_segments = state
        .full_n_segments()
        .map_err(|e| anyhow!("Failed to get segments: {:?}", e))?;
    let mut entries = Vec::new();
    for seg_idx in 0..num_segments {
        let segment_text = state
            .full_get_segment_text(seg_idx)
            .map_err(|e| anyhow!("Failed to get segment text: {:?}", e))?;
        let t0 = state
            .full_get_segment_t0(seg_idx)
            .map_err(|e| anyhow!("Failed to get segment t0: {:?}", e))?;
        let t1 = state
            .full_get_segment_t1(seg_idx)
            .map_err(|e| anyhow!("Failed to get segment t1: {:?}", e))?;
        let start = (t0 as f64) * 0.01;
        let end = (t1 as f64) * 0.01;
        entries.push(OutputEntry {
            id: 0,
            speaker: speaker.to_string(),
            start,
            end,
            text: segment_text.trim().to_string(),
        });
    }
    report(100);
    Ok(entries)
}
