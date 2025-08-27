// SPDX-License-Identifier: MIT

#![forbid(elided_lifetimes_in_paths)]
#![forbid(unused_must_use)]
#![warn(clippy::all)]

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::prelude::*;
use anyhow::{Context, anyhow};
use chrono::Local;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use libc::{O_WRONLY, close, dup, dup2, open};

static QUIET: AtomicBool = AtomicBool::new(false);
static NO_INTERACTION: AtomicBool = AtomicBool::new(false);
static VERBOSE: AtomicU8 = AtomicU8::new(0);
static NO_PROGRESS: AtomicBool = AtomicBool::new(false);

pub fn set_quiet(enabled: bool) {
    QUIET.store(enabled, Ordering::Relaxed);
}
pub fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

pub fn set_no_interaction(enabled: bool) {
    NO_INTERACTION.store(enabled, Ordering::Relaxed);
}
pub fn is_no_interaction() -> bool {
    NO_INTERACTION.load(Ordering::Relaxed)
}

pub fn set_verbose(level: u8) {
    VERBOSE.store(level, Ordering::Relaxed);
}
pub fn verbose_level() -> u8 {
    VERBOSE.load(Ordering::Relaxed)
}

pub fn set_no_progress(enabled: bool) {
    NO_PROGRESS.store(enabled, Ordering::Relaxed);
}
pub fn is_no_progress() -> bool {
    NO_PROGRESS.load(Ordering::Relaxed)
}

pub fn stdin_is_tty() -> bool {
    use std::io::IsTerminal as _;
    std::io::stdin().is_terminal()
}

pub struct StderrSilencer {
    #[cfg(unix)]
    old_stderr_fd: i32,
    #[cfg(unix)]
    devnull_fd: i32,
    active: bool,
}

impl StderrSilencer {
    pub fn activate_if_quiet() -> Self {
        if !is_quiet() {
            return Self {
                active: false,
                #[cfg(unix)]
                old_stderr_fd: -1,
                #[cfg(unix)]
                devnull_fd: -1,
            };
        }
        Self::activate()
    }

    pub fn activate() -> Self {
        #[cfg(unix)]
        unsafe {
            let old_fd = dup(2);
            if old_fd < 0 {
                return Self {
                    active: false,
                    old_stderr_fd: -1,
                    devnull_fd: -1,
                };
            }
            let devnull_cstr = std::ffi::CString::new("/dev/null").unwrap();
            let devnull_fd = open(devnull_cstr.as_ptr(), O_WRONLY);
            if devnull_fd < 0 {
                let _ = close(old_fd);
                return Self {
                    active: false,
                    old_stderr_fd: -1,
                    devnull_fd: -1,
                };
            }
            if dup2(devnull_fd, 2) < 0 {
                let _ = close(devnull_fd);
                let _ = close(old_fd);
                return Self {
                    active: false,
                    old_stderr_fd: -1,
                    devnull_fd: -1,
                };
            }
            Self {
                active: true,
                old_stderr_fd: old_fd,
                devnull_fd,
            }
        }
        #[cfg(not(unix))]
        {
            Self { active: false }
        }
    }
}

impl Drop for StderrSilencer {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        #[cfg(unix)]
        unsafe {
            let _ = dup2(self.old_stderr_fd, 2);
            let _ = close(self.old_stderr_fd);
            let _ = close(self.devnull_fd);
        }
    }
}

pub fn with_suppressed_stderr<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let silencer = StderrSilencer::activate_if_quiet();
    let result = f();
    drop(silencer);
    result
}

#[macro_export]
macro_rules! elog {
    ($($arg:tt)*) => {{ $crate::ui::error(format!($($arg)*)); }}
}

#[macro_export]
macro_rules! ilog {
    ($($arg:tt)*) => {{
        if !$crate::is_quiet() { $crate::ui::info(format!($($arg)*)); }
    }}
}

#[macro_export]
macro_rules! dlog {
    ($lvl:expr, $($arg:tt)*) => {{
        if !$crate::is_quiet() && $crate::verbose_level() >= $lvl { $crate::ui::info(format!("DEBUG{}: {}", $lvl, format!($($arg)*))); }
    }}
}


pub mod backend;
pub mod config;
pub mod models;
pub mod error;
pub mod ui;
pub use error::Error;
pub mod prelude;

#[derive(Debug, serde::Serialize, Clone)]
pub struct OutputEntry {
    pub id: u64,
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

pub fn date_prefix() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

pub fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as i64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let sec = total_secs % 60;
    let min = (total_secs / 60) % 60;
    let hour = total_secs / 3600;
    format!("{hour:02}:{min:02}:{sec:02},{ms:03}")
}

pub fn render_srt(entries: &[OutputEntry]) -> String {
    let mut srt = String::new();
    for (index, entry) in entries.iter().enumerate() {
        let srt_index = index + 1;
        srt.push_str(&format!("{srt_index}\n"));
        srt.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(entry.start),
            format_srt_time(entry.end)
        ));
        if !entry.speaker.is_empty() {
            srt.push_str(&format!("{}: {}\n", entry.speaker, entry.text));
        } else {
            srt.push_str(&format!("{}\n", entry.text));
        }
        srt.push('\n');
    }
    srt
}

pub mod model_manager;

pub fn models_dir_path() -> PathBuf {
    if let Ok(env_val) = env::var("POLYSCRIBE_MODELS_DIR") {
        let env_path = PathBuf::from(env_val);
        if !env_path.as_os_str().is_empty() {
            return env_path;
        }
    }
    if cfg!(debug_assertions) {
        return PathBuf::from("models");
    }
    if let Ok(xdg) = env::var("XDG_DATA_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("polyscribe").join("models");
    }
    if let Ok(home) = env::var("HOME")
        && !home.is_empty()
    {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("polyscribe")
            .join("models");
    }
    PathBuf::from("models")
}

pub fn normalize_lang_code(input: &str) -> Option<String> {
    let mut lang = input.trim().to_lowercase();
    if lang.is_empty() || lang == "auto" || lang == "c" || lang == "posix" {
        return None;
    }
    if let Some((prefix, _)) = lang.split_once('.') {
        lang = prefix.to_string();
    }
    if let Some((prefix, _)) = lang.split_once('_') {
        lang = prefix.to_string();
    }
    let code = match lang.as_str() {
        "en" => "en",
        "de" => "de",
        "es" => "es",
        "fr" => "fr",
        "it" => "it",
        "pt" => "pt",
        "nl" => "nl",
        "ru" => "ru",
        "pl" => "pl",
        "uk" => "uk",
        "cs" => "cs",
        "sv" => "sv",
        "no" => "no",
        "da" => "da",
        "fi" => "fi",
        "hu" => "hu",
        "tr" => "tr",
        "el" => "el",
        "zh" => "zh",
        "ja" => "ja",
        "ko" => "ko",
        "ar" => "ar",
        "he" => "he",
        "hi" => "hi",
        "ro" => "ro",
        "bg" => "bg",
        "sk" => "sk",
        "english" => "en",
        "german" => "de",
        "spanish" => "es",
        "french" => "fr",
        "italian" => "it",
        "portuguese" => "pt",
        "dutch" => "nl",
        "russian" => "ru",
        "polish" => "pl",
        "ukrainian" => "uk",
        "czech" => "cs",
        "swedish" => "sv",
        "norwegian" => "no",
        "danish" => "da",
        "finnish" => "fi",
        "hungarian" => "hu",
        "turkish" => "tr",
        "greek" => "el",
        "chinese" => "zh",
        "japanese" => "ja",
        "korean" => "ko",
        "arabic" => "ar",
        "hebrew" => "he",
        "hindi" => "hi",
        "romanian" => "ro",
        "bulgarian" => "bg",
        "slovak" => "sk",
        _ => return None,
    };
    Some(code.to_string())
}

pub fn find_model_file() -> Result<PathBuf> {
    if let Ok(path) = env::var("WHISPER_MODEL") {
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(anyhow!(
                "WHISPER_MODEL points to a non-existing path: {}",
                p.display()
            )
            .into());
        }
        if !p.is_file() {
            return Err(anyhow!(
                "WHISPER_MODEL must point to a file, but is not: {}",
                p.display()
            )
            .into());
        }
        return Ok(p);
    }

    let models_dir = models_dir_path();
    if models_dir.exists() && !models_dir.is_dir() {
        return Err(anyhow!(
            "Models path exists but is not a directory: {}",
            models_dir.display()
        )
        .into());
    }
    std::fs::create_dir_all(&models_dir).with_context(|| {
        format!(
            "Failed to ensure models dir exists: {}",
            models_dir.display()
        )
    })?;

    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(&models_dir)
        .with_context(|| format!("Failed to read models dir: {}", models_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();

        let is_bin = path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("bin"));
        if !is_bin {
            continue;
        }

        let md = match std::fs::metadata(&path) {
            Ok(m) if m.is_file() => m,
            _ => continue,
        };

        candidates.push((md.len(), path));
    }

    if candidates.is_empty() {
        let fallback = models_dir.join("ggml-tiny.en.bin");
        if fallback.is_file() {
            return Ok(fallback);
        }
        return Err(anyhow!(
            "No Whisper model files (*.bin) found in {}. \
             Please download a model or set WHISPER_MODEL.",
            models_dir.display()
        )
        .into());
    }

    candidates.sort_by_key(|(size, _)| *size);
    let (_size, path) = candidates.into_iter().last().expect("non-empty");
    Ok(path)
}

pub fn decode_audio_to_pcm_f32_ffmpeg(audio_path: &Path) -> Result<Vec<f32>> {
    let in_path = audio_path
        .to_str()
        .ok_or_else(|| anyhow!("Audio path must be valid UTF-8: {}", audio_path.display()))?;

    let tmp_raw = std::env::temp_dir().join("polyscribe_tmp_input.f32le");
    let tmp_raw_str = tmp_raw
        .to_str()
        .ok_or_else(|| anyhow!("Temp path not valid UTF-8: {}", tmp_raw.display()))?;

    let status = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(in_path)
        .arg("-f")
        .arg("f32le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-y")
        .arg(tmp_raw_str)
        .status()
        .with_context(|| format!("Failed to invoke ffmpeg to decode: {}", in_path))?;

    if !status.success() {
        return Err(anyhow!(
            "ffmpeg exited with non-zero status when decoding {}",
            in_path
        )
        .into());
    }

    let raw = std::fs::read(&tmp_raw)
        .with_context(|| format!("Failed to read temp PCM file: {}", tmp_raw.display()))?;

    let _ = std::fs::remove_file(&tmp_raw);

    if raw.len() % 4 != 0 {
        return Err(anyhow!("Decoded PCM file length not multiple of 4: {}", raw.len()).into());
    }
    let mut samples = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        samples.push(v);
    }
    Ok(samples)
}
