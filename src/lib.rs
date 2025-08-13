// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

#![forbid(elided_lifetimes_in_paths)]
#![forbid(unused_must_use)]
#![deny(missing_docs)]
#![warn(clippy::all)]
//! PolyScribe library: business logic and core types.
//!
//! This crate exposes the reusable parts of the PolyScribe CLI as a library.
//! The binary entry point (main.rs) remains a thin CLI wrapper.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// Global runtime flags
static QUIET: AtomicBool = AtomicBool::new(false);
static NO_INTERACTION: AtomicBool = AtomicBool::new(false);
static VERBOSE: AtomicU8 = AtomicU8::new(0);
static NO_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Set quiet mode: when true, non-interactive logs should be suppressed.
pub fn set_quiet(enabled: bool) {
    QUIET.store(enabled, Ordering::Relaxed);
}
/// Return current quiet mode state.
pub fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// Set non-interactive mode: when true, interactive prompts must be skipped.
pub fn set_no_interaction(enabled: bool) {
    NO_INTERACTION.store(enabled, Ordering::Relaxed);
}
/// Return current non-interactive state.
pub fn is_no_interaction() -> bool {
    NO_INTERACTION.load(Ordering::Relaxed)
}

/// Set verbose level (0 = normal, 1 = verbose, 2 = super-verbose)
pub fn set_verbose(level: u8) {
    VERBOSE.store(level, Ordering::Relaxed);
}
/// Get current verbose level.
pub fn verbose_level() -> u8 {
    VERBOSE.load(Ordering::Relaxed)
}

/// Disable interactive progress indicators (bars/spinners)
pub fn set_no_progress(enabled: bool) {
    NO_PROGRESS.store(enabled, Ordering::Relaxed);
}
/// Return current no-progress state
pub fn is_no_progress() -> bool {
    NO_PROGRESS.load(Ordering::Relaxed)
}

/// Check whether stdin is connected to a TTY. Used to avoid blocking prompts when not interactive.
pub fn stdin_is_tty() -> bool {
    use std::io::IsTerminal as _;
    std::io::stdin().is_terminal()
}

/// A guard that temporarily redirects stderr to /dev/null on Unix when quiet mode is active.
/// No-op on non-Unix or when quiet is disabled. Restores stderr on drop.
pub struct StderrSilencer {
    #[cfg(unix)]
    old_stderr_fd: i32,
    #[cfg(unix)]
    devnull_fd: i32,
    active: bool,
}

impl StderrSilencer {
    /// Activate stderr silencing if quiet is set and on Unix; otherwise returns a no-op guard.
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

    /// Activate stderr silencing unconditionally (used internally); no-op on non-Unix.
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
            // Open /dev/null for writing
            let devnull_cstr = std::ffi::CString::new("/dev/null").unwrap();
            let devnull_fd = open(devnull_cstr.as_ptr(), O_WRONLY);
            if devnull_fd < 0 {
                close(old_fd);
                return Self {
                    active: false,
                    old_stderr_fd: -1,
                    devnull_fd: -1,
                };
            }
            if dup2(devnull_fd, 2) < 0 {
                close(devnull_fd);
                close(old_fd);
                return Self {
                    active: false,
                    old_stderr_fd: -1,
                    devnull_fd: -1,
                };
            }
            Self {
                active: true,
                old_stderr_fd: old_fd,
                devnull_fd: devnull_fd,
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
            let _ = close(self.devnull_fd);
            let _ = close(self.old_stderr_fd);
        }
        self.active = false;
    }
}

/// Run a closure while temporarily suppressing stderr on Unix when appropriate.
/// On Windows/non-Unix, this is a no-op wrapper.
/// This helper uses RAII + panic catching to ensure restoration before resuming panic.
pub fn with_suppressed_stderr<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    // Suppress noisy native logs unless super-verbose (-vv) is enabled.
    if verbose_level() < 2 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = StderrSilencer::activate();
            f()
        }));
        match result {
            Ok(value) => value,
            Err(panic_payload) => std::panic::resume_unwind(panic_payload),
        }
    } else {
        f()
    }
}

/// Centralized UI helpers (TTY-aware, quiet/verbose-aware)
pub mod ui;

/// Logging macros and helpers
/// Log an error using the UI helper (always printed). Recommended for user-visible errors.
#[macro_export]
macro_rules! elog {
    ($($arg:tt)*) => {{
        $crate::ui::error(format!($($arg)*));
    }}
}

/// Log a warning using the UI helper (printed even in quiet mode).
#[macro_export]
macro_rules! wlog {
    ($($arg:tt)*) => {{
        $crate::ui::warn(format!($($arg)*));
    }}
}

/// Log an informational line using the UI helper unless quiet mode is enabled.
#[macro_export]
macro_rules! ilog {
    ($($arg:tt)*) => {{
        if !$crate::is_quiet() { $crate::ui::info(format!($($arg)*)); }
    }}
}

/// Log a debug/trace line when verbose level is at least the given level (u8).
#[macro_export]
macro_rules! dlog {
    ($lvl:expr, $($arg:tt)*) => {{
        if !$crate::is_quiet() && $crate::verbose_level() >= $lvl { $crate::ui::info(format!("DEBUG{}: {}", $lvl, format!($($arg)*))); }
    }}
}

/// Backward-compatibility: map old qlog! to ilog!
#[macro_export]
macro_rules! qlog {
    ($($arg:tt)*) => {{ $crate::ilog!($($arg)*); }}
}

use anyhow::{Context, Result, anyhow};
use chrono::Local;
use std::env;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use libc::{O_WRONLY, close, dup, dup2, open};

/// Re-export backend module (GPU/CPU selection and transcription).
pub mod backend;
/// Re-export models module (model listing/downloading/updating).
pub mod models;

/// Transcript entry for a single segment.
#[derive(Debug, serde::Serialize, Clone)]
pub struct OutputEntry {
    /// Sequential id in output ordering.
    pub id: u64,
    /// Speaker label associated with the segment.
    pub speaker: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Text content.
    pub text: String,
}

/// Return a YYYY-MM-DD date prefix string for output file naming.
pub fn date_prefix() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

/// Format a floating-point number of seconds as SRT timestamp (HH:MM:SS,mmm).
pub fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as i64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let sec = total_secs % 60;
    let min = (total_secs / 60) % 60;
    let hour = total_secs / 3600;
    format!("{hour:02}:{min:02}:{sec:02},{ms:03}")
}

/// Render a list of transcript entries to SRT format.
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

/// Determine the default models directory, honoring POLYSCRIBE_MODELS_DIR override.
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
    if let Ok(xdg) = env::var("XDG_DATA_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("polyscribe").join("models");
        }
    }
    if let Ok(home) = env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("polyscribe")
                .join("models");
        }
    }
    PathBuf::from("models")
}

/// Normalize a language identifier to a short ISO code when possible.
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

/// Locate a Whisper model file, prompting user to download/select when necessary.
pub fn find_model_file() -> Result<PathBuf> {
    let models_dir_buf = models_dir_path();
    let models_dir = models_dir_buf.as_path();
    if !models_dir.exists() {
        create_dir_all(models_dir).with_context(|| {
            format!(
                "Failed to create models directory: {}",
                models_dir.display()
            )
        })?;
    }

    if let Ok(env_model) = env::var("WHISPER_MODEL") {
        let model_path = PathBuf::from(env_model);
        if model_path.is_file() {
            let _ = std::fs::write(models_dir.join(".last_model"), model_path.display().to_string());
            return Ok(model_path);
        }
    }

    // Non-interactive mode: automatic selection and optional download
    if crate::is_no_interaction() {
        if let Some(local) = crate::models::pick_best_local_model(models_dir) {
            let _ = std::fs::write(models_dir.join(".last_model"), local.display().to_string());
            return Ok(local);
        } else {
            ilog!("No local models found; downloading large-v3-turbo-q8_0...");
            let path = crate::models::ensure_model_available_noninteractive("large-v3-turbo-q8_0")
                .with_context(|| "Failed to download required model 'large-v3-turbo-q8_0'")?;
            let _ = std::fs::write(models_dir.join(".last_model"), path.display().to_string());
            return Ok(path);
        }
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    let dir_entries = std::fs::read_dir(models_dir)
        .with_context(|| format!("Failed to read models directory: {}", models_dir.display()))?;
    for entry in dir_entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_lowercase())
            {
                if ext == "bin" {
                    candidates.push(path);
                }
            }
        }
    }

    if candidates.is_empty() {
        // No models found: prompt interactively (TTY only)
        wlog!(
            "{}",
            format!(
                "No Whisper model files (*.bin) found in {}.",
                models_dir.display()
            )
        );
        if crate::is_no_interaction() || !crate::stdin_is_tty() {
            return Err(anyhow!(
                "No models available and interactive mode is disabled. Please set WHISPER_MODEL or run with --download-models."
            ));
        }
        let input = crate::ui::prompt_line("Would you like to download models now? [Y/n]: ").unwrap_or_default();
        let answer = input.trim().to_lowercase();
        if answer.is_empty() || answer == "y" || answer == "yes" {
            if let Err(e) = models::run_interactive_model_downloader() {
                elog!("Downloader failed: {:#}", e);
            }
            candidates.clear();
            let dir_entries2 = std::fs::read_dir(models_dir).with_context(|| {
                format!("Failed to read models directory: {}", models_dir.display())
            })?;
            for entry in dir_entries2 {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_lowercase())
                    {
                        if ext == "bin" {
                            candidates.push(path);
                        }
                    }
                }
            }
        }
    }

    if candidates.is_empty() {
        return Err(anyhow!(
            "No Whisper model files (*.bin) available in {}",
            models_dir.display()
        ));
    }

    if candidates.len() == 1 {
        let only_model = candidates.remove(0);
        let _ = std::fs::write(models_dir.join(".last_model"), only_model.display().to_string());
        return Ok(only_model);
    }

    let last_file = models_dir.join(".last_model");
    if let Ok(previous_content) = std::fs::read_to_string(&last_file) {
        let previous_content = previous_content.trim();
        if !previous_content.is_empty() {
            let previous_path = PathBuf::from(previous_content);
            if previous_path.is_file() && candidates.iter().any(|c| c == &previous_path) {
                return Ok(previous_path);
            }
        }
    }

    crate::ui::println_above_bars(format!("Multiple Whisper models found in {}:", models_dir.display()));
    for (index, path) in candidates.iter().enumerate() {
        crate::ui::println_above_bars(format!("  {}) {}", index + 1, path.display()));
    }
    let input = crate::ui::prompt_line(&format!("Select model by number [1-{}]: ", candidates.len()))
        .map_err(|_| anyhow!("Failed to read selection"))?;
    let selection: usize = input
        .trim()
        .parse()
        .map_err(|_| anyhow!("Invalid selection: {}", input.trim()))?;
    if selection == 0 || selection > candidates.len() {
        return Err(anyhow!("Selection out of range"));
    }
    let chosen = candidates.swap_remove(selection - 1);
    let _ = std::fs::write(models_dir.join(".last_model"), chosen.display().to_string());
    Ok(chosen)
}

/// Decode an input media file to 16kHz mono f32 PCM using ffmpeg available on PATH.
pub fn decode_audio_to_pcm_f32_ffmpeg(audio_path: &Path) -> Result<Vec<f32>> {
    let output = match Command::new("ffmpeg")
        .arg("-i")
        .arg(audio_path)
        .arg("-f")
        .arg("f32le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("pipe:1")
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                return Err(anyhow!(
                    "ffmpeg not found on PATH. Please install ffmpeg and ensure it is available."
                ));
            } else {
                return Err(anyhow!(
                    "Failed to execute ffmpeg for {}: {}",
                    audio_path.display(),
                    e
                ));
            }
        }
    };
    if !output.status.success() {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "Failed to decode audio from {} using ffmpeg. This may indicate the file is not a valid or supported audio/video file, is corrupted, or cannot be opened. ffmpeg stderr: {}",
            audio_path.display(),
            stderr_str.trim()
        ));
    }
    let data = output.stdout;
    if data.len() % 4 != 0 {
        let truncated = data.len() - (data.len() % 4);
        let mut samples = Vec::with_capacity(truncated / 4);
        for chunk in data[..truncated].chunks_exact(4) {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            samples.push(f32::from_le_bytes(arr));
        }
        Ok(samples)
    } else {
        let mut samples = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            samples.push(f32::from_le_bytes(arr));
        }
        Ok(samples)
    }
}
