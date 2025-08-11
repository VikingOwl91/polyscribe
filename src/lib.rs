// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

#![forbid(elided_lifetimes_in_paths)]
#![forbid(unused_must_use)]
#![deny(missing_docs)]
// Lint policy for incremental refactor toward 2024:
// - Keep basic clippy warnings enabled; skip pedantic/nursery for now (will revisit in step 7).
// - cargo lints can be re-enabled later once codebase is tidied.
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

/// Set quiet mode: when true, non-interactive logs should be suppressed.
pub fn set_quiet(q: bool) {
    QUIET.store(q, Ordering::Relaxed);
}
/// Return current quiet mode state.
pub fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// Set non-interactive mode: when true, interactive prompts must be skipped.
pub fn set_no_interaction(b: bool) {
    NO_INTERACTION.store(b, Ordering::Relaxed);
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

/// Check whether stdin is connected to a TTY. Used to avoid blocking prompts when not interactive.
pub fn stdin_is_tty() -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        unsafe { libc::isatty(std::io::stdin().as_raw_fd()) == 1 }
    }
    #[cfg(not(unix))]
    {
        // Best-effort on non-Unix: assume TTY when not redirected by common CI vars
        // This avoids introducing a new dependency for atty.
        !(std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok())
    }
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
            // Duplicate current stderr (fd 2)
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
            let dn = open(devnull_cstr.as_ptr(), O_WRONLY);
            if dn < 0 {
                // failed to open devnull; restore and bail
                close(old_fd);
                return Self {
                    active: false,
                    old_stderr_fd: -1,
                    devnull_fd: -1,
                };
            }
            // Redirect fd 2 to devnull
            if dup2(dn, 2) < 0 {
                close(dn);
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
                devnull_fd: dn,
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
            // Restore old stderr and close devnull and old copies
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
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = StderrSilencer::activate();
            f()
        }));
        match res {
            Ok(v) => v,
            Err(p) => std::panic::resume_unwind(p),
        }
    } else {
        f()
    }
}

/// Logging macros and helpers
/// Log an error to stderr (always printed). Recommended for user-visible errors.
#[macro_export]
macro_rules! elog {
    ($($arg:tt)*) => {{
        eprintln!("ERROR: {}", format!($($arg)*));
    }}
}
/// Internal helper macro used by other logging macros to centralize the
/// common behavior: build formatted message, check quiet/verbose flags,
/// and print to stderr with a label.
#[macro_export]
macro_rules! log_with_level {
    ($label:expr, $min_lvl:expr, $always:expr, $($arg:tt)*) => {{
        let should_print = if $always {
            true
        } else if let Some(minv) = $min_lvl {
            !$crate::is_quiet() && $crate::verbose_level() >= minv
        } else {
            !$crate::is_quiet()
        };
        if should_print {
            eprintln!("{}: {}", $label, format!($($arg)*));
        }
    }}
}

/// Log a warning to stderr (printed even in quiet mode).
#[macro_export]
macro_rules! wlog {
    ($($arg:tt)*) => {{ $crate::log_with_level!("WARN", None, true, $($arg)*); }}
}

/// Log an informational line to stderr unless quiet mode is enabled.
#[macro_export]
macro_rules! ilog {
    ($($arg:tt)*) => {{ $crate::log_with_level!("INFO", None, false, $($arg)*); }}
}

/// Log a debug/trace line when verbose level is at least the given level (u8).
#[macro_export]
macro_rules! dlog {
    ($lvl:expr, $($arg:tt)*) => {{
        $crate::log_with_level!(&format!("DEBUG{}", &$lvl), Some($lvl), false, $($arg)*);
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
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use libc::{O_WRONLY, close, dup, dup2, open};

/// Re-export backend module (GPU/CPU selection and transcription).
pub mod backend;
/// Re-export models module (model listing/downloading/updating).
pub mod models;
/// Progress and progress bar abstraction (TTY-aware, stderr-only)
pub mod progress;

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
    let s = total_secs % 60;
    let m = (total_secs / 60) % 60;
    let h = total_secs / 3600;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Render a list of transcript entries to SRT format.
pub fn render_srt(items: &[OutputEntry]) -> String {
    let mut out = String::new();
    for (i, e) in items.iter().enumerate() {
        let idx = i + 1;
        out.push_str(&format!("{idx}\n"));
        out.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(e.start),
            format_srt_time(e.end)
        ));
        if !e.speaker.is_empty() {
            out.push_str(&format!("{}: {}\n", e.speaker, e.text));
        } else {
            out.push_str(&format!("{}\n", e.text));
        }
        out.push('\n');
    }
    out
}

/// Determine the default models directory, honoring POLYSCRIBE_MODELS_DIR override.
pub fn models_dir_path() -> PathBuf {
    if let Ok(p) = env::var("POLYSCRIBE_MODELS_DIR") {
        let pb = PathBuf::from(p);
        if !pb.as_os_str().is_empty() {
            return pb;
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
    let mut s = input.trim().to_lowercase();
    if s.is_empty() || s == "auto" || s == "c" || s == "posix" {
        return None;
    }
    if let Some((lhs, _)) = s.split_once('.') {
        s = lhs.to_string();
    }
    if let Some((lhs, _)) = s.split_once('_') {
        s = lhs.to_string();
    }
    let code = match s.as_str() {
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
    // Silent model resolution used during processing to avoid interfering with progress bars.
    // Preflight prompting should be done by the caller before bars are created (use find_model_file_with_printer).
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
    // 1) Explicit environment override
    if let Ok(env_model) = env::var("WHISPER_MODEL") {
        let p = PathBuf::from(env_model);
        if p.is_file() {
            let _ = std::fs::write(models_dir.join(".last_model"), p.display().to_string());
            return Ok(p);
        }
    }
    // 2) Previously selected model
    let last_file = models_dir.join(".last_model");
    if let Ok(prev) = std::fs::read_to_string(&last_file) {
        let prev = prev.trim();
        if !prev.is_empty() {
            let p = PathBuf::from(prev);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    // 3) Best local model without prompting
    if let Some(local) = crate::models::pick_best_local_model(models_dir) {
        let _ = std::fs::write(models_dir.join(".last_model"), local.display().to_string());
        return Ok(local);
    }
    // 4) No model available; avoid interactive prompts here to prevent progress bar redraw issues.
    //    Callers should run find_model_file_with_printer(...) before starting progress bars to interactively select/download.
    Err(anyhow!(
        "No Whisper model available. Run with --download-models or ensure WHISPER_MODEL is set before processing."
    ))
}

/// Locate a Whisper model file, prompting user to download/select when necessary.
/// All prompts are printed using the provided printer closure (e.g., MultiProgress::println)
/// to avoid interfering with active progress bars.
pub fn find_model_file_with_printer<F>(printer: F) -> Result<PathBuf>
where
    F: Fn(&str),
{
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
        let p = PathBuf::from(env_model);
        if p.is_file() {
            let _ = std::fs::write(models_dir.join(".last_model"), p.display().to_string());
            return Ok(p);
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
    let rd = std::fs::read_dir(models_dir)
        .with_context(|| format!("Failed to read models directory: {}", models_dir.display()))?;
    for entry in rd {
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
        printer("Would you like to download models now? [Y/n]:");
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let ans = input.trim().to_lowercase();
        if ans.is_empty() || ans == "y" || ans == "yes" {
            if let Err(e) = models::run_interactive_model_downloader() {
                elog!("Downloader failed: {:#}", e);
            }
            candidates.clear();
            let rd2 = std::fs::read_dir(models_dir).with_context(|| {
                format!("Failed to read models directory: {}", models_dir.display())
            })?;
            for entry in rd2 {
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
        let only = candidates.remove(0);
        let _ = std::fs::write(models_dir.join(".last_model"), only.display().to_string());
        return Ok(only);
    }

    let last_file = models_dir.join(".last_model");
    if let Ok(prev) = std::fs::read_to_string(&last_file) {
        let prev = prev.trim();
        if !prev.is_empty() {
            let p = PathBuf::from(prev);
            if p.is_file() && candidates.iter().any(|c| c == &p) {
                // Previously printed: INFO about using previously selected model.
                // Suppress this to avoid duplicate/noisy messages; per-file progress will be shown elsewhere.
                return Ok(p);
            }
        }
    }

    printer(&"Multiple Whisper models found:".to_string());
    for (i, p) in candidates.iter().enumerate() {
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| p.display().to_string());
        printer(&format!("  {}) {}", i + 1, name));
    }
    // Print a blank line and the selection prompt using the provided printer to
    // keep output synchronized with any active progress rendering.
    printer("");
    printer(&format!("Select model by number [1-{}]:", candidates.len()));
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("Failed to read selection")?;
    let sel: usize = input
        .trim()
        .parse()
        .map_err(|_| anyhow!("Invalid selection: {}", input.trim()))?;
    if sel == 0 || sel > candidates.len() {
        return Err(anyhow!("Selection out of range"));
    }
    let chosen = candidates.swap_remove(sel - 1);
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
            return if e.kind() == std::io::ErrorKind::NotFound {
                Err(anyhow!(
                    "ffmpeg not found on PATH. Please install ffmpeg and ensure it is available."
                ))
            } else {
                Err(anyhow!(
                    "Failed to execute ffmpeg for {}: {}",
                    audio_path.display(),
                    e
                ))
            }
        }
    };
    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg failed for {}: {}",
            audio_path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let bytes = output.stdout;
    if bytes.len() % 4 != 0 {
        let truncated = bytes.len() - (bytes.len() % 4);
        let mut v = Vec::with_capacity(truncated / 4);
        for chunk in bytes[..truncated].chunks_exact(4) {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            v.push(f32::from_le_bytes(arr));
        }
        Ok(v)
    } else {
        let mut v = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            v.push(f32::from_le_bytes(arr));
        }
        Ok(v)
    }
}
