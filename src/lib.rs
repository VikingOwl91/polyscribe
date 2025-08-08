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

use anyhow::{Context, Result, anyhow};
use chrono::Local;
use std::env;
use std::fs::create_dir_all;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    let ms = (total_ms % 1000) as i64;
    let total_secs = total_ms / 1000;
    let s = (total_secs % 60) as i64;
    let m = ((total_secs / 60) % 60) as i64;
    let h = (total_secs / 3600) as i64;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

/// Render a list of transcript entries to SRT format.
pub fn render_srt(items: &[OutputEntry]) -> String {
    let mut out = String::new();
    for (i, e) in items.iter().enumerate() {
        let idx = i + 1;
        out.push_str(&format!("{}\n", idx));
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
        eprintln!(
            "WARN: No Whisper model files (*.bin) found in {}.",
            models_dir.display()
        );
        eprint!("Would you like to download models now? [Y/n]: ");
        io::stderr().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let ans = input.trim().to_lowercase();
        if ans.is_empty() || ans == "y" || ans == "yes" {
            if let Err(e) = models::run_interactive_model_downloader() {
                eprintln!("ERROR: Downloader failed: {:#}", e);
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
            if p.is_file() {
                if candidates.iter().any(|c| c == &p) {
                    eprintln!("INFO: Using previously selected model: {}", p.display());
                    return Ok(p);
                }
            }
        }
    }

    eprintln!("Multiple Whisper models found in {}:", models_dir.display());
    for (i, p) in candidates.iter().enumerate() {
        eprintln!("  {}) {}", i + 1, p.display());
    }
    eprint!("Select model by number [1-{}]: ", candidates.len());
    io::stderr().flush().ok();
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
