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

/// Run a closure while temporarily suppressing stderr on Unix.
/// Noisy native (whisper/ggml) logs are silenced unless super-verbose (-vv)
/// is enabled. The guard restores stderr even if the closure panics.
pub fn with_suppressed_stderr<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
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
pub mod chunked;
pub mod config;
pub mod error;
pub mod gpu;
pub mod models;
pub mod output;
pub mod stream;
pub mod ui;
pub use error::Error;
pub mod prelude;

pub use output::{
    OutputRoot, is_audio_file, is_json_file, load_json_transcript, sort_and_reindex,
    write_json_stdout, write_output_files,
};

#[derive(Debug, serde::Serialize, Clone, PartialEq)]
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

/// Canonical resolution of the transcription models directory.
///
/// Policy (identical in debug and release builds):
/// 1. `POLYSCRIBE_MODELS_DIR` when set to a non-empty value.
/// 2. `$XDG_DATA_HOME/polyscribe/models` when `XDG_DATA_HOME` is set.
/// 3. `~/.local/share/polyscribe/models` when `HOME` is set.
/// 4. `models` (relative to the working directory) as a last resort.
///
/// This is the single resolver behind model discovery (`find_model_file`)
/// and the interactive downloader (`ConfigService::models_dir`), so model
/// installation and transcription always agree on the directory. Models
/// installed by `models add` live in the ModelManager cache dir
/// (`$XDG_CACHE_HOME/polyscribe/models`) and are picked up by
/// `find_model_file` as a fallback.
pub fn models_dir_path() -> PathBuf {
    resolve_models_dir_with(|name| env::var(name).ok())
}

/// Pure core of [`models_dir_path`]: resolves the models directory from
/// environment lookups without touching the process environment, so tests
/// can drive it with a fixed variable map.
fn resolve_models_dir_with(get: impl Fn(&str) -> Option<String>) -> PathBuf {
    if let Some(env_val) = get("POLYSCRIBE_MODELS_DIR")
        && !env_val.is_empty()
    {
        return PathBuf::from(env_val);
    }
    if let Some(xdg) = get("XDG_DATA_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("polyscribe").join("models");
    }
    if let Some(home) = get("HOME")
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
    let whisper_model = env::var("WHISPER_MODEL").ok().map(PathBuf::from);
    let models_dir = models_dir_path();
    let cache_dir = crate::model_manager::Paths::resolve()
        .ok()
        .map(|paths| paths.cache_dir);
    Ok(find_model_file_in(
        whisper_model.as_deref(),
        &models_dir,
        cache_dir.as_deref(),
    )?)
}

/// Model discovery with explicit inputs, so tests do not have to touch the
/// process environment.
///
/// `whisper_model` is the `WHISPER_MODEL` override (must point at an existing
/// file), `models_dir` the primary transcription models dir, and `cache_dir`
/// the optional ModelManager cache dir used as a fallback so models installed
/// via `polyscribe models add` are found.
fn find_model_file_in(
    whisper_model: Option<&Path>,
    models_dir: &Path,
    cache_dir: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    if let Some(p) = whisper_model {
        if !p.exists() {
            return Err(anyhow!(
                "WHISPER_MODEL points to a non-existing path: {}",
                p.display()
            ));
        }
        if !p.is_file() {
            return Err(anyhow!(
                "WHISPER_MODEL must point to a file, but is not: {}",
                p.display()
            ));
        }
        return Ok(p.to_path_buf());
    }

    if models_dir.exists() && !models_dir.is_dir() {
        return Err(anyhow!(
            "Models path exists but is not a directory: {}",
            models_dir.display()
        ));
    }
    std::fs::create_dir_all(models_dir).with_context(|| {
        format!(
            "Failed to ensure models dir exists: {}",
            models_dir.display()
        )
    })?;

    let mut candidates = scan_model_dir(models_dir)
        .with_context(|| format!("Failed to read models dir: {}", models_dir.display()))?;

    // Fall back to the ModelManager cache dir so models installed via
    // `polyscribe models add` are found by transcribe. A missing cache dir
    // simply holds no models and must not mask the actionable "no model"
    // message; genuine I/O failures on an existing dir still propagate.
    if candidates.is_empty()
        && let Some(cache_dir) = cache_dir
        && cache_dir != models_dir
    {
        match scan_model_dir(cache_dir) {
            Ok(found) => candidates = found,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(e).context(format!(
                    "Failed to read models dir: {}",
                    cache_dir.display()
                ));
            }
        }
    }

    if candidates.is_empty() {
        return Err(anyhow!(
            "No Whisper model files (*.bin) found in {}. \
             Please download a model (polyscribe models add) or set WHISPER_MODEL.",
            models_dir.display()
        ));
    }

    candidates.sort_by_key(|(size, _)| *size);
    let (_size, path) = candidates.into_iter().last().expect("non-empty");
    Ok(path)
}

/// List `*.bin` model files in a directory as (size, path) pairs.
fn scan_model_dir(dir: &Path) -> std::io::Result<Vec<(u64, PathBuf)>> {
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(dir)? {
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
    Ok(candidates)
}

/// Decode an input media file to 16kHz mono f32 PCM using ffmpeg available on PATH.
///
/// Samples are streamed from ffmpeg's stdout as they are produced, so peak
/// memory stays close to the PCM buffer itself instead of additionally holding
/// the raw byte stream, and no temp file is used (a fixed temp file name would
/// collide between concurrent runs). A trailing partial sample (if any) is dropped.
///
/// This holds the whole recording in memory, so it scales with the duration
/// of the input. The transcriber does not use it: it decodes the audio in
/// windows with [`chunked::ChunkedAudio`], which keeps the memory of a
/// transcription independent of the length of the recording.
pub fn decode_audio_to_pcm_f32_ffmpeg(audio_path: &Path) -> Result<Vec<f32>> {
    use std::io::Read as _;

    let mut child = crate::chunked::spawn_ffmpeg_decoder(audio_path)?;

    let stdout = child.stdout.take().expect("ffmpeg stdout was piped");
    let mut stderr = child.stderr.take().expect("ffmpeg stderr was piped");
    // Drain stderr on a separate thread so ffmpeg can never deadlock on a full pipe.
    let stderr_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });

    let mut pcm = Vec::new();
    let mut reader = std::io::BufReader::new(stdout);
    let mut sample = [0u8; 4];
    loop {
        match reader.read_exact(&mut sample) {
            Ok(()) => pcm.push(f32::from_le_bytes(sample)),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!(
                    "Failed to read ffmpeg output for {}: {}",
                    audio_path.display(),
                    e
                )
                .into());
            }
        }
    }

    let status = child.wait().context("Failed to wait for ffmpeg")?;
    let stderr_bytes = stderr_reader.join().unwrap_or_default();
    if !status.success() {
        return Err(anyhow!(
            "ffmpeg failed for {}: {}",
            audio_path.display(),
            String::from_utf8_lossy(&stderr_bytes)
        )
        .into());
    }
    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_srt_time_basic_and_rounding() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.0), "00:00:01,000");
        assert_eq!(format_srt_time(61.0), "00:01:01,000");
        assert_eq!(format_srt_time(3661.789), "01:01:01,789");
        // rounding
        assert_eq!(format_srt_time(0.0014), "00:00:00,001");
        assert_eq!(format_srt_time(0.0015), "00:00:00,002");
    }

    #[test]
    fn test_render_srt_with_and_without_speaker() {
        let items = vec![
            OutputEntry {
                id: 0,
                speaker: "Alice".to_string(),
                start: 0.0,
                end: 1.0,
                text: "Hello".to_string(),
            },
            OutputEntry {
                id: 1,
                speaker: String::new(),
                start: 1.0,
                end: 2.0,
                text: "World".to_string(),
            },
        ];
        let srt = render_srt(&items);
        let expected = "1\n00:00:00,000 --> 00:00:01,000\nAlice: Hello\n\n2\n00:00:01,000 --> 00:00:02,000\nWorld\n\n";
        assert_eq!(srt, expected);
    }

    #[test]
    fn test_date_prefix_format_shape() {
        let d = date_prefix();
        assert_eq!(d.len(), 10);
        let bytes = d.as_bytes();
        assert!(
            bytes[0].is_ascii_digit()
                && bytes[1].is_ascii_digit()
                && bytes[2].is_ascii_digit()
                && bytes[3].is_ascii_digit()
        );
        assert_eq!(bytes[4], b'-');
        assert!(bytes[5].is_ascii_digit() && bytes[6].is_ascii_digit());
        assert_eq!(bytes[7], b'-');
        assert!(bytes[8].is_ascii_digit() && bytes[9].is_ascii_digit());
    }

    #[test]
    fn test_normalize_lang_code() {
        assert_eq!(normalize_lang_code("en"), Some("en".to_string()));
        assert_eq!(normalize_lang_code("German"), Some("de".to_string()));
        assert_eq!(normalize_lang_code("en_US.UTF-8"), Some("en".to_string()));
        assert_eq!(normalize_lang_code("AUTO"), None);
        assert_eq!(normalize_lang_code("  \t  "), None);
        assert_eq!(normalize_lang_code("zh"), Some("zh".to_string()));
    }

    // --- model discovery (find_model_file_in) ---

    fn write_file(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// Assert the actionable "no model installed" error (audit F-01): it must
    /// name the models dir, hint at `models add` / `WHISPER_MODEL`, and never
    /// surface as a raw filesystem error.
    fn assert_actionable_no_model_error(err: anyhow::Error, models_dir: &Path) {
        let msg = format!("{err:#}");
        assert!(
            msg.contains("No Whisper model files (*.bin) found"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains(models_dir.to_str().unwrap()),
            "error should name the models dir: {msg}"
        );
        assert!(msg.contains("polyscribe models add"), "missing hint: {msg}");
        assert!(msg.contains("WHISPER_MODEL"), "missing hint: {msg}");
        assert!(
            !msg.contains("Failed to read models dir"),
            "filesystem error must not mask the actionable message: {msg}"
        );
    }

    #[test]
    fn test_find_model_file_empty_models_dir_no_cache_dir_is_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();

        let err = find_model_file_in(None, &models_dir, None).unwrap_err();
        assert_actionable_no_model_error(err, &models_dir);
    }

    #[test]
    fn test_find_model_file_missing_cache_dir_does_not_mask_actionable_error() {
        // F-01 regression: the ModelManager cache-dir fallback used to
        // propagate the read_dir error when the cache dir did not exist.
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        let missing_cache = tmp.path().join("missing-cache");

        let err = find_model_file_in(None, &models_dir, Some(&missing_cache)).unwrap_err();
        assert_actionable_no_model_error(err, &models_dir);
    }

    #[test]
    fn test_find_model_file_empty_cache_dir_is_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        let cache_dir = tmp.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        let err = find_model_file_in(None, &models_dir, Some(&cache_dir)).unwrap_err();
        assert_actionable_no_model_error(err, &models_dir);
    }

    #[test]
    fn test_find_model_file_falls_back_to_cache_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        let cache_dir = tmp.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let model = cache_dir.join("ggml-tiny.en.bin");
        write_file(&model, b"fake model bytes");
        // Non-.bin files are ignored.
        write_file(&cache_dir.join("notes.txt"), b"not a model");

        let found = find_model_file_in(None, &models_dir, Some(&cache_dir)).unwrap();
        assert_eq!(found, model);
    }

    #[test]
    fn test_find_model_file_picks_largest_bin_in_models_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        write_file(&models_dir.join("small.bin"), &[0u8; 8]);
        let large = models_dir.join("large.bin");
        write_file(&large, &[0u8; 64]);
        write_file(&models_dir.join("ignore.txt"), b"not a model");
        // A subdirectory named *.bin is not a candidate.
        std::fs::create_dir_all(models_dir.join("dir.bin")).unwrap();

        let found = find_model_file_in(None, &models_dir, None).unwrap();
        assert_eq!(found, large);
    }

    #[test]
    fn test_find_model_file_creates_missing_models_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("fresh-models");

        let err = find_model_file_in(None, &models_dir, None).unwrap_err();
        assert!(models_dir.is_dir(), "models dir should be created");
        assert_actionable_no_model_error(err, &models_dir);
    }

    #[test]
    fn test_find_model_file_models_dir_must_be_a_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("not-a-dir");
        write_file(&file_path, b"x");

        let err = find_model_file_in(None, &file_path, None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("exists but is not a directory"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_find_model_file_whisper_model_missing_path() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope.bin");

        let err = find_model_file_in(Some(&missing), &tmp.path().join("models"), None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("WHISPER_MODEL points to a non-existing path"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("nope.bin"),
            "error should name the path: {msg}"
        );
    }

    #[test]
    fn test_find_model_file_whisper_model_must_be_a_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("model-dir");
        std::fs::create_dir_all(&dir).unwrap();

        let err = find_model_file_in(Some(&dir), &tmp.path().join("models"), None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("WHISPER_MODEL must point to a file"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_find_model_file_whisper_model_override_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("custom.bin");
        write_file(&model, b"fake model bytes");

        // The explicit override is used even when no models dir exists.
        let found = find_model_file_in(Some(&model), &tmp.path().join("missing"), None).unwrap();
        assert_eq!(found, model);
    }

    // --- models dir resolution (models_dir_path / resolve_models_dir_with) ---

    /// Build an environment lookup over a fixed variable map.
    fn lookup<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name: &str| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn test_models_dir_resolution_policy() {
        // An explicit override wins over everything.
        assert_eq!(
            resolve_models_dir_with(lookup(&[
                ("POLYSCRIBE_MODELS_DIR", "/opt/models"),
                ("XDG_DATA_HOME", "/xdg"),
                ("HOME", "/home/u"),
            ])),
            PathBuf::from("/opt/models")
        );
        // An empty override is ignored.
        assert_eq!(
            resolve_models_dir_with(lookup(&[
                ("POLYSCRIBE_MODELS_DIR", ""),
                ("XDG_DATA_HOME", "/xdg"),
            ])),
            PathBuf::from("/xdg/polyscribe/models")
        );
        // XDG_DATA_HOME follows the XDG convention.
        assert_eq!(
            resolve_models_dir_with(lookup(&[("XDG_DATA_HOME", "/xdg")])),
            PathBuf::from("/xdg/polyscribe/models")
        );
        // HOME falls back to the XDG default location.
        assert_eq!(
            resolve_models_dir_with(lookup(&[("HOME", "/home/u")])),
            PathBuf::from("/home/u/.local/share/polyscribe/models")
        );
        // Last resort: a relative dir when no HOME is available.
        assert_eq!(
            resolve_models_dir_with(lookup(&[])),
            PathBuf::from("models")
        );
    }

    #[test]
    fn test_models_dir_resolution_has_no_debug_special_case() {
        // F-02 regression guard: cargo test runs with debug assertions
        // enabled, so if the resolver ever special-cases debug builds (the
        // old "./models" default), these assertions fail.
        assert_eq!(
            resolve_models_dir_with(lookup(&[("HOME", "/home/u")])),
            PathBuf::from("/home/u/.local/share/polyscribe/models"),
            "default must be the XDG data dir, not a debug-only relative dir"
        );
        assert_eq!(
            resolve_models_dir_with(lookup(&[("XDG_DATA_HOME", "/xdg"), ("HOME", "/home/u")])),
            PathBuf::from("/xdg/polyscribe/models")
        );
    }

    #[test]
    fn test_models_dir_path_matches_resolver_policy() {
        // The live env-driven entry point must agree with the pure resolver
        // (and therefore with ConfigService::models_dir).
        assert_eq!(
            models_dir_path(),
            resolve_models_dir_with(|name| env::var(name).ok())
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_find_model_file_unreadable_cache_dir_error_is_preserved() {
        use std::os::unix::fs::PermissionsExt;

        // Root bypasses file permission checks, so the assertion is skipped.
        if unsafe { libc::geteuid() } == 0 {
            eprintln!("SKIPPED: running as root, permission bits are not enforced");
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let models_dir = tmp.path().join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        let cache_dir = tmp.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::set_permissions(&cache_dir, std::fs::Permissions::from_mode(0o000)).unwrap();

        let result = find_model_file_in(None, &models_dir, Some(&cache_dir));
        // Restore permissions so the tempdir can be cleaned up.
        std::fs::set_permissions(&cache_dir, std::fs::Permissions::from_mode(0o755)).unwrap();

        let err = result.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Failed to read models dir"),
            "genuine I/O failures must stay visible: {msg}"
        );
        assert!(
            msg.contains(cache_dir.to_str().unwrap()),
            "error should name the cache dir: {msg}"
        );
    }
}
