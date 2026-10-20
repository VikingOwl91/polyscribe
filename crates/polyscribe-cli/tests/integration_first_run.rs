// SPDX-License-Identifier: MIT

//! First-run integration tests for model discovery error paths.
//!
//! These tests run the built binary with a fully isolated environment (an
//! empty models dir, a cache dir that does not exist, and no WHISPER_MODEL
//! override) and assert the actionable "no model installed" message instead
//! of an internal filesystem error (audit F-01). Each spawned CLI process
//! gets its own environment, so no test mutates shared process state.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

/// Write a minimal valid 16 kHz mono s16 WAV file (silence) so the CLI
/// passes input validation and ffmpeg decoding and reaches model discovery.
fn write_silent_wav(path: &Path, seconds: f32) {
    let sample_rate: u32 = 16_000;
    let n_samples = (seconds * sample_rate as f32) as u32;
    let data_len = n_samples * 2;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    bytes.resize(bytes.len() + data_len as usize, 0);
    fs::write(path, bytes).unwrap();
}

/// True when ffmpeg is available on PATH. The first-run scenario needs the
/// CLI to get past ffmpeg decoding to reach model discovery.
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// An isolated first-run environment: empty models dir, missing cache dir,
/// empty HOME, and a valid (silent) audio input.
struct FirstRunEnv {
    // Kept alive for the duration of the test.
    _tmp: tempfile::TempDir,
    models_dir: PathBuf,
    cache_root: PathBuf,
    home: PathBuf,
    audio: PathBuf,
}

/// Build the isolated environment, or `None` (with a note) when ffmpeg is
/// not on PATH. This matches the skip convention of the other
/// ffmpeg-dependent tests; CI installs ffmpeg so the tests run there.
fn setup() -> Option<FirstRunEnv> {
    if !ffmpeg_available() {
        eprintln!(
            "NOT RUN: ffmpeg is not available on PATH; skipping first-run model discovery test"
        );
        return None;
    }
    let tmp = tempfile::tempdir().unwrap();
    let models_dir = tmp.path().join("models");
    fs::create_dir_all(&models_dir).unwrap();
    // POLYSCRIBE_CACHE_DIR is set to this root; the ModelManager cache dir
    // is `<root>/models`, which deliberately does not exist (F-01 scenario).
    let cache_root = tmp.path().join("cache");
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).unwrap();
    let audio = tmp.path().join("input.wav");
    write_silent_wav(&audio, 0.25);
    Some(FirstRunEnv {
        _tmp: tmp,
        models_dir,
        cache_root,
        home,
        audio,
    })
}

impl FirstRunEnv {
    /// A CLI invocation with a fully controlled environment.
    fn command(&self) -> Command {
        let mut cmd = Command::new(bin());
        cmd.env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", &self.home)
            .env("POLYSCRIBE_MODELS_DIR", &self.models_dir)
            .env("POLYSCRIBE_CACHE_DIR", &self.cache_root)
            // Both XDG locations are missing on purpose.
            .env("XDG_CACHE_HOME", self.home.join("xdg-cache"))
            .env("XDG_DATA_HOME", self.home.join("xdg-data"));
        cmd
    }

    fn transcribe(&self) -> Command {
        let mut cmd = self.command();
        cmd.arg("transcribe")
            .arg(&self.audio)
            .args([OsStr::new("--language"), OsStr::new("en")]);
        cmd
    }
}

fn stderr_string(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr not UTF-8")
}

#[test]
fn first_run_without_model_shows_actionable_message() {
    let Some(env) = setup() else {
        return;
    };
    let output = env
        .transcribe()
        .output()
        .expect("failed to spawn polyscribe");

    assert!(
        !output.status.success(),
        "expected a non-zero exit when no model is installed"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("No Whisper model files (*.bin) found"),
        "expected the actionable no-model message, got: {stderr}"
    );
    assert!(
        stderr.contains("polyscribe models add"),
        "expected an install hint, got: {stderr}"
    );
    assert!(
        stderr.contains("WHISPER_MODEL"),
        "expected a WHISPER_MODEL hint, got: {stderr}"
    );
    assert!(
        !stderr.contains("Failed to read models dir"),
        "the missing cache dir must not surface as a filesystem error: {stderr}"
    );
}

#[test]
fn first_run_with_missing_whisper_model_path_is_actionable() {
    let Some(env) = setup() else {
        return;
    };
    let output = env
        .transcribe()
        .env("WHISPER_MODEL", env.home.join("missing-model.bin"))
        .output()
        .expect("failed to spawn polyscribe");

    assert!(!output.status.success(), "expected a non-zero exit");
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("WHISPER_MODEL points to a non-existing path"),
        "unexpected stderr: {stderr}"
    );
}

#[test]
fn first_run_with_whisper_model_pointing_at_directory_is_actionable() {
    let Some(env) = setup() else {
        return;
    };
    let output = env
        .transcribe()
        .env("WHISPER_MODEL", &env.models_dir)
        .output()
        .expect("failed to spawn polyscribe");

    assert!(!output.status.success(), "expected a non-zero exit");
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("WHISPER_MODEL must point to a file"),
        "unexpected stderr: {stderr}"
    );
}
