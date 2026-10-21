// SPDX-License-Identifier: MIT

//! Integration tests for the chunked transcription pipeline.
//!
//! The tests that need a Whisper model live in `e2e_transcribe.rs`. The
//! ones here cover the paths that fail before a model is loaded: the
//! decoding of the input and the error reporting.

use std::path::Path;
use std::process::{Command, Output};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

/// Write a WAV file with a valid header and no samples.
fn write_empty_wav(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&36u32.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&16_000u32.to_le_bytes());
    bytes.extend_from_slice(&32_000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&0u32.to_le_bytes());
    std::fs::write(path, bytes).unwrap();
}

fn transcribe(input: &Path) -> Output {
    Command::new(bin())
        .args(["--no-interaction", "--no-progress", "transcribe"])
        .arg(input)
        .args(["--language", "en", "-o"])
        .arg(std::env::temp_dir().join("polyscribe-chunked-test-out"))
        .output()
        .expect("failed to spawn polyscribe")
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn empty_audio_is_reported_as_no_samples() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("empty.wav");
    write_empty_wav(&input);

    let output = transcribe(&input);
    assert!(
        !output.status.success(),
        "expected a non-zero exit for an empty recording"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("no audio samples"),
        "expected a no-samples error, got: {stderr}"
    );
}

#[test]
fn decode_failure_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("broken.wav");
    // A file with an audio extension that is not decodable.
    std::fs::write(&input, b"this is not audio").unwrap();

    let output = transcribe(&input);
    assert!(
        !output.status.success(),
        "expected a non-zero exit for an undecodable input"
    );
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("ffmpeg failed"),
        "expected an ffmpeg error, got: {stderr}"
    );
}

#[test]
fn a_missing_input_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("does-not-exist.wav");

    let output = transcribe(&input);
    assert!(!output.status.success(), "expected a non-zero exit");
    let stderr = stderr_of(&output);
    assert!(
        stderr.contains("Failed to open") || stderr.contains("ffmpeg failed"),
        "expected an input error, got: {stderr}"
    );
}
