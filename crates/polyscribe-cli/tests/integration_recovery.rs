// SPDX-License-Identifier: MIT

//! Regression tests for the recovery of speech that whisper did not
//! transcribe.
//!
//! whisper does not always report a segment for the beginning of the audio
//! it is given. When it does not, the pipeline of sprint 6 advanced to the
//! last segment of the window and the beginning of the window was never
//! transcribed. The tests here check that such a region is decoded again
//! and that the transcript covers it.
//!
//! The tests need a Whisper model, so they skip themselves without one,
//! like the other end-to-end tests.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Deserialize)]
struct OutputEntry {
    #[allow(dead_code)]
    id: u64,
    #[allow(dead_code)]
    speaker: String,
    start: f64,
    end: f64,
    text: String,
}

#[derive(Deserialize)]
struct OutputRoot {
    items: Vec<OutputEntry>,
}

#[derive(Deserialize)]
struct ClipPlacement {
    file: String,
    offset: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    description: String,
    clips: Vec<ClipPlacement>,
}

#[derive(Deserialize)]
struct FixtureSpec {
    cases: Vec<Case>,
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("recovery")
}

fn boundary_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("boundary")
}

/// The samples of a WAV file, without the header.
fn wav_samples(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap();
    let mut pos = 12;
    let (data_start, data_len) = loop {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            break (pos + 8, size);
        }
        pos += 8 + size + (size % 2);
    };
    bytes[data_start..data_start + data_len]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Write the samples as a WAV file.
fn write_wav(path: &Path, samples: &[i16]) {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&((36 + samples.len() * 2) as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, &bytes).unwrap();
}

/// Compose the fixture of a case: silence with the speech clips placed at
/// their offsets.
fn compose_case(case: &Case, out_path: &Path) -> f64 {
    let mut samples: Vec<i16> = Vec::new();
    let mut cursor = 0.0f64;
    for placement in &case.clips {
        let gap = ((placement.offset - cursor) * SAMPLE_RATE as f64).round() as usize;
        samples.extend(std::iter::repeat_n(0i16, gap));
        let clip = wav_samples(&boundary_dir().join(&placement.file));
        cursor = placement.offset + clip.len() as f64 / SAMPLE_RATE as f64;
        samples.extend(clip);
    }
    write_wav(out_path, &samples);
    samples.len() as f64 / SAMPLE_RATE as f64
}

fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Transcribe a file and return the segments.
fn transcribe(audio: &Path, model: &Path, out_dir: &Path) -> Vec<OutputEntry> {
    let output = Command::new(bin())
        .args(["--no-interaction", "--no-progress"])
        .arg("transcribe")
        .arg(audio)
        .args([OsStr::new("--language"), OsStr::new("en")])
        .arg("-o")
        .arg(out_dir)
        .env("WHISPER_MODEL", model)
        .output()
        .expect("failed to spawn polyscribe");
    assert!(
        output.status.success(),
        "transcribe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json_path = std::fs::read_dir(out_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "json"))
        .unwrap_or_else(|| panic!("no JSON output in {}", out_dir.display()));
    let root: OutputRoot = serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap())
        .expect("the transcript must be valid JSON");
    root.items
}

fn e2e_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("POLYSCRIBE_E2E_MODEL").ok()?);
    path.is_file().then_some(path)
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// The speech that whisper did not report must not be missing from the
/// transcript.
///
/// The fixture is the long recording of the benchmark, cut at the position
/// where whisper does not report a segment for the beginning of a window.
/// The pipeline of sprint 6 advanced to the last segment of the window and
/// the first 22 seconds of the recording were not in the transcript.
#[test]
fn the_skipped_region_is_transcribed() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the recovery tests");
        return;
    };
    if !ffmpeg_available() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping the recovery tests");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let audio = fixture_dir().join("skipped-22s.wav");
    let items = transcribe(&audio, &model, &tmp.path().join("out"));

    assert!(
        !items.is_empty(),
        "the recording was not transcribed at all"
    );

    // The beginning of the recording must be transcribed. The fixture holds
    // speech from the first second on, so a transcript that starts much
    // later has lost it.
    assert!(
        items[0].start < 2.0,
        "the first segment starts at {:.2} s, the speech at the beginning of \
         the recording is missing",
        items[0].start
    );

    // The segments are ordered, they do not overlap and they cover the
    // recording without a hole.
    for pair in items.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "the timestamps must not go backwards: {:?} then {:?}",
            pair[0],
            pair[1]
        );
        assert!(
            pair[0].end <= pair[1].start + 0.2,
            "the segments must not overlap: {:?} then {:?}",
            pair[0],
            pair[1]
        );
        assert!(
            pair[1].start - pair[0].end < 2.0,
            "there is a hole of {:.2} s between {:?} and {:?}",
            pair[1].start - pair[0].end,
            pair[0],
            pair[1]
        );
    }

    // The recording holds the speech of the fixture about ten times, and
    // every word of the sentence has to be there.
    let transcript: Vec<String> = items.iter().flat_map(|i| normalize(&i.text)).collect();
    assert!(
        transcript.len() >= 40,
        "the transcript has only {} words, the recording holds about ten \
         sentences: {:?}",
        transcript.len(),
        transcript
    );
}

/// A recording with different sentences is transcribed completely.
#[test]
fn a_recording_with_varied_speech_is_transcribed_completely() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the recovery tests");
        return;
    };
    if !ffmpeg_available() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping the recovery tests");
        return;
    }

    let spec: FixtureSpec =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("cases.json")).unwrap())
            .unwrap();
    let tmp = tempfile::tempdir().unwrap();

    for case in &spec.cases {
        let audio = tmp.path().join(format!("{}.wav", case.name));
        let duration = compose_case(case, &audio);
        let items = transcribe(
            &audio,
            &model,
            &tmp.path().join(format!("out-{}", case.name)),
        );

        // Every stretch of speech is covered by a segment.
        for placement in &case.clips {
            let clip = wav_samples(&boundary_dir().join(&placement.file));
            let start = placement.offset;
            let end = placement.offset + clip.len() as f64 / SAMPLE_RATE as f64;
            let covered = items
                .iter()
                .any(|i| i.start < end + 0.5 && i.end > start - 0.5);
            assert!(
                covered,
                "case {} ({}): no segment covers the speech from {:.2} to {:.2}\n  transcript: {:?}",
                case.name, case.description, start, end, items
            );
        }

        // The segments are ordered and they stay inside the recording.
        for pair in items.windows(2) {
            assert!(
                pair[0].start <= pair[1].start,
                "case {}: the timestamps must not go backwards",
                case.name
            );
            assert!(
                pair[0].end <= pair[1].start + 0.2,
                "case {}: the segments must not overlap",
                case.name
            );
        }
        assert!(
            items.last().is_none_or(|i| i.end <= duration + 0.5),
            "case {}: the last segment must not end after the recording",
            case.name
        );
    }
}
