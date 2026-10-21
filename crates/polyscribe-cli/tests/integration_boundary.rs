// SPDX-License-Identifier: MIT

//! Regression tests for the chunk boundaries of the transcription pipeline.
//!
//! The fixtures are built from the committed speech clips and the case
//! definitions in `tests/fixtures/boundary/cases.json`. Every case places
//! speech around the point where the pipeline cuts the recording into
//! windows, so the reconciliation of the segments at that point is
//! exercised with real speech.
//!
//! The tests need a Whisper model, so they are skipped without one, like
//! the other end-to-end tests.

use std::collections::HashMap;
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
struct ClipInfo {
    text: String,
    #[allow(dead_code)]
    duration: f64,
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
    reference: String,
}

#[derive(Deserialize)]
struct FixtureSpec {
    clips: HashMap<String, ClipInfo>,
    cases: Vec<Case>,
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("boundary")
}

fn load_spec() -> FixtureSpec {
    let raw = std::fs::read_to_string(fixture_dir().join("cases.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

/// The samples of a WAV file, without the header.
fn wav_samples(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
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

/// Write the fixture of a case: silence with the speech clips placed at
/// their offsets.
fn compose_case(case: &Case, out_path: &Path) -> f64 {
    let mut samples: Vec<i16> = Vec::new();
    let mut cursor = 0.0f64;
    for placement in &case.clips {
        let gap = ((placement.offset - cursor) * SAMPLE_RATE as f64).round() as usize;
        samples.extend(std::iter::repeat_n(0i16, gap));
        let clip = wav_samples(&fixture_dir().join(&placement.file));
        cursor = placement.offset + clip.len() as f64 / SAMPLE_RATE as f64;
        samples.extend(clip);
    }

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
    for s in &samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(out_path, &bytes).unwrap();
    samples.len() as f64 / SAMPLE_RATE as f64
}

fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

fn e2e_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("POLYSCRIBE_E2E_MODEL").ok()?);
    path.is_file().then_some(path)
}

#[test]
fn boundary_cases_keep_their_speech() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping boundary tests");
        return;
    };
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping boundary tests");
        return;
    }

    let spec = load_spec();
    let tmp = tempfile::tempdir().unwrap();

    for case in &spec.cases {
        let audio = tmp.path().join(format!("{}.wav", case.name));
        let duration = compose_case(case, &audio);
        let out_dir = tmp.path().join(format!("out-{}", case.name));

        let output = Command::new(bin())
            .args(["--no-interaction", "--no-progress"])
            .arg("transcribe")
            .arg(&audio)
            .args([OsStr::new("--language"), OsStr::new("en")])
            .arg("-o")
            .arg(&out_dir)
            .env("WHISPER_MODEL", &model)
            .output()
            .expect("failed to spawn polyscribe");
        assert!(
            output.status.success(),
            "case {} failed: {}",
            case.name,
            String::from_utf8_lossy(&output.stderr)
        );

        let json_path = std::fs::read_dir(&out_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "json"))
            .unwrap_or_else(|| panic!("case {} produced no JSON output", case.name));
        let root: OutputRoot = serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap())
            .expect("the transcript must be valid JSON");

        // The transcript is ordered and its segments do not overlap.
        for pair in root.items.windows(2) {
            assert!(
                pair[0].start <= pair[1].start,
                "case {}: the timestamps must not go backwards",
                case.name
            );
            assert!(
                pair[0].end <= pair[1].start + 0.2,
                "case {}: the segments must not overlap: {:?} then {:?}",
                case.name,
                pair[0],
                pair[1]
            );
        }
        assert!(
            root.items.first().is_none_or(|i| i.start >= 0.0),
            "case {}: the first segment must not start before the recording",
            case.name
        );
        assert!(
            root.items.last().is_none_or(|i| i.end <= duration + 0.5),
            "case {}: the last segment must not end after the recording",
            case.name
        );

        // Every stretch of speech is covered by a segment. A segment that
        // was lost or dropped at a window boundary would leave its time
        // range without any segment, which is what this checks. The words
        // themselves are not compared: whisper can replace a word by a
        // similar sounding one, and that is a recognition error, not a
        // defect of the window handling.
        for placement in &case.clips {
            let clip = spec
                .clips
                .get(&placement.file)
                .unwrap_or_else(|| panic!("unknown clip {}", placement.file));
            let start = placement.offset;
            let end = placement.offset + clip.duration;
            let covered = root
                .items
                .iter()
                .any(|i| i.start < end + 0.3 && i.end > start - 0.3);
            assert!(
                covered,
                "case {} ({}): no segment covers the speech from {:.2} to {:.2}\n  transcript: {:?}",
                case.name,
                case.description,
                start,
                end,
                root.items
                    .iter()
                    .map(|i| (i.start, i.end, &i.text))
                    .collect::<Vec<_>>()
            );
        }

        // The transcript holds most of the words of the reference. A whole
        // lost segment would drop a phrase, which this catches even when
        // whisper misrecognizes single words.
        let reference = normalize(&case.reference);
        let transcript: Vec<String> = root.items.iter().flat_map(|i| normalize(&i.text)).collect();
        assert!(
            transcript.len() * 2 >= reference.len(),
            "case {} ({}): the transcript has {} words, the reference has {}\n  transcript: {:?}",
            case.name,
            case.description,
            transcript.len(),
            reference.len(),
            transcript
        );
    }
}

#[test]
fn boundary_case_definitions_are_consistent() {
    // Every clip of every case is known, and the reference is the text of
    // the clips in the order they are placed.
    let spec = load_spec();
    for case in &spec.cases {
        let mut expected: Vec<String> = Vec::new();
        for placement in &case.clips {
            let clip = spec.clips.get(&placement.file).unwrap_or_else(|| {
                panic!(
                    "case {} uses the unknown clip {}",
                    case.name, placement.file
                )
            });
            expected.push(clip.text.clone());
        }
        assert_eq!(
            case.reference,
            expected.join(" "),
            "the reference of case {} must match its clips",
            case.name
        );
    }
}

#[test]
fn a_long_silence_is_not_decoded_over_and_over() {
    // whisper reports short segments for silence. A window that started
    // where the last segment ended would decode the same silence again
    // and again, which is slow and produces a hallucination per window.
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the silence test");
        return;
    };
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping the silence test");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let audio = tmp.path().join("silence.wav");
    // One minute of silence.
    let samples = (60 * SAMPLE_RATE) as usize;
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&((36 + samples * 2) as u32).to_le_bytes());
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
    bytes.extend_from_slice(&(samples as u32 * 2).to_le_bytes());
    bytes.extend(std::iter::repeat_n(0u8, samples * 2));
    std::fs::write(&audio, &bytes).unwrap();

    let out_dir = tmp.path().join("out");
    let output = Command::new(bin())
        .args(["-vv", "--no-interaction", "--no-progress"])
        .arg("transcribe")
        .arg(&audio)
        .args([OsStr::new("--language"), OsStr::new("en")])
        .arg("-o")
        .arg(&out_dir)
        .env("WHISPER_MODEL", &model)
        .output()
        .expect("failed to spawn polyscribe");
    assert!(
        output.status.success(),
        "transcribe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    let windows: usize = stderr
        .lines()
        .find_map(|l| l.split("Transcribed ").nth(1))
        .and_then(|l| l.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("the window count is missing from the log: {stderr}"));
    // The window holds 40 seconds of audio, so a minute of silence needs two
    // windows. More than four means that the silence is decoded again and
    // again. Before the window handling was fixed, the same recording took
    // five windows with the small model and sixteen with the medium one.
    assert!(
        windows <= 4,
        "a minute of silence must not be decoded over and over, but it took {windows} windows"
    );
}
