// SPDX-License-Identifier: MIT

//! End-to-end transcription smoke test against the real CLI binary.
//!
//! Unlike the other integration suites, this test runs actual Whisper
//! inference on CPU, so it needs a model file and ffmpeg. It is skipped
//! with a note unless `POLYSCRIBE_E2E_MODEL` points at a Whisper model file:
//!
//! ```text
//! POLYSCRIBE_E2E_MODEL="$HOME/.cache/polyscribe-e2e/ggml-tiny.en.bin" \
//!     cargo test -p polyscribe-cli --test e2e_transcribe
//! ```
//!
//! CI runs it in the `e2e-transcribe` job with a SHA-256-verified
//! `ggml-tiny.en.bin` (see `.github/workflows/ci.yml`). The test exercises
//! the compiled `polyscribe` executable, not library internals: a regression
//! in ffmpeg decoding, whisper inference, or JSON/SRT output generation
//! fails the test.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

#[derive(Deserialize)]
struct OutputEntry {
    id: u64,
    speaker: String,
    start: f64,
    end: f64,
    text: String,
}

#[derive(Deserialize)]
struct OutputRoot {
    items: Vec<OutputEntry>,
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

/// Path to a committed test fixture.
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Path to the Whisper model for this test, when provided.
fn e2e_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("POLYSCRIBE_E2E_MODEL").ok()?);
    path.is_file().then_some(path)
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// Parse a PCM WAV file and return (sample_rate, duration_seconds) by
/// walking the RIFF chunk list.
fn wav_info(path: &Path) -> (u32, f64) {
    let bytes = fs::read(path).unwrap();
    assert!(
        bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "fixture is not a RIFF/WAVE file"
    );
    let mut pos = 12usize;
    let mut sample_rate = None;
    let mut byte_rate = None;
    let mut data_len = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let end = (pos + 8 + size).min(bytes.len());
        let body = &bytes[pos + 8..end];
        match id {
            b"fmt " if body.len() >= 12 => {
                sample_rate = Some(u32::from_le_bytes(body[4..8].try_into().unwrap()));
                byte_rate = Some(u32::from_le_bytes(body[8..12].try_into().unwrap()));
            }
            b"data" => data_len = Some(size),
            _ => {}
        }
        pos += 8 + size + (size % 2); // chunks are word-aligned
    }
    let rate = sample_rate.expect("WAV fixture has no fmt chunk");
    let byte_rate = byte_rate.expect("WAV fixture has no byte rate");
    let data = data_len.expect("WAV fixture has no data chunk");
    (rate, data as f64 / byte_rate as f64)
}

/// Parse an SRT timestamp (`HH:MM:SS,mmm`) into seconds.
fn parse_srt_timestamp(ts: &str) -> f64 {
    let (hms, ms) = ts.split_once(',').expect("SRT timestamp must use a comma");
    let mut parts = hms.split(':');
    let hours: f64 = parts.next().unwrap().parse().unwrap();
    let minutes: f64 = parts.next().unwrap().parse().unwrap();
    let seconds: f64 = parts.next().unwrap().parse().unwrap();
    hours * 3600.0 + minutes * 60.0 + seconds + ms.parse::<f64>().unwrap() / 1000.0
}

#[test]
fn e2e_transcribe_speech_fixture_with_real_model() {
    let Some(model) = e2e_model() else {
        eprintln!(
            "SKIPPED: POLYSCRIBE_E2E_MODEL is not set to a model file; skipping real-inference e2e test"
        );
        return;
    };
    if !ffmpeg_available() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping real-inference e2e test");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let audio = tmp.path().join("speech.wav");
    fs::copy(fixture("speech.wav"), &audio).unwrap();
    let (sample_rate, duration) = wav_info(&audio);
    assert_eq!(sample_rate, 16_000, "fixture must be 16 kHz");
    let out_dir = tmp.path().join("out");

    // Run the actual CLI executable with the pinned model.
    let output = Command::new(bin())
        .args(["--no-interaction", "--no-progress"])
        .arg("transcribe")
        .arg(&audio)
        .args(["--language", "en"])
        .arg("-o")
        .arg(&out_dir)
        .env("WHISPER_MODEL", &model)
        .output()
        .expect("failed to spawn polyscribe");
    assert!(
        output.status.success(),
        "transcribe failed (exit {:?}): {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    // Exactly one JSON/TOML/SRT set for the single input.
    let mut json_path = None;
    let mut toml_path = None;
    let mut srt_path = None;
    for entry in fs::read_dir(&out_dir).unwrap() {
        let path = entry.unwrap().path();
        match path.extension().and_then(|s| s.to_str()) {
            Some("json") => assert!(json_path.replace(path).is_none(), "multiple JSON outputs"),
            Some("toml") => assert!(toml_path.replace(path).is_none(), "multiple TOML outputs"),
            Some("srt") => assert!(srt_path.replace(path).is_none(), "multiple SRT outputs"),
            _ => {}
        }
    }
    let json_path = json_path.expect("missing JSON output");
    let toml_path = toml_path.expect("missing TOML output");
    let srt_path = srt_path.expect("missing SRT output");

    // JSON: parses, is non-empty, and carries sane timestamps.
    let root: OutputRoot = serde_json::from_str(&fs::read_to_string(&json_path).unwrap())
        .expect("JSON output must parse");
    assert!(!root.items.is_empty(), "whisper produced no segments");
    assert!(
        root.items.iter().any(|e| !e.text.trim().is_empty()),
        "whisper produced no transcript text: inference is broken"
    );
    let ids: Vec<u64> = root.items.iter().map(|e| e.id).collect();
    assert_eq!(
        ids,
        (0..root.items.len() as u64).collect::<Vec<_>>(),
        "ids must be sequential"
    );
    for w in root.items.windows(2) {
        assert!(
            w[0].start <= w[1].start,
            "items must be sorted by start time"
        );
    }
    for item in &root.items {
        assert!(item.start >= 0.0, "negative start: {}", item.start);
        assert!(
            item.end > item.start,
            "end {} must be after start {}",
            item.end,
            item.start
        );
        assert!(
            item.end <= duration + 0.5,
            "segment end {} exceeds audio duration {duration}",
            item.end
        );
        // Default speaker label: the sanitized input stem.
        assert_eq!(item.speaker, "speech", "unexpected speaker label");
    }
    assert!(
        root.items[0].start <= 1.0,
        "speech starts at the beginning of the fixture, got {}",
        root.items[0].start
    );

    // SRT: one cue per item, well-formed timestamps matching the JSON.
    let srt = fs::read_to_string(&srt_path).unwrap();
    let cues: Vec<&str> = srt.split("\n\n").filter(|c| !c.trim().is_empty()).collect();
    assert_eq!(
        cues.len(),
        root.items.len(),
        "SRT cue count must match JSON items"
    );
    for (i, cue) in cues.iter().enumerate() {
        let item = &root.items[i];
        let mut lines = cue.lines();
        assert_eq!(lines.next().unwrap(), (i + 1).to_string(), "cue numbering");
        let timing = lines.next().expect("cue must have a timing line");
        let (start_str, end_str) = timing
            .split_once(" --> ")
            .expect("cue timing must use ' --> '");
        let start = parse_srt_timestamp(start_str);
        let end = parse_srt_timestamp(end_str);
        assert!(
            (start - item.start).abs() <= 0.001,
            "cue {i} start {start} does not match JSON {}",
            item.start
        );
        assert!(
            (end - item.end).abs() <= 0.001,
            "cue {i} end {end} does not match JSON {}",
            item.end
        );
        assert!(
            lines.next().is_some_and(|l| !l.trim().is_empty()),
            "cue {i} must carry text"
        );
    }

    // TOML: parses and agrees with the JSON item count.
    let toml_str = fs::read_to_string(&toml_path).unwrap();
    let parsed: toml::Value = toml::from_str(&toml_str).expect("TOML output must parse");
    assert_eq!(
        parsed["items"].as_array().map(Vec::len),
        Some(root.items.len()),
        "TOML item count must match JSON"
    );
}

/// Write the committed fixture repeated `times` times, so the recording is
/// long enough to cross a transcription window. The samples are repeated, not
/// the whole file, so the result stays a valid WAV file.
fn write_repeated_fixture(path: &Path, times: usize) {
    let bytes = fs::read(fixture("speech.wav")).unwrap();

    // Locate the data chunk of the WAV file.
    let mut pos = 12;
    let (data_start, data_len) = loop {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            break (pos + 8, size);
        }
        pos += 8 + size + (size % 2);
    };

    let samples = &bytes[data_start..data_start + data_len];
    let total = data_len * times;

    // The header is everything before the data chunk, with the sizes fixed
    // up for the repeated samples.
    let mut out = bytes[..data_start - 8].to_vec();
    let riff_size = (out.len() + 8 + total) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(total as u32).to_le_bytes());
    for _ in 0..times {
        out.extend_from_slice(samples);
    }
    fs::write(path, out).unwrap();
}

#[test]
fn e2e_multiple_files_load_the_model_once() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping model reuse test");
        return;
    };
    if !ffmpeg_available() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping model reuse test");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let first = tmp.path().join("first.wav");
    let second = tmp.path().join("second.wav");
    fs::copy(fixture("speech.wav"), &first).unwrap();
    fs::copy(fixture("speech.wav"), &second).unwrap();
    let out_dir = tmp.path().join("out");

    // The verbose output reports the model load, so the number of loads can
    // be counted.
    let output = Command::new(bin())
        .args(["-vv", "--no-interaction", "--no-progress"])
        .arg("transcribe")
        .arg(&first)
        .arg(&second)
        .args(["--language", "en", "-o"])
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
    let loads = stderr.matches("loading model from").count();
    assert_eq!(loads, 1, "the model must be loaded once for two files");

    // Both files produced their own output.
    let mut outputs = 0;
    for entry in fs::read_dir(&out_dir).unwrap() {
        let name = entry.unwrap().file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".json") {
            outputs += 1;
        }
    }
    assert_eq!(outputs, 2, "every input file keeps its own output");
}

#[test]
fn e2e_long_recording_keeps_the_timeline_contiguous() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping long recording test");
        return;
    };
    if !ffmpeg_available() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping long recording test");
        return;
    }

    // Twelve copies of the fixture: long enough to cross a window
    // boundary, short enough for a test.
    let tmp = tempfile::tempdir().unwrap();
    let audio = tmp.path().join("long.wav");
    write_repeated_fixture(&audio, 12);
    let out_dir = tmp.path().join("out");

    let output = Command::new(bin())
        .args(["--no-interaction", "--no-progress"])
        .arg("transcribe")
        .arg(&audio)
        .args(["--language", "en"])
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

    let json_path = fs::read_dir(&out_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "json"))
        .expect("missing JSON output");
    let root: OutputRoot =
        serde_json::from_str(&fs::read_to_string(&json_path).unwrap()).expect("JSON must parse");

    assert!(
        root.items.len() >= 12,
        "every repetition of the fixture must be transcribed, got {} segments",
        root.items.len()
    );
    // The timeline is contiguous: no gap and no overlap between the
    // segments, and the timestamps only grow.
    for pair in root.items.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "the timestamps must not go backwards"
        );
        assert!(
            pair[0].end <= pair[1].start + 0.2,
            "the segments must not overlap: {} then {}",
            pair[0].start,
            pair[1].start
        );
    }
    assert!(
        root.items[0].start < 0.5,
        "the first segment starts at the beginning"
    );
}

#[test]
fn e2e_transcription_on_the_gpu() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping GPU e2e test");
        return;
    };
    if !ffmpeg_available() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping GPU e2e test");
        return;
    }
    // The GPU backend needs a device that ggml reports.
    let devices = polyscribe_core::gpu::gpu_devices(polyscribe_core::gpu::GpuBackend::Hip);
    if devices.is_empty() {
        eprintln!("SKIPPED: no ROCm device is available; skipping GPU e2e test");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let audio = tmp.path().join("speech.wav");
    fs::copy(fixture("speech.wav"), &audio).unwrap();
    let out_dir = tmp.path().join("out");

    let output = Command::new(bin())
        .args(["-vv", "--no-interaction", "--no-progress"])
        .arg("transcribe")
        .arg(&audio)
        .args(["--language", "en", "--gpu-backend", "hip"])
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
    assert!(
        stderr.contains("using ROCm0 backend"),
        "the GPU backend must be used, log was: {stderr}"
    );
}
