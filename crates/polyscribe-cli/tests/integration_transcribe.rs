// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! Integration tests for the `transcribe` subcommand.
//!
//! These tests run the built binary end to end against the deterministic
//! fixtures in `tests/fixtures`. They never download models, need a GPU, or
//! touch the network. One test exercises the ffmpeg failure path and is
//! skipped with a note when ffmpeg is not on PATH.

use std::ffi::OsStr;
use std::fs;
use std::io::Read as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde::Deserialize;

#[derive(Deserialize)]
#[allow(dead_code)]
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

/// Run `polyscribe transcribe` with arguments and capture output.
fn run_transcribe(args: &[&OsStr]) -> std::process::Output {
    let mut cmd = Command::new(bin());
    cmd.arg("transcribe").args(args);
    cmd.output().expect("failed to spawn polyscribe")
}

/// Run `polyscribe` with global flags before the subcommand.
fn run_with_globals(globals: &[&OsStr], args: &[&OsStr]) -> std::process::Output {
    let mut cmd = Command::new(bin());
    cmd.args(globals).arg("transcribe").args(args);
    cmd.output().expect("failed to spawn polyscribe")
}

fn stdout_string(output: &std::process::Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout not UTF-8")
}

fn stderr_string(output: &std::process::Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr not UTF-8")
}

fn parse_stdout_json(output: &std::process::Output) -> OutputRoot {
    serde_json::from_str(&stdout_string(output)).expect("stdout is not valid transcript JSON")
}

/// Expected merged transcript of fixtures 1-s0wlz.json + 2-vikingowl.json,
/// sorted by start time with sequential ids.
fn assert_merged_1_and_2(root: &OutputRoot) {
    let texts: Vec<&str> = root.items.iter().map(|e| e.text.as_str()).collect();
    assert_eq!(
        texts,
        [
            "Viking speaks first",
            "S0wlz speaks second",
            "Viking speaks again",
            "S0wlz speaks last"
        ]
    );
    let ids: Vec<u64> = root.items.iter().map(|e| e.id).collect();
    assert_eq!(ids, [0, 1, 2, 3], "ids must be reassigned sequentially");
    let speakers: std::collections::HashSet<&str> =
        root.items.iter().map(|e| e.speaker.as_str()).collect();
    assert_eq!(speakers, ["s0wlz", "vikingowl"].into_iter().collect());
    let starts: Vec<f64> = root.items.iter().map(|e| e.start).collect();
    assert!(
        starts.windows(2).all(|w| w[0] <= w[1]),
        "items must be sorted by start time"
    );
}

#[test]
fn cli_writes_separate_outputs_by_default() {
    let out_dir = tempfile::tempdir().unwrap();
    let output = run_transcribe(&[
        fixture("1-s0wlz.json").as_os_str(),
        fixture("2-vikingowl.json").as_os_str(),
        OsStr::new("-o"),
        out_dir.path().as_os_str(),
    ]);
    assert!(
        output.status.success(),
        "CLI did not exit successfully: {}",
        stderr_string(&output)
    );

    let mut json_paths: Vec<PathBuf> = Vec::new();
    let mut toml_count = 0;
    let mut srt_count = 0;
    for entry in fs::read_dir(out_dir.path()).unwrap() {
        let p = entry.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with(".json") {
                json_paths.push(p.clone());
            }
            if name.ends_with(".toml") {
                toml_count += 1;
            }
            if name.ends_with(".srt") {
                srt_count += 1;
            }
        }
    }
    assert!(
        json_paths.len() >= 2,
        "expected at least 2 JSON files, found {}",
        json_paths.len()
    );
    assert!(toml_count >= 2, "expected at least 2 TOML files");
    assert!(srt_count >= 2, "expected at least 2 SRT files");

    // The JSON output for fixture 1 parses and carries the default speaker.
    let json_path = json_paths
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.contains("_1-s0wlz."))
                .unwrap_or(false)
        })
        .expect("missing JSON output for fixture 1");
    let content = fs::read_to_string(json_path).unwrap();
    let root: OutputRoot = serde_json::from_str(&content).unwrap();
    assert_eq!(root.items.len(), 2);
    assert!(root.items.iter().all(|e| e.speaker == "s0wlz"));
    assert_eq!(root.items[0].text, "S0wlz speaks second");

    // The SRT output for fixture 1 has exact expected content.
    let srt_path = out_dir
        .path()
        .read_dir()
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.contains("_1-s0wlz.") && n.ends_with(".srt"))
                .unwrap_or(false)
        })
        .expect("missing SRT output for fixture 1");
    let srt = fs::read_to_string(srt_path).unwrap();
    assert_eq!(
        srt,
        "1\n00:00:01,000 --> 00:00:02,000\ns0wlz: S0wlz speaks second\n\n2\n00:00:05,000 --> 00:00:06,500\ns0wlz: S0wlz speaks last\n\n"
    );

    // The TOML output parses and contains the transcript text.
    let toml_path = out_dir
        .path()
        .read_dir()
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.contains("_1-s0wlz.") && n.ends_with(".toml"))
                .unwrap_or(false)
        })
        .expect("missing TOML output for fixture 1");
    let toml_str = fs::read_to_string(toml_path).unwrap();
    let parsed: toml::Value = toml::from_str(&toml_str).expect("TOML output must parse");
    assert_eq!(
        parsed["items"][0]["text"].as_str(),
        Some("S0wlz speaks second")
    );
}

#[test]
fn cli_merges_json_inputs_with_flag_and_writes_outputs_to_temp_dir() {
    let tmp = tempfile::tempdir().unwrap();
    // Use a nested output directory to also verify auto-creation
    let base_dir = tmp.path().join("outdir");
    let base = base_dir.join("out");

    let output = run_transcribe(&[
        fixture("1-s0wlz.json").as_os_str(),
        fixture("2-vikingowl.json").as_os_str(),
        OsStr::new("-m"),
        OsStr::new("-o"),
        base.as_os_str(),
    ]);
    assert!(
        output.status.success(),
        "CLI did not exit successfully: {}",
        stderr_string(&output)
    );

    // Find the created files in the chosen output directory without depending on date prefix
    let mut found_json = None;
    let mut found_toml = false;
    let mut found_srt = false;
    for entry in fs::read_dir(&base_dir).unwrap() {
        let p = entry.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with("_out.json") {
                found_json = Some(p.clone());
            }
            if name.ends_with("_out.toml") {
                found_toml = true;
            }
            if name.ends_with("_out.srt") {
                found_srt = true;
            }
        }
    }
    let json_path = found_json.expect("missing JSON output in temp dir");
    assert!(found_toml, "missing TOML output in temp dir");
    assert!(found_srt, "missing SRT output in temp dir");

    // The merged JSON contains both inputs, sorted, with sequential ids.
    let content = fs::read_to_string(json_path).unwrap();
    let root: OutputRoot = serde_json::from_str(&content).unwrap();
    assert_merged_1_and_2(&root);
}

#[test]
fn cli_prints_json_to_stdout_when_no_output_path_merge_mode() {
    let output = run_transcribe(&[
        fixture("1-s0wlz.json").as_os_str(),
        fixture("2-vikingowl.json").as_os_str(),
        OsStr::new("-m"),
    ]);
    assert!(output.status.success(), "CLI failed");
    let root = parse_stdout_json(&output);
    assert_merged_1_and_2(&root);
}

#[test]
fn cli_single_input_prints_json_to_stdout_by_default() {
    let output = run_transcribe(&[fixture("3-schmendrizzle.json").as_os_str()]);
    assert!(output.status.success(), "CLI failed");
    let root = parse_stdout_json(&output);
    assert_eq!(root.items.len(), 1);
    assert_eq!(root.items[0].text, "Schmendrizzle here");
    assert_eq!(root.items[0].speaker, "schmendrizzle");
    assert_eq!(root.items[0].id, 0);
}

#[test]
fn cli_merge_and_separate_writes_both_kinds_of_outputs() {
    let out_dir = tempfile::tempdir().unwrap();
    let output = run_transcribe(&[
        fixture("1-s0wlz.json").as_os_str(),
        fixture("2-vikingowl.json").as_os_str(),
        OsStr::new("--merge-and-separate"),
        OsStr::new("-o"),
        out_dir.path().as_os_str(),
    ]);
    assert!(
        output.status.success(),
        "CLI did not exit successfully: {}",
        stderr_string(&output)
    );

    // Count outputs: per-file outputs (2 JSON/TOML/SRT) plus a merged set.
    let mut json_count = 0;
    let mut toml_count = 0;
    let mut srt_count = 0;
    let mut merged_json = None;
    for entry in fs::read_dir(out_dir.path()).unwrap() {
        let p = entry.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with(".json") {
                json_count += 1;
            }
            if name.ends_with(".toml") {
                toml_count += 1;
            }
            if name.ends_with(".srt") {
                srt_count += 1;
            }
            if name.ends_with("_merged.json") {
                merged_json = Some(p.clone());
            }
        }
    }
    assert!(
        json_count >= 3,
        "expected at least 3 JSON files, found {}",
        json_count
    );
    assert!(toml_count >= 3, "expected at least 3 TOML files");
    assert!(srt_count >= 3, "expected at least 3 SRT files");

    let merged_json = merged_json.expect("missing merged JSON output ending with _merged.json");
    let content = fs::read_to_string(merged_json).unwrap();
    let root: OutputRoot = serde_json::from_str(&content).unwrap();
    assert_merged_1_and_2(&root);
}

#[test]
fn cli_set_speaker_names_merge_prompts_and_uses_names() {
    // Also validate that -q does not suppress prompts by running with -q
    let mut child = Command::new(bin())
        .args([OsStr::new("--quiet")])
        .arg("transcribe")
        .arg(fixture("1-s0wlz.json"))
        .arg(fixture("2-vikingowl.json"))
        .arg("-m")
        .arg("--set-speaker-names")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn polyscribe");

    {
        use std::io::Write as _;
        let stdin = child.stdin.as_mut().expect("failed to open stdin");
        // Provide two names for two files
        writeln!(stdin, "Alpha").unwrap();
        writeln!(stdin, "Beta").unwrap();
    }

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(output.status.success(), "CLI did not exit successfully");

    let root = parse_stdout_json(&output);
    let speakers: std::collections::HashSet<String> =
        root.items.into_iter().map(|e| e.speaker).collect();
    assert!(speakers.contains("Alpha"), "Alpha not found in speakers");
    assert!(speakers.contains("Beta"), "Beta not found in speakers");
}

#[test]
fn cli_no_interaction_skips_speaker_prompts_and_uses_defaults() {
    let output = run_with_globals(
        &[OsStr::new("--no-interaction")],
        &[
            fixture("1-s0wlz.json").as_os_str(),
            fixture("2-vikingowl.json").as_os_str(),
            OsStr::new("-m"),
            OsStr::new("--set-speaker-names"),
        ],
    );
    assert!(
        output.status.success(),
        "CLI did not exit successfully: {}",
        stderr_string(&output)
    );

    let root = parse_stdout_json(&output);
    let speakers: std::collections::HashSet<String> =
        root.items.into_iter().map(|e| e.speaker).collect();
    // Defaults are the sanitized file stems: "1-s0wlz" -> "s0wlz".
    assert!(speakers.contains("s0wlz"), "default s0wlz not used");
    assert!(speakers.contains("vikingowl"), "default vikingowl not used");
}

#[test]
fn cli_set_speaker_names_separate_single_input() {
    let out_dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(bin())
        .arg("transcribe")
        .arg(fixture("3-schmendrizzle.json"))
        .arg("--set-speaker-names")
        .arg("-o")
        .arg(out_dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn polyscribe");

    {
        use std::io::Write as _;
        let stdin = child.stdin.as_mut().expect("failed to open stdin");
        writeln!(stdin, "ChosenOne").unwrap();
    }

    let status = child.wait().expect("failed to wait on child");
    assert!(status.success(), "CLI did not exit successfully");

    // Find created JSON
    let mut json_paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(out_dir.path()).unwrap() {
        let p = entry.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str())
            && name.ends_with(".json")
        {
            json_paths.push(p.clone());
        }
    }
    assert!(!json_paths.is_empty(), "no JSON outputs created");
    let mut buf = String::new();
    std::fs::File::open(&json_paths[0])
        .unwrap()
        .read_to_string(&mut buf)
        .unwrap();
    let root: OutputRoot = serde_json::from_str(&buf).unwrap();
    assert!(root.items.iter().all(|e| e.speaker == "ChosenOne"));
}

// Verbosity behavior tests
#[test]
fn verbosity_quiet_suppresses_logs_but_keeps_stdout() {
    let output = run_with_globals(
        &[OsStr::new("--quiet"), OsStr::new("-v")], // ensure -q overrides -v
        &[
            fixture("1-s0wlz.json").as_os_str(),
            fixture("2-vikingowl.json").as_os_str(),
            OsStr::new("-m"),
        ],
    );
    assert!(output.status.success());
    let stdout = stdout_string(&output);
    assert!(
        stdout.contains("\"items\""),
        "stdout JSON should be present in quiet mode"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.trim().is_empty(),
        "stderr should be empty in quiet mode, got: {}",
        stderr
    );
}

#[test]
fn verbosity_verbose_emits_debug_logs_on_stderr() {
    let output = run_with_globals(
        &[OsStr::new("-v")],
        &[
            fixture("1-s0wlz.json").as_os_str(),
            fixture("2-vikingowl.json").as_os_str(),
            OsStr::new("-m"),
        ],
    );
    assert!(output.status.success());
    let stdout = stdout_string(&output);
    assert!(stdout.contains("\"items\""));
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("Mode: merge"),
        "stderr should contain debug log with -v"
    );
}

// CLI argument validation and error-path tests
#[test]
fn cli_transcribe_requires_inputs() {
    let output = Command::new(bin())
        .arg("transcribe")
        .output()
        .expect("failed to spawn polyscribe");
    assert!(
        !output.status.success(),
        "expected failure with no inputs (clap requires them)"
    );
    let stderr = stderr_string(&output);
    assert!(stderr.contains("required"), "unexpected stderr: {}", stderr);
}

#[test]
fn cli_rejects_unsupported_input_type() {
    let output = run_transcribe(&[fixture("notes.txt").as_os_str()]);
    assert!(!output.status.success(), "expected failure for .txt input");
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("Unsupported input type"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn cli_requires_language_for_audio_inputs() {
    let output = run_transcribe(&[fixture("sample.mp3").as_os_str()]);
    assert!(
        !output.status.success(),
        "expected failure without --language"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("--language"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn cli_errors_on_missing_input_file() {
    let missing = fixture("does-not-exist.json");
    let output = run_transcribe(&[missing.as_os_str(), OsStr::new("-m")]);
    assert!(
        !output.status.success(),
        "expected failure for missing file"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("Failed to open"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn cli_errors_on_invalid_json_transcript() {
    let output = run_transcribe(&[fixture("invalid.json").as_os_str(), OsStr::new("-m")]);
    assert!(
        !output.status.success(),
        "expected failure for invalid JSON"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("Invalid JSON transcript"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn cli_merge_and_separate_requires_output_dir() {
    let output = run_transcribe(&[
        fixture("1-s0wlz.json").as_os_str(),
        fixture("2-vikingowl.json").as_os_str(),
        OsStr::new("--merge-and-separate"),
    ]);
    assert!(
        !output.status.success(),
        "expected failure without -o for --merge-and-separate"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("requires -o"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn cli_multiple_inputs_without_merge_require_output_dir() {
    let output = run_transcribe(&[
        fixture("1-s0wlz.json").as_os_str(),
        fixture("2-vikingowl.json").as_os_str(),
    ]);
    assert!(
        !output.status.success(),
        "expected failure for multiple inputs without -m and without -o"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("require -o OUTPUT_DIR"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn cli_rejects_vulkan_backend_value() {
    let output = run_transcribe(&[
        OsStr::new("--gpu-backend"),
        OsStr::new("vulkan"),
        fixture("1-s0wlz.json").as_os_str(),
    ]);
    assert_eq!(output.status.code(), Some(2), "clap should reject vulkan");
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("invalid value") && stderr.contains("cuda"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn cli_trailing_output_argument_is_treated_as_output() {
    // Documented heuristic: with >= 2 inputs and no -o, a trailing argument
    // that does not exist on disk is treated as the output path. In merge mode
    // the output path is a file base, so the merged JSON lands in its parent.
    let tmp = tempfile::tempdir().unwrap();
    let out_base = tmp.path().join("out_base");
    let output = Command::new(bin())
        .arg("transcribe")
        .arg(fixture("1-s0wlz.json"))
        .arg(fixture("2-vikingowl.json"))
        .arg("-m")
        .arg(&out_base)
        .current_dir(tmp.path())
        .output()
        .expect("failed to spawn polyscribe");
    assert!(
        output.status.success(),
        "CLI did not exit successfully: {}",
        stderr_string(&output)
    );
    // The merged output is written next to the given base path (parent dir).
    let produced: Vec<PathBuf> = fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.ends_with("_out_base.json"))
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        produced.len(),
        1,
        "expected one merged JSON named *_out_base.json in temp dir"
    );
    let content = fs::read_to_string(&produced[0]).unwrap();
    let root: OutputRoot = serde_json::from_str(&content).unwrap();
    assert_merged_1_and_2(&root);
}

#[test]
fn cli_reports_ffmpeg_failure_for_invalid_media() {
    // The fixture is not real audio; ffmpeg must fail and the error must
    // surface with context. Requires ffmpeg on PATH.
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("NOT RUN: ffmpeg is not available on PATH; skipping ffmpeg failure test");
        return;
    }
    let output = run_transcribe(&[
        fixture("sample.mp3").as_os_str(),
        OsStr::new("--language"),
        OsStr::new("en"),
    ]);
    assert!(
        !output.status.success(),
        "expected failure when ffmpeg cannot decode the input"
    );
    let stderr = stderr_string(&output);
    assert!(
        stderr.contains("ffmpeg failed"),
        "unexpected stderr: {}",
        stderr
    );
}
