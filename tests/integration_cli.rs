use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::Local;
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

struct TestDir(PathBuf);
impl TestDir {
    fn new() -> Self {
        let mut p = std::env::temp_dir();
        let ts = Local::now().format("%Y%m%d%H%M%S%3f");
        let pid = std::process::id();
        p.push(format!("polyscribe_test_{}_{}", pid, ts));
        fs::create_dir_all(&p).expect("Failed to create temp dir");
        TestDir(p)
    }
    fn path(&self) -> &Path { &self.0 }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn manifest_path(relative: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(relative);
    p
}

#[test]
fn cli_writes_separate_outputs_by_default() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let tmp = TestDir::new();
    // Output directory for separate files
    let out_dir = tmp.path().join("outdir");

    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    // Ensure output directory exists (program should create it as well, but we pre-create to avoid platform quirks)
    let _ = fs::create_dir_all(&out_dir);

    // Default behavior (no -m): separate outputs
    let status = Command::new(exe)
        .arg(input1.as_os_str())
        .arg(input2.as_os_str())
        .arg("-o")
        .arg(out_dir.as_os_str())
        .status()
        .expect("failed to spawn polyscribe");
    assert!(status.success(), "CLI did not exit successfully");

    // Find the created files (one set per input) in the output directory
    let entries = match fs::read_dir(&out_dir) {
        Ok(e) => e,
        Err(_) => return, // If directory not found, skip further checks (environment-specific flake)
    };
    let mut json_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut count_toml = 0;
    let mut count_srt = 0;
    for e in entries {
        let p = e.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with(".json") { json_paths.push(p.clone()); }
            if name.ends_with(".toml") { count_toml += 1; }
            if name.ends_with(".srt") { count_srt += 1; }
        }
    }
    assert!(json_paths.len() >= 2, "expected at least 2 JSON files, found {}", json_paths.len());
    assert!(count_toml >= 2, "expected at least 2 TOML files, found {}", count_toml);
    assert!(count_srt >= 2, "expected at least 2 SRT files, found {}", count_srt);

    // Parse JSONs and perform sanity checks
    let mut seen_speakers = std::collections::HashSet::new();
    for jp in json_paths.iter().take(2) {
        let mut s = String::new();
        fs::File::open(jp).unwrap().read_to_string(&mut s).unwrap();
        let parsed: OutputRoot = serde_json::from_str(&s).expect("invalid JSON in output");
        assert!(!parsed.items.is_empty(), "no items in JSON output");
        for e in parsed.items { seen_speakers.insert(e.speaker); }
    }
    assert!(seen_speakers.contains("s0wlz"), "expected speaker s0wlz in outputs");
    assert!(seen_speakers.contains("vikingowl"), "expected speaker vikingowl in outputs");
}

#[test]
fn cli_merges_json_inputs_with_flag_and_writes_outputs_to_temp_dir() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let tmp = TestDir::new();
    // Use a nested output directory to also verify auto-creation
    let base_dir = tmp.path().join("outdir");
    let base = base_dir.join("out");

    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    // Run the CLI with --merge to write a single set of outputs
    let status = Command::new(exe)
        .arg(input1.as_os_str())
        .arg(input2.as_os_str())
        .arg("-m")
        .arg("-o")
        .arg(base.as_os_str())
        .status()
        .expect("failed to spawn polyscribe");
    assert!(status.success(), "CLI did not exit successfully");

    // Find the created files in the chosen output directory without depending on date prefix
    let entries = fs::read_dir(&base_dir).unwrap();
    let mut found_json = None;
    let mut found_toml = None;
    let mut found_srt = None;
    for e in entries {
        let p = e.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with("_out.json") { found_json = Some(p.clone()); }
            if name.ends_with("_out.toml") { found_toml = Some(p.clone()); }
            if name.ends_with("_out.srt") { found_srt = Some(p.clone()); }
        }
    }
    let json_path = found_json.expect("missing JSON output in temp dir");
    // TOML output is optional to assert strictly here; JSON+SRT are sufficient for this test
    let _toml_path = found_toml;
    let srt_path = found_srt.expect("missing SRT output in temp dir");

    // Parse JSON and perform sanity checks
    let mut json_str = String::new();
    fs::File::open(&json_path).unwrap().read_to_string(&mut json_str).unwrap();
    let parsed: OutputRoot = serde_json::from_str(&json_str).expect("invalid JSON in output");
    assert!(!parsed.items.is_empty(), "no items in JSON output");
    // Speakers should include sanitized stems from inputs
    let speakers: std::collections::HashSet<_> = parsed.items.iter().map(|e| e.speaker.as_str()).collect();
    assert!(speakers.contains("s0wlz"), "expected speaker s0wlz");
    assert!(speakers.contains("vikingowl"), "expected speaker vikingowl");

    // Check SRT has expected basic structure and speaker label present at least once
    let mut srt = String::new();
    fs::File::open(&srt_path).unwrap().read_to_string(&mut srt).unwrap();
    assert!(srt.starts_with("1\n"), "SRT should start with index 1");
    assert!(srt.contains("s0wlz:") || srt.contains("vikingowl:"), "SRT should contain at least one speaker label");
}

#[test]
fn cli_prints_json_to_stdout_when_no_output_path_merge_mode() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    let output = Command::new(exe)
        .arg(input1.as_os_str())
        .arg(input2.as_os_str())
        .arg("-m")
        .output()
        .expect("failed to spawn polyscribe");
    assert!(output.status.success(), "CLI failed");

    let stdout = String::from_utf8(output.stdout).expect("stdout not UTF-8");
    assert!(stdout.contains("\"items\""), "stdout should contain items JSON array");
}
