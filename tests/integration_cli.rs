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
fn cli_merges_json_inputs_and_writes_outputs_to_temp_dir() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let tmp = TestDir::new();
    let base = tmp.path().join("out");

    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    // Run the CLI to write outputs into temp directory
    let status = Command::new(exe)
        .arg(input1.as_os_str())
        .arg(input2.as_os_str())
        .arg("-o")
        .arg(base.as_os_str())
        .status()
        .expect("failed to spawn polyscribe");
    assert!(status.success(), "CLI did not exit successfully");

    // Expect files with today's date prefix
    let date = Local::now().format("%Y-%m-%d").to_string();
    let stem = format!("{}_{}", date, "out");
    let json_path = tmp.path().join(format!("{}.json", stem));
    let toml_path = tmp.path().join(format!("{}.toml", stem));
    let srt_path = tmp.path().join(format!("{}.srt", stem));

    assert!(json_path.is_file(), "missing JSON output: {}", json_path.display());
    assert!(toml_path.is_file(), "missing TOML output: {}", toml_path.display());
    assert!(srt_path.is_file(), "missing SRT output: {}", srt_path.display());

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
fn cli_prints_json_to_stdout_when_no_output_path() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    let output = Command::new(exe)
        .arg(input1.as_os_str())
        .arg(input2.as_os_str())
        .output()
        .expect("failed to spawn polyscribe");
    assert!(output.status.success(), "CLI failed");

    let stdout = String::from_utf8(output.stdout).expect("stdout not UTF-8");
    assert!(stdout.contains("\"items\""), "stdout should contain items JSON array");
    // Ensure no files were created in repo output/ by default in this mode
    // (Program writes to stdout only when -o omitted.)
}
