// SPDX-License-Identifier: MIT
// Tests for --out-format flag behavior

use std::fs;
use std::process::Command;
use std::path::PathBuf;

fn manifest_path(relative: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(relative);
    p
}

#[test]
fn out_format_single_json_only() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let out_dir = manifest_path("target/tmp/itest_outfmt_json_only");
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir).unwrap();

    let input1 = manifest_path("input/1-s0wlz.json");

    let status = Command::new(exe)
        .arg(input1.as_os_str())
        .arg("-o")
        .arg(&out_dir)
        .arg("--out-format")
        .arg("json")
        .status()
        .expect("failed to spawn polyscribe");
    assert!(status.success(), "CLI did not exit successfully");

    let mut has_json = false;
    let mut has_toml = false;
    let mut has_srt = false;
    for e in fs::read_dir(&out_dir).unwrap() {
        let p = e.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with(".json") { has_json = true; }
            if name.ends_with(".toml") { has_toml = true; }
            if name.ends_with(".srt") { has_srt = true; }
        }
    }
    assert!(has_json, "expected JSON file to be written");
    assert!(!has_toml, "did not expect TOML file");
    assert!(!has_srt, "did not expect SRT file");

    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
fn out_format_multiple_json_and_srt() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let out_dir = manifest_path("target/tmp/itest_outfmt_json_srt");
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir).unwrap();

    let input1 = manifest_path("input/2-vikingowl.json");

    let status = Command::new(exe)
        .arg(input1.as_os_str())
        .arg("-o")
        .arg(&out_dir)
        .arg("--out-format")
        .arg("json")
        .arg("--out-format")
        .arg("srt")
        .status()
        .expect("failed to spawn polyscribe");
    assert!(status.success(), "CLI did not exit successfully");

    let mut has_json = false;
    let mut has_toml = false;
    let mut has_srt = false;
    for e in fs::read_dir(&out_dir).unwrap() {
        let p = e.unwrap().path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with(".json") { has_json = true; }
            if name.ends_with(".toml") { has_toml = true; }
            if name.ends_with(".srt") { has_srt = true; }
        }
    }
    assert!(has_json, "expected JSON file to be written");
    assert!(has_srt, "expected SRT file to be written");
    assert!(!has_toml, "did not expect TOML file");

    let _ = fs::remove_dir_all(&out_dir);
}
