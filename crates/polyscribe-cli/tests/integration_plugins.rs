// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! End-to-end tests for the plugin system (`polyscribe plugins ...`).
//!
//! These tests run the `polyscribe-plugin-example` reference plugin through
//! the CLI's plugin host. They need no network. The plugin binary is a
//! different package than the CLI, so `cargo test -p polyscribe-cli` does not
//! build it; the tests then build it themselves (once) instead of skipping, and
//! fail with a clear message if that does not work.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

/// Locate the reference plugin binary next to the `polyscribe` binary
/// (same target dir and profile, both are workspace members).
fn find_plugin_bin() -> Option<PathBuf> {
    let dir = PathBuf::from(bin()).parent()?.to_path_buf();
    let candidate = dir.join("polyscribe-plugin-example");
    candidate.is_file().then_some(candidate)
}

/// Build the plugin into the same target dir and profile as this test run.
fn build_plugin() {
    let polyscribe = PathBuf::from(bin());
    let profile_dir = polyscribe.parent().expect("binary has a directory");
    let target_dir = profile_dir.parent().expect("profile dir has a target dir");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.args(["build", "--locked", "-p", "polyscribe-plugin-example"])
        .arg("--target-dir")
        .arg(target_dir);
    if profile_dir
        .file_name()
        .is_some_and(|name| name == "release")
    {
        cmd.arg("--release");
    }
    let output = cmd
        .output()
        .expect("failed to start cargo to build the plugin");
    assert!(
        output.status.success(),
        "building polyscribe-plugin-example failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The reference plugin binary. Built on first use if it is missing; a
/// missing plugin fails the test run instead of silently skipping it.
fn require_plugin() -> PathBuf {
    static PLUGIN: OnceLock<PathBuf> = OnceLock::new();
    PLUGIN
        .get_or_init(|| {
            if let Some(path) = find_plugin_bin() {
                return path;
            }
            build_plugin();
            find_plugin_bin().expect(
                "polyscribe-plugin-example was built but is not next to the polyscribe binary",
            )
        })
        .clone()
}

/// Run the CLI with an isolated PATH (containing the plugin) and XDG dirs.
fn run_with_plugin(plugin: &Path, args: &[&str]) -> std::process::Output {
    let plugin_dir = plugin.parent().unwrap();
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = vec![plugin_dir.to_path_buf()];
    paths.extend(std::env::split_paths(&original_path));
    let new_path = std::env::join_paths(paths).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    Command::new(bin())
        .args(args)
        .env("PATH", new_path)
        .env("XDG_DATA_HOME", tmp.path().join("data"))
        .env("XDG_CACHE_HOME", tmp.path().join("cache"))
        .env("XDG_CONFIG_HOME", tmp.path().join("config"))
        .output()
        .expect("failed to spawn polyscribe")
}

#[test]
fn plugins_list_finds_example() {
    let plugin = require_plugin();
    let out = run_with_plugin(&plugin, &["plugins", "list"]);
    assert!(
        out.status.success(),
        "plugins list failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("example"),
        "expected example in plugins list output, got: {stdout}"
    );
}

#[test]
fn plugins_info_shows_capabilities() {
    let plugin = require_plugin();
    let out = run_with_plugin(&plugin, &["plugins", "info", "example"]);
    assert!(
        out.status.success(),
        "plugins info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let caps: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("info output is not capabilities JSON");
    assert_eq!(caps["name"], "example");
    assert_eq!(caps["protocol"], "psp/1");
    assert!(
        caps["commands"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c == "generate_metadata")
    );
}

#[test]
fn plugins_run_streams_progress_and_result() {
    let plugin = require_plugin();
    let out = run_with_plugin(&plugin, &["plugins", "run", "example", "generate_metadata"]);
    assert!(
        out.status.success(),
        "plugins run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // NDJSON stream: progress events and a final result line.
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(
        lines.len() >= 2,
        "expected streamed NDJSON lines, got: {stdout}"
    );
    let progress: Vec<serde_json::Value> = lines
        .iter()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["type"] == "progress")
        .collect();
    assert!(
        !progress.is_empty(),
        "expected progress events, got: {stdout}"
    );
    let result: Vec<serde_json::Value> = lines
        .iter()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["type"] == "result")
        .collect();
    assert_eq!(
        result.len(),
        1,
        "expected exactly one result line, got: {stdout}"
    );
    assert_eq!(result[0]["result"]["title"], "Canned title");
}

#[test]
fn plugins_run_unknown_method_reports_error_item() {
    let plugin = require_plugin();
    let out = run_with_plugin(&plugin, &["plugins", "run", "example", "does_not_exist"]);
    assert!(
        out.status.success(),
        "plugins run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Method not found"),
        "expected method-not-found error item, got: {stdout}"
    );
}

#[test]
fn plugins_info_unknown_plugin_fails() {
    let plugin = require_plugin();
    // The plugin dir contains only example.
    let out = run_with_plugin(&plugin, &["plugins", "info", "nope"]);
    assert!(!out.status.success(), "expected failure for unknown plugin");
}
