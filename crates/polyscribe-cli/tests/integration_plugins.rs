// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! End-to-end tests for the plugin system (`polyscribe plugins ...`).
//!
//! These tests run the built `polyscribe-plugin-tubescribe` reference plugin
//! through the CLI's plugin host. They need no network. If the plugin binary
//! is not built (e.g. when testing only this crate with `-p polyscribe-cli`),
//! the tests are skipped with a note; run `cargo test --workspace` to include
//! them.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

/// Locate the reference plugin binary next to the `polyscribe` binary
/// (same target dir, both are workspace members).
fn plugin_bin() -> Option<PathBuf> {
    let polyscribe = PathBuf::from(bin());
    let dir = polyscribe.parent()?;
    let candidate = dir.join("polyscribe-plugin-tubescribe");
    candidate.is_file().then_some(candidate)
}

fn skip_if_no_plugin() -> Option<PathBuf> {
    match plugin_bin() {
        Some(p) => Some(p),
        None => {
            eprintln!(
                "NOT RUN: polyscribe-plugin-tubescribe binary not found next to polyscribe; \
                 run `cargo test --workspace` to include plugin tests"
            );
            None
        }
    }
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
fn plugins_list_finds_tubescribe() {
    let Some(plugin) = skip_if_no_plugin() else {
        return;
    };
    let out = run_with_plugin(&plugin, &["plugins", "list"]);
    assert!(
        out.status.success(),
        "plugins list failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("tubescribe"),
        "expected tubescribe in plugins list output, got: {stdout}"
    );
}

#[test]
fn plugins_info_shows_capabilities() {
    let Some(plugin) = skip_if_no_plugin() else {
        return;
    };
    let out = run_with_plugin(&plugin, &["plugins", "info", "tubescribe"]);
    assert!(
        out.status.success(),
        "plugins info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let caps: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("info output is not capabilities JSON");
    assert_eq!(caps["name"], "tubescribe");
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
    let Some(plugin) = skip_if_no_plugin() else {
        return;
    };
    let out = run_with_plugin(
        &plugin,
        &["plugins", "run", "tubescribe", "generate_metadata"],
    );
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
    let Some(plugin) = skip_if_no_plugin() else {
        return;
    };
    let out = run_with_plugin(&plugin, &["plugins", "run", "tubescribe", "does_not_exist"]);
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
    let Some(plugin) = skip_if_no_plugin() else {
        return;
    };
    // The plugin dir contains only tubescribe.
    let out = run_with_plugin(&plugin, &["plugins", "info", "nope"]);
    assert!(!out.status.success(), "expected failure for unknown plugin");
}
