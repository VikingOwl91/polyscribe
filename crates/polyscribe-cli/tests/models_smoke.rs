use assert_cmd::cargo::cargo_bin;
use std::process::Command;

fn bin() -> std::path::PathBuf { cargo_bin("polyscribe") }

#[test]
fn models_help_shows_global_output_flags() {
    let out = Command::new(bin())
        .args(["models", "--help"]) // subcommand help
        .output()
        .expect("failed to run polyscribe models --help");
    assert!(out.status.success(), "help exited non-zero: {:?}", out.status);
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(stdout.contains("--json"), "--json not shown in help: {stdout}");
    assert!(stdout.contains("--quiet"), "--quiet not shown in help: {stdout}");
}

#[test]
fn models_version_contains_pkg_version() {
    let out = Command::new(bin())
        .args(["models", "--version"]) // propagate_version
        .output()
        .expect("failed to run polyscribe models --version");
    assert!(out.status.success(), "version exited non-zero: {:?}", out.status);
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    let want = env!("CARGO_PKG_VERSION");
    assert!(stdout.contains(want), "version output missing {want}: {stdout}");
}

#[test]
fn models_ls_json_quiet_emits_pure_json() {
    let out = Command::new(bin())
        .args(["models", "ls", "--json", "--quiet"]) // global flags
        .output()
        .expect("failed to run polyscribe models ls --json --quiet");
    assert!(out.status.success(), "ls exited non-zero: {:?}", out.status);
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    serde_json::from_str::<serde_json::Value>(stdout.trim()).expect("stdout is not valid JSON");
    // Expect no extra logs on stdout; stderr should be empty in success path
    assert!(out.stderr.is_empty(), "expected no stderr noise");
}

