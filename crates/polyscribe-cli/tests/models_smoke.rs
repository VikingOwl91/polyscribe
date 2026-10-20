use assert_cmd::cargo::cargo_bin;
use std::process::Command;

fn bin() -> std::path::PathBuf {
    cargo_bin("polyscribe")
}

#[test]
fn models_help_shows_global_output_flags() {
    let out = Command::new(bin())
        .args(["models", "--help"]) // subcommand help
        .output()
        .expect("failed to run polyscribe models --help");
    assert!(
        out.status.success(),
        "help exited non-zero: {:?}",
        out.status
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(
        stdout.contains("--json"),
        "--json not shown in help: {stdout}"
    );
    assert!(
        stdout.contains("--quiet"),
        "--quiet not shown in help: {stdout}"
    );
}

#[test]
fn models_version_contains_pkg_version() {
    let out = Command::new(bin())
        .args(["models", "--version"]) // propagate_version
        .output()
        .expect("failed to run polyscribe models --version");
    assert!(
        out.status.success(),
        "version exited non-zero: {:?}",
        out.status
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    let want = env!("CARGO_PKG_VERSION");
    assert!(
        stdout.contains(want),
        "version output missing {want}: {stdout}"
    );
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

#[test]
fn models_clear_cache_offline_succeeds() {
    // Isolated XDG cache dir: clearing a (non-existent) cache file is a no-op success.
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(bin())
        .args(["models", "clear-cache"])
        .env("XDG_CACHE_HOME", tmp.path().join("cache"))
        .env("XDG_CONFIG_HOME", tmp.path().join("config"))
        .output()
        .expect("failed to run polyscribe models clear-cache");
    assert!(
        out.status.success(),
        "clear-cache exited non-zero: {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(stdout.contains("cleared"), "unexpected stdout: {stdout}");
}

#[test]
fn models_download_no_interaction_skips_prompt() {
    // In non-interactive mode the interactive downloader skips itself and succeeds.
    let out = Command::new(bin())
        .args(["--no-interaction", "models", "download"])
        .output()
        .expect("failed to run polyscribe --no-interaction models download");
    assert!(
        out.status.success(),
        "download --no-interaction exited non-zero: {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}
