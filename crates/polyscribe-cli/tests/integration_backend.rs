// SPDX-License-Identifier: MIT

//! Integration tests for the backend selection contract of the `transcribe`
//! subcommand.
//!
//! These tests run the built binary. It is compiled without the optional GPU
//! features, so an explicit GPU request has to fail with an actionable error
//! instead of silently continuing on the CPU.

use std::path::PathBuf;
use std::process::{Command, Output};

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

fn run_with_backend(backend: &str) -> Output {
    Command::new(bin())
        .args(["--no-interaction", "transcribe"])
        .arg(fixture("sample.mp3"))
        .args(["--language", "en"])
        .arg("--gpu-backend")
        .arg(backend)
        .output()
        .expect("failed to spawn polyscribe")
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn explicit_gpu_backend_without_the_feature_fails_with_guidance() {
    for (backend, expected) in [
        ("hip", "built without HIP support"),
        ("cuda", "built without CUDA support"),
    ] {
        let output = run_with_backend(backend);
        assert!(
            !output.status.success(),
            "expected a non-zero exit for --gpu-backend {backend}"
        );
        let stderr = stderr_of(&output);
        assert!(
            stderr.contains(expected),
            "expected \"{expected}\" for --gpu-backend {backend}, got: {stderr}"
        );
    }
}

#[test]
fn explicit_cpu_backend_is_accepted() {
    let output = run_with_backend("cpu");
    let stderr = stderr_of(&output);
    assert!(
        !stderr.contains("built without"),
        "the CPU backend must be available: {stderr}"
    );
}

#[test]
fn automatic_backend_is_accepted() {
    let output = run_with_backend("auto");
    let stderr = stderr_of(&output);
    assert!(
        !stderr.contains("built without"),
        "the automatic backend must resolve to something: {stderr}"
    );
}
