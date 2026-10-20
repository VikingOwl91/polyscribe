// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

use assert_cmd::cargo::cargo_bin;
use std::process::Command;

fn bin() -> std::path::PathBuf {
    cargo_bin("polyscribe")
}

#[test]
fn aux_completions_bash_outputs_script() {
    let out = Command::new(bin())
        .arg("completions")
        .arg("bash")
        .output()
        .expect("failed to run polyscribe completions bash");
    assert!(
        out.status.success(),
        "completions bash exited with failure: {:?}",
        out.status
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(
        !stdout.trim().is_empty(),
        "completions bash stdout is empty"
    );
    // Heuristic: bash completion scripts often contain 'complete -F' lines
    assert!(
        stdout.contains("complete") || stdout.contains("_polyscribe"),
        "bash completion script did not contain expected markers"
    );
}

#[test]
fn aux_completions_zsh_outputs_script() {
    let out = Command::new(bin())
        .arg("completions")
        .arg("zsh")
        .output()
        .expect("failed to run polyscribe completions zsh");
    assert!(
        out.status.success(),
        "completions zsh exited with failure: {:?}",
        out.status
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(!stdout.trim().is_empty(), "completions zsh stdout is empty");
    // Heuristic: zsh completion scripts often start with '#compdef'
    assert!(
        stdout.contains("#compdef") || stdout.contains("#compdef polyscribe"),
        "zsh completion script did not contain expected markers"
    );
}

#[test]
fn aux_man_outputs_roff() {
    let out = Command::new(bin())
        .arg("man")
        .output()
        .expect("failed to run polyscribe man");
    assert!(
        out.status.success(),
        "man exited with failure: {:?}",
        out.status
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(!stdout.trim().is_empty(), "man stdout is empty");
    // clap_mangen typically emits roff with .TH and/or section headers
    let looks_like_roff = stdout.contains(".TH ")
        || stdout.starts_with(".TH")
        || stdout.contains(".SH NAME")
        || stdout.contains(".SH SYNOPSIS");
    assert!(
        looks_like_roff,
        "man output does not look like a roff manpage; got: {}",
        stdout.lines().take(3).collect::<Vec<_>>().join(" | ")
    );
}
