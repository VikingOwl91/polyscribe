// SPDX-License-Identifier: MIT
// Validation and error-handling integration tests

use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

fn manifest_path(relative: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(relative);
    p
}

#[test]
fn error_on_missing_input_file() {
    let exe = bin();
    let missing = manifest_path("input/definitely_missing_123.json");
    let out = Command::new(exe)
        .arg(missing.as_os_str())
        .output()
        .expect("failed to run polyscribe with missing input");
    assert!(!out.status.success(), "command should fail on missing input");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Input not found") || stderr.contains("No input files provided"),
        "stderr should mention missing input; got: {}",
        stderr
    );
}

#[test]
fn error_on_directory_as_input() {
    let exe = bin();
    // Use the repo's input directory which exists and is a directory
    let input_dir = manifest_path("input");
    let out = Command::new(exe)
        .arg(input_dir.as_os_str())
        .output()
        .expect("failed to run polyscribe with directory input");
    assert!(!out.status.success(), "command should fail on dir input");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("directory") || stderr.contains("Unsupported input type"),
        "stderr should mention directory/unsupported; got: {}",
        stderr
    );
}

#[test]
fn error_on_no_ffmpeg_present() {
    let exe = bin();
    // Create a tiny temp .wav file (may be empty; ffmpeg will be attempted but PATH will be empty)
    let tmp_dir = manifest_path("target/tmp/itest_no_ffmpeg");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();
    let wav = tmp_dir.join("dummy.wav");
    fs::write(&wav, b"\0\0\0\0").unwrap();

    let out = Command::new(exe)
        .arg(wav.as_os_str())
        .env("PATH", "") // simulate ffmpeg missing
        .env_remove("WHISPER_MODEL")
        .env("POLYSCRIBE_MODELS_BASE_COPY_DIR", manifest_path("models").as_os_str())
        .arg("--language").arg("en")
        .output()
        .expect("failed to run polyscribe with empty PATH");
    assert!(
        !out.status.success(),
        "command should fail when ffmpeg is not found"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ffmpeg not found") || stderr.contains("Failed to execute ffmpeg"),
        "stderr should mention ffmpeg not found; got: {}",
        stderr
    );
}

#[cfg(unix)]
#[test]
fn error_on_readonly_output_dir() {
    use std::os::unix::fs::PermissionsExt;

    let exe = bin();
    let input1 = manifest_path("input/1-s0wlz.json");

    // Prepare a read-only directory
    let tmp_dir = manifest_path("target/tmp/itest_readonly_out");
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();
    let mut perms = fs::metadata(&tmp_dir).unwrap().permissions();
    perms.set_mode(0o555); // read & execute, no write
    fs::set_permissions(&tmp_dir, perms).unwrap();

    let out = Command::new(exe)
        .arg(input1.as_os_str())
        .arg("-o")
        .arg(tmp_dir.as_os_str())
        .output()
        .expect("failed to run polyscribe with read-only output dir");

    // Restore perms for cleanup
    let mut perms2 = fs::metadata(&tmp_dir).unwrap().permissions();
    perms2.set_mode(0o755);
    let _ = fs::set_permissions(&tmp_dir, perms2);

    assert!(
        !out.status.success(),
        "command should fail when outputs cannot be created"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Failed to create output") || stderr.contains("permission"),
        "stderr should mention failure to create outputs; got: {}",
        stderr
    );

    // Cleanup
    let _ = fs::remove_dir_all(&tmp_dir);
}
