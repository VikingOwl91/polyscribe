// SPDX-License-Identifier: MIT

//! End-to-end tests of the `record` command.
//!
//! The tests replace the capture program with one that delivers a known
//! amount of raw PCM, so that they need neither a microphone nor a running
//! PipeWire session. The tests that transcribe need a Whisper model, so
//! they skip themselves without one, like the other end-to-end tests.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use assert_cmd::cargo::cargo_bin;

#[cfg(unix)]
use std::io::BufRead;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

/// The program that stands in for `pw-record`.
const FAKE_PROGRAM: &str = r#"#!/bin/sh
. "$(dirname "$0")/config"
if [ "${FAKE_SKIP_FORMAT:-}" != "1" ]; then
  printf '%s\n' "${FAKE_FORMAT_LINE:-raw: rate=16000 channels=1 fmt=s16 samplesize=2 stride=2}" >&2
fi
if [ -n "$FAKE_STDERR" ]; then
  printf '%s\n' "$FAKE_STDERR" >&2
fi
if [ -n "$FAKE_PID_FILE" ]; then
  printf '%s\n' "$$" > "$FAKE_PID_FILE"
fi
if [ "${FAKE_IGNORE_TERM:-}" = "1" ]; then
  trap '' TERM
fi
if [ -n "$FAKE_PCM" ] && [ -f "$FAKE_PCM" ]; then
  chunk=${FAKE_CHUNK:-32000}
  size=$(wc -c < "$FAKE_PCM")
  index=0
  while [ $((index * chunk)) -lt "$size" ]; do
    dd if="$FAKE_PCM" bs="$chunk" skip=$index count=1 2>/dev/null
    index=$((index + 1))
    if [ -n "$FAKE_DELAY" ]; then
      sleep "$FAKE_DELAY"
    fi
  done
fi
if [ -n "$FAKE_SLEEP" ]; then
  sleep "$FAKE_SLEEP"
fi
exit ${FAKE_EXIT:-0}
"#;

/// The environment variable that points the command at the capture
/// program.
const ENV_CAPTURE_PROGRAM: &str = "POLYSCRIBE_CAPTURE_PROGRAM";

fn bin() -> PathBuf {
    cargo_bin("polyscribe")
}

/// The model the tests transcribe with.
fn e2e_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("POLYSCRIBE_E2E_MODEL").ok()?);
    path.is_file().then_some(path)
}

/// A fake capture program with its settings.
struct Fake {
    _dir: tempfile::TempDir,
    program: PathBuf,
}

impl Fake {
    fn new(settings: &[(&str, &str)]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let program = dir.path().join("fake-pw-record");
        let mut file = std::fs::File::create(&program).unwrap();
        file.write_all(FAKE_PROGRAM.as_bytes()).unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let mut config = String::new();
        for (key, value) in settings {
            config.push_str(&format!("{key}={}\n", shell_quote(value)));
        }
        std::fs::write(dir.path().join("config"), config).unwrap();
        Self { _dir: dir, program }
    }

    fn program(&self) -> String {
        self.program.display().to_string()
    }
}

/// A value that a shell reads as it is written.
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// The raw PCM of the speech fixture, repeated the given number of times.
fn speech_pcm(dir: &Path, repeats: usize) -> PathBuf {
    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("speech.wav");
    let bytes = std::fs::read(&wav).unwrap();
    // The fixture is a WAV file with a 44 byte header.
    let mut pcm = Vec::new();
    for _ in 0..repeats {
        pcm.extend_from_slice(&bytes[44..]);
    }
    let path = dir.join("speech.raw");
    std::fs::write(&path, pcm).unwrap();
    path
}

/// The command, with the capture program replaced.
fn command(fake: &Fake) -> Command {
    let mut cmd = Command::new(bin());
    cmd.env(ENV_CAPTURE_PROGRAM, fake.program());
    cmd
}

/// A capture that fails is reported with a nonzero status.
///
/// The command needs a model, and the model is looked up when the pipeline
/// starts, so the test names one. Without a model the run would fail with
/// the model error instead of the error of the capture.
#[test]
fn a_failing_capture_is_reported_with_a_nonzero_status() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let fake = Fake::new(&[
        ("FAKE_EXIT", "4"),
        ("FAKE_STDERR", "the microphone is gone"),
    ]);
    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .output()
        .expect("the command runs");
    assert!(
        !output.status.success(),
        "the failing capture was reported as a success: {output:?}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("the microphone is gone"),
        "the diagnostics of the capture are missing: {stderr}"
    );
}

#[test]
fn a_capture_without_a_program_is_reported() {
    let mut cmd = Command::new(bin());
    cmd.env(ENV_CAPTURE_PROGRAM, "/nonexistent/pw-record");
    let output = cmd.arg("record").output().expect("the command runs");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("pw-record"),
        "the error does not name the program: {stderr}"
    );
}

#[test]
fn the_transcript_goes_to_stdout_and_the_status_to_stderr() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 1);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);

    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("en")
        .output()
        .expect("the command runs");
    assert!(
        output.status.success(),
        "the command failed: {:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.trim().is_empty(),
        "the transcript is empty\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Every line of the standard output is a segment of the transcript and
    // nothing else: the status messages belong to the standard error.
    for line in stdout.lines() {
        assert!(
            !line.starts_with('●') && !line.contains("recording from"),
            "the standard output carries a status message: {line}"
        );
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("recording from"),
        "the readiness of the capture is missing: {stderr}"
    );
}

#[test]
fn the_transcript_is_written_to_the_output_file() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 1);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);
    let transcript = dir.path().join("transcript.txt");

    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("en")
        .arg("--output")
        .arg(&transcript)
        .output()
        .expect("the command runs");
    assert!(
        output.status.success(),
        "the command failed: {:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let written = std::fs::read_to_string(&transcript).unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(written, stdout, "the file and the standard output differ");
}

#[test]
fn the_transcript_output_file_replaces_existing_contents() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 1);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);
    let transcript = dir.path().join("transcript.txt");
    std::fs::write(&transcript, "previous recording\n").unwrap();

    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("en")
        .arg("--output")
        .arg(&transcript)
        .output()
        .expect("the command runs");
    assert!(
        output.status.success(),
        "the command failed: {:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let written = std::fs::read_to_string(&transcript).unwrap();
    assert_eq!(
        written,
        String::from_utf8_lossy(&output.stdout),
        "the existing output file was not replaced with this run's transcript"
    );
}

#[test]
fn the_json_output_holds_one_object_per_line() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 1);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);

    let output = command(&fake)
        .arg("--json")
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("en")
        .output()
        .expect("the command runs");
    assert!(
        output.status.success(),
        "the command failed: {:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = 0;
    for line in stdout.lines() {
        let value: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|error| panic!("not JSON: {line} ({error})"));
        assert!(
            value.get("text").is_some(),
            "the object has no text: {line}"
        );
        assert!(
            value.get("start").is_some(),
            "the object has no start: {line}"
        );
        lines += 1;
    }
    assert!(lines > 0, "the transcript is empty");
    assert!(
        output.stderr.is_empty(),
        "JSON mode wrote status or provisional text to stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn an_english_only_model_uses_english_without_a_language_flag() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let english_only = model
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains(".en.") || name.ends_with(".en.bin"));
    if !english_only {
        eprintln!("SKIPPED: the configured end-to-end model is multilingual");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 1);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);
    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .output()
        .expect("the command runs");

    assert!(
        output.status.success(),
        "the command failed: {:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let transcript = String::from_utf8_lossy(&output.stdout).to_lowercase();
    assert!(
        transcript.contains("short test recording"),
        "the English-only model did not transcribe the English fixture: {transcript}"
    );
}

#[test]
fn an_english_only_model_rejects_a_non_english_language_hint() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let english_only = model
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains(".en.") || name.ends_with(".en.bin"));
    if !english_only {
        eprintln!("SKIPPED: the configured end-to-end model is multilingual");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 1);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);
    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("de")
        .output()
        .expect("the command runs");

    assert!(!output.status.success(), "the language hint was accepted");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("English-only") && stderr.contains("non-English"),
        "the unsupported language hint was not explained: {stderr}"
    );
}

#[test]
fn silent_language_startup_reports_the_audio_duration_not_the_window_count() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let english_only = model
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains(".en.") || name.ends_with(".en.bin"));
    if english_only {
        eprintln!("SKIPPED: the configured end-to-end model is English-only");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let silence = dir.path().join("silence.raw");
    std::fs::write(&silence, vec![0u8; 30 * 16_000 * 2]).unwrap();
    let fake = Fake::new(&[("FAKE_PCM", &silence.display().to_string())]);
    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--window")
        .arg("5")
        .output()
        .expect("the command runs");

    assert!(
        output.status.success(),
        "the command failed: {:?}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "silent audio produced transcript text: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("first 30 s of audio"),
        "the warning did not report the 30 seconds of silent startup audio: {stderr}"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_transcript_write_failure_is_reported() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 1);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);

    let output = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("en")
        .arg("--output")
        .arg("/dev/full")
        .output()
        .expect("the command runs");
    assert!(
        !output.status.success(),
        "a failed transcript write was reported as a success"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No space left on device"),
        "the output failure is missing from stderr: {stderr}"
    );
}

#[test]
fn the_transcript_grows_before_the_capture_ends() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let pcm = speech_pcm(dir.path(), 3);
    // The audio is delivered in chunks with a pause between them, so that
    // the recording is still running while the first segments are already
    // transcribed.
    let fake = Fake::new(&[
        ("FAKE_PCM", &pcm.display().to_string()),
        ("FAKE_CHUNK", "32000"),
        ("FAKE_DELAY", "1"),
    ]);

    let mut child = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("en")
        .arg("--window")
        .arg("2")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the command starts");

    // The diagnostics are read in a thread, so that a failure can name
    // them.
    let stderr = child.stderr.take().expect("the error output is piped");
    let diagnostics = std::thread::spawn(move || {
        let mut text = String::new();
        let mut stderr = stderr;
        let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
        text
    });

    // Wait for the first line of the transcript.
    let stdout = child.stdout.take().expect("the output is piped");
    let reader = std::io::BufReader::new(stdout);
    let mut lines = std::io::BufRead::lines(reader);
    let first = match lines.next() {
        Some(Ok(line)) => line,
        other => {
            let _ = child.kill();
            let _ = child.wait();
            let text = diagnostics.join().unwrap_or_default();
            panic!("the transcript has no line: {other:?}\n{text}");
        }
    };
    assert!(!first.trim().is_empty(), "the transcript line is empty");

    // The capture is still running: the program has not delivered all the
    // audio yet.
    let status = child.try_wait().expect("the status is readable");
    assert!(
        status.is_none(),
        "the recording ended before the first segment was written"
    );

    // Stop the run.
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn an_unknown_device_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let capture_started = dir.path().join("capture-started");
    let fake = Fake::new(&[("FAKE_PID_FILE", &capture_started.display().to_string())]);
    let pw_cli = dir.path().join("pw-cli");
    std::fs::write(
        &pw_cli,
        r#"#!/bin/sh
printf 'id 42, type PipeWire:Interface:Node\n\tnode.name = "test_source"\n\tnode.description = "Test source"\n\tmedia.class = "Audio/Source"\n'
"#,
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&pw_cli, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let mut paths = vec![dir.path().to_path_buf()];
    if let Some(path) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&path));
    }
    let path = std::env::join_paths(paths).unwrap();
    let output = command(&fake)
        .arg("record")
        .arg("--device")
        .arg("no_such_node_in_this_session")
        .env("PATH", path)
        .output()
        .expect("the command runs");
    assert!(!output.status.success(), "the unknown device was accepted");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("there is no capture source named 'no_such_node_in_this_session'"),
        "the command did not report an unknown source: {stderr}"
    );
    assert!(
        stderr.contains("test_source"),
        "the source list is missing: {stderr}"
    );
    assert!(
        !capture_started.exists(),
        "the capture program started for an unknown source"
    );
}

/// A second stop signal ends the run at once.
///
/// The first signal asks the run to stop at its next safe boundary. A
/// whisper call that is running is not interrupted, so a user that asks
/// twice waits for nothing: the run ends.
#[cfg(unix)]
#[test]
fn a_second_stop_signal_ends_the_run_at_once() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the record tests");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("pid");
    // The capture ignores the stop signal of the group and stays open, so
    // that the run is still busy when the second signal arrives.
    let fake = Fake::new(&[
        ("FAKE_IGNORE_TERM", "1"),
        ("FAKE_SLEEP", "30"),
        ("FAKE_PID_FILE", &pid_file.display().to_string()),
    ]);

    let mut child = command(&fake)
        .arg("record")
        .arg("--model")
        .arg(&model)
        .arg("--language")
        .arg("en")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the command starts");

    let stderr = child.stderr.take().expect("the error output is piped");
    let mut lines = std::io::BufReader::new(stderr).lines();
    let mut ready = false;
    for line in lines.by_ref() {
        if line.unwrap_or_default().contains("recording from") {
            ready = true;
            break;
        }
    }
    assert!(ready, "the run never reported that it was recording");
    let diagnostics = std::thread::spawn(move || {
        let mut text = String::new();
        for line in lines {
            text.push_str(&line.unwrap_or_default());
            text.push('\n');
        }
        text
    });

    // The first signal asks the run to stop, the second one ends it.
    let pid = child.id() as libc::pid_t;
    // SAFETY: the signal is sent to the process that was started here.
    unsafe { libc::kill(pid, libc::SIGINT) };
    std::thread::sleep(std::time::Duration::from_millis(300));
    // The capture ignores the stop signal of its group, so the first
    // signal cannot end the run: it waits for the capture to stop.
    assert!(
        child.try_wait().expect("the status is readable").is_none(),
        "the first signal ended the run"
    );
    // SAFETY: the signal is sent to the process that was started here.
    unsafe { libc::kill(pid, libc::SIGINT) };

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let status = loop {
        match child.try_wait().expect("the status is readable") {
            Some(status) => break status,
            None => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("the run did not end after the second signal");
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    };
    let text = diagnostics.join().unwrap_or_default();

    // The capture is left behind by the forced exit, so it is stopped here.
    // The capture runs in its own process group, so the group is stopped
    // and not only the program itself.
    if let Ok(pid) = std::fs::read_to_string(&pid_file)
        && let Ok(pid) = pid.trim().parse::<libc::pid_t>()
    {
        // SAFETY: the signal is sent to the group of the capture that was
        // started here.
        unsafe { libc::kill(-pid, libc::SIGKILL) };
    }

    assert_eq!(
        status.signal(),
        Some(libc::SIGINT),
        "the run was not ended by the second signal: {status:?}\n{text}"
    );
}
