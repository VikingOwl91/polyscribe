// SPDX-License-Identifier: MIT

//! Tests of the live capture.
//!
//! The tests use a program that stands in for `pw-record`: it delivers a
//! known amount of raw PCM on its standard output, writes diagnostics on its
//! standard error and leaves with a chosen exit code. Nothing here needs a
//! microphone or a running PipeWire session.
//!
//! The program reads its settings from a file next to it, so that the tests
//! do not share the environment of the process.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use polyscribe_core::capture::{CaptureConfig, PipeWireCapture};
use polyscribe_core::stream::{AudioSource, CancellationToken};

/// The program that stands in for `pw-record`.
///
/// It is written by the test, so that the tests do not depend on a file
/// that has to stay executable in the repository.
const FAKE_PROGRAM: &str = r#"#!/bin/sh
. "$(dirname "$0")/config"
if [ -n "$FAKE_FORMAT_LINE" ]; then
  printf '%s\n' "$FAKE_FORMAT_LINE" >&2
fi
if [ -n "$FAKE_STDERR" ]; then
  printf '%s\n' "$FAKE_STDERR" >&2
fi
if [ -n "$FAKE_SLEEP" ]; then
  sleep "$FAKE_SLEEP"
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
exit ${FAKE_EXIT:-0}
"#;

/// A fake capture program with its settings.
struct Fake {
    _dir: tempfile::TempDir,
    program: PathBuf,
}

impl Fake {
    /// A program with the given settings.
    ///
    /// The settings are the ones of the program, written as shell
    /// assignments.
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

    /// The path of the program.
    fn program(&self) -> String {
        self.program.display().to_string()
    }
}

/// A value that a shell reads as it is written.
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// The samples of a raw PCM file of the given length.
fn raw_pcm(dir: &Path, name: &str, samples: usize) -> PathBuf {
    let path = dir.join(name);
    let mut bytes = Vec::with_capacity(samples * 2);
    for index in 0..samples {
        let value = ((index % 1000) as i16) - 500;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();
    path
}

/// Read the samples of a capture until it ends.
fn drain(capture: &mut PipeWireCapture) -> Result<Vec<f32>, String> {
    let mut samples = Vec::new();
    while let Some(block) = capture.next_block().map_err(|error| error.to_string())? {
        samples.extend(block.samples);
    }
    Ok(samples)
}

#[test]
fn the_capture_delivers_the_audio_of_the_program() {
    let dir = tempfile::tempdir().unwrap();
    let pcm = raw_pcm(dir.path(), "audio.raw", 8000);
    let fake = Fake::new(&[("FAKE_PCM", &pcm.display().to_string())]);

    let config = CaptureConfig::new().with_program(fake.program());
    let mut capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
    let samples = drain(&mut capture).unwrap();

    assert_eq!(samples.len(), 8000);
    for (index, sample) in samples.iter().enumerate() {
        let expected = ((index % 1000) as i16 - 500) as f32 / 32768.0;
        assert_eq!(
            *sample, expected,
            "the sample {index} is not the one that was sent"
        );
    }
}

#[test]
fn the_order_of_the_samples_is_kept_across_the_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let pcm = raw_pcm(dir.path(), "audio.raw", 40000);
    // A chunk size that does not divide the file, so that the last chunk is
    // short.
    let fake = Fake::new(&[
        ("FAKE_PCM", &pcm.display().to_string()),
        ("FAKE_CHUNK", "7000"),
    ]);

    let config = CaptureConfig::new().with_program(fake.program());
    let mut capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
    let samples = drain(&mut capture).unwrap();

    assert_eq!(samples.len(), 40000);
    for (index, sample) in samples.iter().enumerate() {
        let expected = ((index % 1000) as i16 - 500) as f32 / 32768.0;
        assert_eq!(*sample, expected, "the sample {index} is out of order");
    }
}

#[test]
fn a_capture_that_cannot_start_is_reported() {
    let config = CaptureConfig::new().with_program("/nonexistent/pw-record".to_string());
    let error = match PipeWireCapture::new(config, CancellationToken::new()) {
        Ok(_) => panic!("the capture started with a program that does not exist"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("pw-record"),
        "the error does not name the program: {error}"
    );
}

#[test]
fn a_capture_that_fails_is_reported() {
    let fake = Fake::new(&[("FAKE_EXIT", "3")]);

    let config = CaptureConfig::new().with_program(fake.program());
    let mut capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
    let error = match drain(&mut capture) {
        Ok(_) => panic!("the failing capture was not noticed"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("exit code 3"),
        "unexpected error: {message}"
    );
}

#[test]
fn the_diagnostics_of_the_capture_are_kept() {
    let fake = Fake::new(&[
        ("FAKE_EXIT", "2"),
        ("FAKE_STDERR", "the microphone is busy"),
    ]);

    let config = CaptureConfig::new().with_program(fake.program());
    let mut capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
    let error = match drain(&mut capture) {
        Ok(_) => panic!("the failing capture was not noticed"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("the microphone is busy"),
        "the diagnostics of the capture are missing: {message}"
    );
}

#[test]
fn a_capture_with_the_wrong_format_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let pcm = raw_pcm(dir.path(), "audio.raw", 4000);
    // The program claims to deliver 48 kHz, the pipeline works with 16 kHz.
    let fake = Fake::new(&[
        ("FAKE_PCM", &pcm.display().to_string()),
        (
            "FAKE_FORMAT_LINE",
            "raw: rate=48000 channels=1 fmt=s16 samplesize=2 stride=2",
        ),
    ]);

    let config = CaptureConfig::new().with_program(fake.program());
    let mut capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
    let error = match drain(&mut capture) {
        Ok(_) => panic!("the capture with the wrong format was not noticed"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("48000 Hz"),
        "the error does not name the rate: {message}"
    );
}

#[test]
fn a_capture_with_the_right_format_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let pcm = raw_pcm(dir.path(), "audio.raw", 4000);
    let fake = Fake::new(&[
        ("FAKE_PCM", &pcm.display().to_string()),
        (
            "FAKE_FORMAT_LINE",
            "raw: rate=16000 channels=1 fmt=s16 samplesize=2 stride=2",
        ),
    ]);

    let config = CaptureConfig::new().with_program(fake.program());
    let mut capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
    let samples = drain(&mut capture).unwrap();
    assert_eq!(samples.len(), 4000);
}

#[test]
fn a_cancelled_capture_stops_without_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let pcm = raw_pcm(dir.path(), "audio.raw", 400000);
    let fake = Fake::new(&[
        ("FAKE_PCM", &pcm.display().to_string()),
        ("FAKE_CHUNK", "32000"),
        ("FAKE_DELAY", "0.05"),
    ]);

    let cancel = CancellationToken::new();
    let config = CaptureConfig::new().with_program(fake.program());
    let mut capture = PipeWireCapture::new(config, cancel.clone()).unwrap();
    // Let the first blocks arrive, then stop the run.
    let mut samples = 0usize;
    while samples < 1000 {
        let block = capture
            .next_block()
            .unwrap()
            .expect("the capture delivers audio");
        samples += block.samples.len();
    }
    cancel.cancel();
    // The capture notices the cancellation and does not report an error.
    let block = capture.next_block().unwrap();
    assert!(
        block.is_some(),
        "the capture ended instead of being cancelled"
    );
}

#[test]
fn the_capture_leaves_no_child_behind() {
    let dir = tempfile::tempdir().unwrap();
    let pcm = raw_pcm(dir.path(), "audio.raw", 400000);
    let fake = Fake::new(&[
        ("FAKE_PCM", &pcm.display().to_string()),
        ("FAKE_CHUNK", "32000"),
        ("FAKE_DELAY", "0.05"),
    ]);

    let pid = {
        let config = CaptureConfig::new().with_program(fake.program());
        let mut capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
        let pid = capture.child_pid().expect("the capture has a child");
        // Let the child deliver something, so that it is really running.
        let _ = capture.next_block().unwrap();
        pid
    };
    // The capture was dropped, so the child was stopped and reaped.
    assert!(
        !process_alive(pid),
        "the capture program {pid} is still running"
    );
}

#[test]
fn the_buffer_of_the_capture_is_bounded() {
    let dir = tempfile::tempdir().unwrap();
    // A minute of audio: the capture must not hold it.
    let pcm = raw_pcm(dir.path(), "audio.raw", 60 * 16_000);
    let fake = Fake::new(&[
        ("FAKE_PCM", &pcm.display().to_string()),
        ("FAKE_CHUNK", "32000"),
    ]);

    let config = CaptureConfig::new().with_program(fake.program());
    let capture = PipeWireCapture::new(config, CancellationToken::new()).unwrap();
    // Let the reader thread fill its queue and the pipe of the child.
    std::thread::sleep(Duration::from_millis(300));
    let delivered = capture.samples_received();
    let total = 60 * 16_000u64;
    assert!(
        delivered < total / 4,
        "the capture read {delivered} of {total} samples into its memory"
    );
}

/// Whether a process is still running.
fn process_alive(pid: u32) -> bool {
    // SAFETY: the pid names a process that this test started, and the call
    // only asks whether it exists.
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if result == 0 {
        return true;
    }
    // The process is gone, or it belongs to somebody else and the call was
    // refused with a permission error.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}
