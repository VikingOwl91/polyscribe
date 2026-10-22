// SPDX-License-Identifier: MIT

//! The `record` subcommand: transcribe the microphone while it records.
//!
//! The command captures the microphone through PipeWire and hands the audio
//! to the streaming path of the pipeline, so the text of a recording appears
//! while the recording is still running.
//!
//! # The output contract
//!
//! * The standard output carries the transcript: one segment per line as
//!   the segments become final. With `--json` the lines are JSON objects
//!   instead, one per line.
//! * The standard error carries everything else: the readiness of the
//!   capture, the text that is still provisional, and the diagnostics of
//!   the run.
//!
//! A segment that is still provisional may change: the next window decodes
//! the speech at the end of the window again, with the context it needs.
//! Only the finalized segment is final, and it is never reported twice.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};

use polyscribe_core::OutputEntry;
use polyscribe_core::backend::{BackendKind, Language, ModelSession, transcribe_stream};
use polyscribe_core::capture::{CaptureConfig, PcmFormat, PipeWireCapture, list_devices};
use polyscribe_core::stream::{CancellationToken, LiveWindowPolicy, StreamEvent, WindowPolicy};

use crate::cli::GpuBackend;

/// Arguments for the `record` subcommand.
pub struct RecordArgs {
    pub model: Option<String>,
    pub language: Option<String>,
    pub device: Option<String>,
    pub window: Option<f64>,
    pub output: Option<PathBuf>,
    pub gpu_backend: GpuBackend,
    /// Whether the transcript is written as JSON objects.
    pub json: bool,
}

/// The flag that the signal handler sets.
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// The number of stop signals that were received.
///
/// The first one asks the run to stop, the second one ends it at once.
static STOP_SIGNALS: AtomicU32 = AtomicU32::new(0);

/// The process group of the capture.
///
/// The signal handler stops the group when the run ends at once, because
/// the capture is not unwound in that case. Zero means that no capture
/// has been started.
static CAPTURE_GROUP: AtomicU32 = AtomicU32::new(0);

/// The test override for the capture program.
///
/// The tests replace `pw-record` with a program that delivers a known
/// amount of audio, so that they do not need a microphone.
const ENV_CAPTURE_PROGRAM: &str = "POLYSCRIBE_CAPTURE_PROGRAM";

/// How long the command waits for the first audio before it says that the
/// recording started.
const READY_TIMEOUT: Duration = Duration::from_secs(2);

/// The smallest live window that is accepted, in seconds.
///
/// A window below this length does not hold enough speech for whisper to
/// report a segment, so the run would decode the same audio over and over.
const MIN_WINDOW_SECONDS: f64 = 2.0;

pub fn run(args: RecordArgs) -> Result<()> {
    // The backend is selected before the capture starts, so that a missing
    // GPU is reported before the microphone is opened.
    let requested = match args.gpu_backend {
        GpuBackend::Auto => BackendKind::Auto,
        GpuBackend::Cpu => BackendKind::Cpu,
        GpuBackend::Cuda => BackendKind::Cuda,
        GpuBackend::Hip => BackendKind::Hip,
    };
    let selection =
        polyscribe_core::backend::select_backend(requested, polyscribe_core::verbose_level() > 0)?;
    polyscribe_core::dlog!(1, "Using backend: {:?}", selection.chosen);

    let mut session = match &args.model {
        Some(model) => ModelSession::with_model(resolve_model(model)?),
        None => ModelSession::new(),
    };

    let language = match &args.language {
        Some(code) => Language::Explicit(normalize_language(code)?),
        // The language of a recording is detected, so that the caller does
        // not have to know it beforehand.
        None => Language::Auto,
    };

    let policy = match args.window {
        Some(seconds) => {
            if !(seconds.is_finite() && seconds >= MIN_WINDOW_SECONDS) {
                return Err(anyhow!(
                    "the window must be at least {MIN_WINDOW_SECONDS} seconds, got {seconds}"
                ));
            }
            WindowPolicy::Live(LiveWindowPolicy::new(seconds))
        }
        None => WindowPolicy::Live(LiveWindowPolicy::default()),
    };

    let device = resolve_device(args.device.as_deref())?;

    let mut config = CaptureConfig::new()
        .with_device(device.clone())
        .with_program(capture_program());
    config.format = PcmFormat::S16Le;

    STOP_REQUESTED.store(false, Ordering::SeqCst);
    STOP_SIGNALS.store(0, Ordering::SeqCst);
    install_signal_handlers();

    let cancel = CancellationToken::new();
    // The watcher is started before the capture, so that a stop signal
    // that arrives while the capture starts is not lost.
    let _watcher = Watcher::start(cancel.clone());
    let capture = PipeWireCapture::new(config, cancel.clone())?;
    // The group is remembered, so that a run that is ended at once stops
    // the capture instead of leaving it behind.
    if let Some(group) = capture.process_group() {
        CAPTURE_GROUP.store(group, Ordering::SeqCst);
    }

    // The capture is only ready when it delivers audio. Waiting for the
    // first samples tells the user that the microphone works, instead of
    // leaving them in front of a silent command.
    let ready = wait_for_audio(&capture);
    if ready {
        polyscribe_core::ilog!(
            "recording from {} (Ctrl+C stops the recording)",
            device.as_deref().unwrap_or("the default source")
        );
    } else {
        polyscribe_core::ilog!(
            "recording from {} (no audio yet, Ctrl+C stops the recording)",
            device.as_deref().unwrap_or("the default source")
        );
    }

    if stop_requested() {
        cancel.cancel();
        polyscribe_core::ilog!("stopped with 0 finalized segment(s)");
        return Ok(());
    }

    let mut writer = SegmentWriter::new(args.output.clone(), args.json)?;
    let show_provisional = !args.json && !polyscribe_core::is_quiet();
    let mut output_error = None;
    let result = {
        let mut observer = |event: StreamEvent| {
            match event {
                StreamEvent::SegmentProvisional { entry } => {
                    // The text may still change, so it goes to the standard
                    // error and not to the transcript.
                    if show_provisional {
                        eprintln!("… {}", entry.text);
                    }
                }
                StreamEvent::SegmentFinalized { entry } => {
                    if output_error.is_none()
                        && let Err(error) = writer.write(&entry)
                    {
                        output_error = Some(error.context("the transcript could not be written"));
                        cancel.cancel();
                    }
                }
                _ => {}
            }
        };

        // The capture is stopped by its own drop, which happens inside the
        // call above.
        transcribe_stream(
            &mut session,
            selection.chosen,
            Box::new(capture),
            "speaker",
            language,
            policy,
            &mut observer,
            &cancel,
        )
    };
    if let Some(error) = output_error {
        return Err(error);
    }
    let result = result?;
    writer.finish()?;

    match result.outcome {
        polyscribe_core::stream::StreamOutcome::Completed { segments } => {
            polyscribe_core::ilog!("recorded {} segment(s)", segments);
        }
        polyscribe_core::stream::StreamOutcome::Cancelled { segments } => {
            polyscribe_core::ilog!("stopped with {} finalized segment(s)", segments);
        }
    }
    if let Some(language) = &result.detected_language {
        polyscribe_core::dlog!(1, "detected language: {language}");
    }
    Ok(())
}

/// The program that captures the audio.
///
/// The test override replaces `pw-record` with a program that delivers a
/// known amount of audio.
fn capture_program() -> String {
    std::env::var(ENV_CAPTURE_PROGRAM)
        .unwrap_or_else(|_| polyscribe_core::capture::CAPTURE_PROGRAM.to_string())
}

/// Install the handlers that ask the run to stop.
///
/// The handler only sets a flag, which is safe to do in a signal handler.
/// The run notices the flag and stops at the next safe boundary.
fn install_signal_handlers() {
    // SAFETY: the handler sets an atomic flag and nothing else, which is
    // safe to do in a signal handler.
    unsafe {
        libc::signal(libc::SIGINT, handle_stop as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handle_stop as libc::sighandler_t);
    }
}

/// The handler of the stop signals.
///
/// The first signal asks the run to stop at its next safe boundary. The
/// second one ends the run at once: a whisper call that is running is not
/// interrupted, and a user that asks twice means it.
extern "C" fn handle_stop(signal: libc::c_int) {
    if STOP_SIGNALS.fetch_add(1, Ordering::SeqCst) > 0 {
        // The capture is stopped first, because the process ends without
        // unwinding it. `kill` is safe to call from a signal handler.
        let group = CAPTURE_GROUP.load(Ordering::SeqCst);
        if group != 0 {
            // SAFETY: the group id was assigned to the capture child when
            // it was started.
            unsafe { libc::kill(-(group as libc::pid_t), libc::SIGTERM) };
        }
        // SAFETY: the default handler is restored and the signal is sent
        // again, so the process ends the way it would end without a
        // handler. The signal is pending while the handler runs and is
        // delivered when it returns.
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
            let _ = libc::raise(signal);
        }
        return;
    }
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

/// Whether a stop signal was received.
fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

/// The thread that watches for a stop signal.
///
/// The thread is stopped when the guard is dropped, so that it cannot
/// outlive the run, whatever path the run takes.
struct Watcher {
    handle: Option<std::thread::JoinHandle<()>>,
    stop: std::sync::Arc<AtomicBool>,
}

impl Watcher {
    /// Start watching for a stop signal.
    fn start(cancel: CancellationToken) -> Self {
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let thread_stop = std::sync::Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::SeqCst) {
                if cancel.is_cancelled() {
                    return;
                }
                if stop_requested() {
                    cancel.cancel();
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self {
            handle: Some(handle),
            stop,
        }
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Wait for the first audio of the capture.
fn wait_for_audio(capture: &PipeWireCapture) -> bool {
    let deadline = Instant::now() + READY_TIMEOUT;
    while Instant::now() < deadline {
        if stop_requested() {
            // The run was asked to stop while it was starting.
            return false;
        }
        if capture.samples_received() > 0 {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    capture.samples_received() > 0
}

/// The device to record from.
///
/// `pw-record` falls back to the default source when it is given a node
/// that does not exist, and it does not say so. The device is therefore
/// checked against the nodes of the session before the capture starts.
fn resolve_device(requested: Option<&str>) -> Result<Option<String>> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    let devices = match list_devices() {
        Ok(devices) => devices,
        Err(error) => {
            // The nodes cannot be listed, so the device cannot be checked.
            // The run continues, and the risk is named.
            polyscribe_core::ui::warn(format!(
                "the device '{requested}' could not be checked ({error}). pw-record records from the default source when it is given a node that does not exist."
            ));
            return Ok(Some(requested.to_string()));
        }
    };
    if devices.iter().any(|device| device.name == requested) {
        return Ok(Some(requested.to_string()));
    }
    let mut names: Vec<&str> = devices.iter().map(|device| device.name.as_str()).collect();
    names.sort_unstable();
    Err(anyhow!(
        "there is no capture source named '{requested}'. The sources of the session are: {}",
        names.join(", ")
    ))
}

/// The model file for the given name.
///
/// The name is a path to a model file, or the name of a model in the models
/// directory of the platform.
fn resolve_model(name: &str) -> Result<PathBuf> {
    let path = PathBuf::from(name);
    if path.is_file() {
        return Ok(path);
    }
    if name.contains('/') {
        return Err(anyhow!("the model file does not exist: {}", path.display()));
    }
    let in_models_dir = polyscribe_core::models_dir_path().join(name);
    if in_models_dir.is_file() {
        return Ok(in_models_dir);
    }
    if let Ok(paths) = polyscribe_core::model_manager::Paths::resolve() {
        let in_cache = paths.cache_dir.join(name);
        if in_cache.is_file() {
            return Ok(in_cache);
        }
    }
    Err(anyhow!(
        "the model '{name}' was not found. Give the path of a model file, or install one with 'polyscribe models add'."
    ))
}

/// The language code of the recording.
///
/// The code is normalized the way the file path normalizes it, so that the
/// same code works for both.
fn normalize_language(code: &str) -> Result<String> {
    let normalized =
        polyscribe_core::normalize_lang_code(code).unwrap_or_else(|| code.trim().to_lowercase());
    if normalized.is_empty() {
        return Err(anyhow!("the language code is empty"));
    }
    Ok(normalized)
}

/// Writes the segments of a recording as they become final.
struct SegmentWriter {
    stdout: std::io::Stdout,
    file: Option<std::fs::File>,
    /// Whether the lines are JSON objects.
    json: bool,
}

impl SegmentWriter {
    /// A writer for the standard output and, optionally, a file.
    fn new(file: Option<PathBuf>, json: bool) -> Result<Self> {
        let file = match file {
            Some(path) => {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&path)
                    .with_context(|| {
                        format!("the output file could not be opened: {}", path.display())
                    })?;
                Some(file)
            }
            None => None,
        };
        Ok(Self {
            stdout: std::io::stdout(),
            file,
            json,
        })
    }

    /// Write a segment that became final.
    fn write(&mut self, entry: &OutputEntry) -> Result<()> {
        let line = self.line(entry);
        writeln!(self.stdout, "{line}")?;
        self.stdout.flush()?;
        if let Some(file) = self.file.as_mut() {
            writeln!(file, "{line}")?;
            file.flush()?;
        }
        Ok(())
    }

    /// The line of a segment.
    fn line(&self, entry: &OutputEntry) -> String {
        if self.json {
            let value = serde_json::json!({
                "speaker": entry.speaker,
                "start": entry.start,
                "end": entry.end,
                "text": entry.text,
            });
            value.to_string()
        } else {
            entry.text.clone()
        }
    }

    /// Flush what is left.
    fn finish(&mut self) -> Result<()> {
        self.stdout.flush()?;
        if let Some(file) = self.file.as_mut() {
            file.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_language_code_is_normalized() {
        assert_eq!(normalize_language("de").unwrap(), "de");
        assert_eq!(normalize_language(" DE ").unwrap(), "de");
        assert_eq!(normalize_language("german").unwrap(), "de");
    }

    #[test]
    fn an_empty_language_code_is_refused() {
        assert!(normalize_language("  ").is_err());
    }

    #[test]
    fn the_window_is_checked() {
        assert!(LiveWindowPolicy::new(5.0).window_samples() > 0);
    }
}
