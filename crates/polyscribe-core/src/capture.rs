// SPDX-License-Identifier: MIT

//! Live microphone capture from PipeWire.
//!
//! The capture runs `pw-record` in raw mode: the samples of the microphone
//! arrive on the standard output of the child, without a container and
//! without a file. They are handed to the transcription pipeline through
//! the [`AudioSource`](crate::stream::AudioSource) contract of the streaming
//! path, so a recording is transcribed while it is still running.
//!
//! # Why a subprocess and not a native binding
//!
//! `pw-record` is the tool of PipeWire itself, so it talks to the session
//! the same way the rest of the desktop does. Its raw mode delivers exactly
//! what the pipeline consumes, so a native binding would add a dependency
//! without changing the result.
//!
//! # The lifecycle of the capture
//!
//! * The child writes the audio to its standard output and its diagnostics
//!   to its standard error. The two are read by different threads, so the
//!   capture cannot deadlock on a full pipe.
//! * The samples are decoded by the reader thread and handed to the pipeline
//!   through a queue with a fixed number of places. When the queue is full
//!   the reader waits, the pipe of the child fills up and the child waits:
//!   the memory of the capture does not depend on the length of the
//!   recording.
//! * The capture's process group is asked to stop with `SIGTERM` and killed
//!   when it does not stop in time. The direct child is reaped. Linux sends
//!   it a catchable `SIGTERM` when the thread that created it exits; this is
//!   best effort and does not apply to descendants.
//!
//! # What a stop means
//!
//! `pw-record` reports a stop by signal as a failed run: it leaves with the
//! exit code 1. The capture therefore does not judge the exit code of the
//! child on its own. It knows whether it asked the child to stop, and only
//! a child that failed without being asked to stop is an error.

use std::io::{BufRead, BufReader, Read as _};
use std::process::{Child, ChildStderr, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use std::os::unix::process::CommandExt;

use crate::stream::{
    AudioBlock, AudioSource, CancellationToken, STREAM_CHANNELS, STREAM_SAMPLE_RATE,
};

/// The program that captures the audio.
pub const CAPTURE_PROGRAM: &str = "pw-record";

/// The program that lists the capture sources of the session.
pub const DEVICE_PROGRAM: &str = "pw-cli";

/// The number of samples a block of the capture holds.
///
/// The block is the unit the pipeline pulls, so it decides how often the
/// pipeline looks at the queue.
const BLOCK_SAMPLES: usize = 4096;

/// The number of blocks that may wait in the queue.
///
/// The queue is the buffer of the capture. It holds a fixed number of
/// blocks, so the memory of the capture does not grow with the length of
/// the recording.
const QUEUE_BLOCKS: usize = 8;

/// The number of bytes of the diagnostics of the child that are kept.
///
/// The diagnostics are only needed when something went wrong, and a long
/// recording produces a lot of them, so only the tail is kept.
const STDERR_KEEP_BYTES: usize = 8 * 1024;

/// How long the capture waits for audio before it reports that none came.
///
/// The wait bounds the time the pipeline needs to notice a cancellation.
const READ_POLL: Duration = Duration::from_millis(50);

/// How long the child gets to stop before it is killed.
const STOP_GRACE: Duration = Duration::from_secs(2);

/// How long the capture waits for `pw-record` to report its raw format.
const FORMAT_REPORT_TIMEOUT: Duration = Duration::from_secs(2);

/// The raw PCM format the capture delivers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    /// Signed 16 bit, little endian.
    S16Le,
    /// Signed 32 bit float, little endian.
    F32Le,
}

impl PcmFormat {
    /// The name `pw-record` knows the format under.
    pub fn pw_record_name(&self) -> &'static str {
        match self {
            PcmFormat::S16Le => "s16",
            PcmFormat::F32Le => "f32",
        }
    }

    /// The number of bytes a sample holds.
    pub fn sample_bytes(&self) -> usize {
        match self {
            PcmFormat::S16Le => 2,
            PcmFormat::F32Le => 4,
        }
    }
}

/// Turns the raw PCM of the capture into the samples of the pipeline.
///
/// The capture delivers the audio in chunks whose size it does not
/// control, so a sample can be split across two chunks. The decoder keeps
/// the bytes of the incomplete sample and continues with them when the
/// rest arrives, so no sample is lost and none is counted twice.
#[derive(Debug, Clone)]
pub struct RawPcmDecoder {
    format: PcmFormat,
    /// The bytes of a sample that are not complete yet.
    pending: Vec<u8>,
}

impl RawPcmDecoder {
    /// A decoder for the given format.
    pub fn new(format: PcmFormat) -> Self {
        Self {
            format,
            pending: Vec::with_capacity(format.sample_bytes()),
        }
    }

    /// The format the decoder expects.
    pub fn format(&self) -> PcmFormat {
        self.format
    }

    /// Decode the bytes that arrived.
    ///
    /// The samples are returned in the order they arrived. A trailing
    /// partial sample is kept for the next call.
    pub fn decode(&mut self, bytes: &[u8]) -> Vec<f32> {
        let width = self.format.sample_bytes();
        let mut samples = Vec::new();
        let mut rest = bytes;

        if !self.pending.is_empty() {
            let needed = width - self.pending.len();
            if rest.len() < needed {
                self.pending.extend_from_slice(rest);
                return samples;
            }
            self.pending.extend_from_slice(&rest[..needed]);
            rest = &rest[needed..];
            if let Some(sample) = decode_one(self.format, &self.pending) {
                samples.push(sample);
            }
            self.pending.clear();
        }

        let complete = rest.len() - (rest.len() % width);
        samples.reserve(complete / width);
        match self.format {
            PcmFormat::S16Le => {
                for chunk in rest[..complete].chunks_exact(2) {
                    samples.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0);
                }
            }
            PcmFormat::F32Le => {
                for chunk in rest[..complete].chunks_exact(4) {
                    samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
        }
        self.pending.extend_from_slice(&rest[complete..]);
        samples
    }
}

/// The value of one sample of the given format.
fn decode_one(format: PcmFormat, bytes: &[u8]) -> Option<f32> {
    match format {
        PcmFormat::S16Le if bytes.len() >= 2 => {
            Some(i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
        }
        PcmFormat::F32Le if bytes.len() >= 4 => {
            Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        _ => None,
    }
}

/// How the capture is started.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// The program that captures the audio.
    pub program: String,
    /// The PipeWire node to record from. `None` uses the default source.
    pub device: Option<String>,
    /// The format of the raw PCM.
    pub format: PcmFormat,
    /// The sample rate of the delivered samples.
    pub sample_rate: u32,
    /// The number of channels of the delivered samples.
    pub channels: u16,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            program: CAPTURE_PROGRAM.to_string(),
            device: None,
            format: PcmFormat::S16Le,
            sample_rate: STREAM_SAMPLE_RATE,
            channels: STREAM_CHANNELS,
        }
    }
}

impl CaptureConfig {
    /// A configuration that records from the default source.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record from the given node.
    pub fn with_device(mut self, device: Option<String>) -> Self {
        self.device = device;
        self
    }

    /// Capture with the given program.
    pub fn with_program(mut self, program: String) -> Self {
        self.program = program;
        self
    }

    /// The format of the raw PCM.
    pub fn with_format(mut self, format: PcmFormat) -> Self {
        self.format = format;
        self
    }
}

/// What the reader thread reports to the pipeline.
#[derive(Debug)]
enum CaptureEvent {
    /// A block of decoded samples.
    Audio(Vec<f32>),
    /// The capture failed.
    Failed(String),
}

/// The format `pw-record` reported for its raw output.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ReportedFormat {
    rate: u32,
    channels: u16,
    format: String,
    sample_size: usize,
}

#[derive(Debug, Default)]
struct ReportedFormatState {
    format: Option<ReportedFormat>,
    stderr_finished: bool,
}

/// The tail of the diagnostics of the child.
#[derive(Debug, Default)]
struct StderrTail {
    bytes: Vec<u8>,
    /// Whether bytes were dropped because the tail was full.
    truncated: bool,
}

impl StderrTail {
    /// Add a line and drop the oldest bytes when the tail is full.
    fn push(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
        if self.bytes.len() > STDERR_KEEP_BYTES {
            let excess = self.bytes.len() - STDERR_KEEP_BYTES;
            self.bytes.drain(..excess);
            self.truncated = true;
        }
    }

    /// The tail as text.
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

/// A live capture from a PipeWire source.
///
/// The capture implements [`AudioSource`], so the streaming path of the
/// pipeline can transcribe it while the recording is still running.
pub struct PipeWireCapture {
    config: CaptureConfig,
    child: Option<Child>,
    events: Option<Receiver<CaptureEvent>>,
    /// The diagnostics of the child, bounded.
    stderr: Arc<Mutex<StderrTail>>,
    /// The format the child reported for its raw output.
    reported: Arc<(Mutex<ReportedFormatState>, Condvar)>,
    /// Whether the capture was asked to stop.
    stop_requested: Arc<AtomicBool>,
    /// The token that asks the run to stop.
    cancel: CancellationToken,
    /// The number of samples that were read from the child.
    delivered: Arc<AtomicU64>,
    reader: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    /// Whether the end of the capture was reported already.
    finished: bool,
}

impl PipeWireCapture {
    /// Start a capture.
    ///
    /// The child is started in its own process group, so that a stop signal
    /// that reaches the terminal reaches the run first and the capture is
    /// stopped by the run, in a defined order. Linux sends the direct child
    /// a catchable `SIGTERM` when the thread that created it exits. The signal
    /// is best effort and is not inherited by descendants.
    pub fn new(config: CaptureConfig, cancel: CancellationToken) -> Result<Self> {
        let mut command = Command::new(&config.program);
        command
            // The verbose mode reports the format of the raw output, which
            // is checked against the format that was asked for.
            .arg("--verbose")
            .arg("--raw")
            .arg("--format")
            .arg(config.format.pw_record_name())
            .arg("--rate")
            .arg(config.sample_rate.to_string())
            .arg("--channels")
            .arg(config.channels.to_string());
        if let Some(device) = &config.device {
            command.arg("--target").arg(device);
        }
        // The raw PCM goes to the standard output, so the capture is a
        // stream and not a file.
        command
            .arg("-")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // SAFETY: the child is put into its own process group and told to
        // stop when the parent dies. Both are a single system call, which
        // is safe between the fork and the exec.
        let parent_pid = unsafe { libc::getpid() };
        unsafe {
            command.process_group(0);
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                // The parent can die between the fork and this call, and
                // then the signal is never sent. The check catches that
                // case, even if a subreaper adopts the child instead of
                // PID 1.
                if libc::getppid() != parent_pid {
                    libc::_exit(1);
                }
                Ok(())
            });
        }

        let mut child = command.spawn().map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                anyhow!(
                    "the audio capture program '{}' was not found. Install the PipeWire tools (package pipewire-audio) and try again.",
                    config.program
                )
            } else {
                anyhow!(
                    "the audio capture program '{}' could not be started: {error}",
                    config.program
                )
            }
        })?;

        let stdout = child
            .stdout
            .take()
            .expect("the capture was started with a piped standard output");
        let stderr = child
            .stderr
            .take()
            .expect("the capture was started with a piped standard error");

        let (sender, receiver) = sync_channel::<CaptureEvent>(QUEUE_BLOCKS);
        let delivered = Arc::new(AtomicU64::new(0));
        let reader = spawn_reader(stdout, sender, config.format, Arc::clone(&delivered));
        let stderr_tail = Arc::new(Mutex::new(StderrTail::default()));
        let reported = Arc::new((Mutex::new(ReportedFormatState::default()), Condvar::new()));
        let stderr_thread =
            spawn_stderr_reader(stderr, Arc::clone(&stderr_tail), Arc::clone(&reported));

        Ok(Self {
            config,
            child: Some(child),
            events: Some(receiver),
            stderr: stderr_tail,
            reported,
            stop_requested: Arc::new(AtomicBool::new(false)),
            cancel,
            delivered,
            reader: Some(reader),
            stderr_thread: Some(stderr_thread),
            finished: false,
        })
    }

    /// The configuration the capture runs with.
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }

    /// The number of samples that were read from the capture.
    ///
    /// The counter tells whether the microphone delivers anything at all,
    /// which a caller needs before it tells the user that the recording
    /// started.
    pub fn samples_received(&self) -> u64 {
        self.delivered.load(Ordering::Relaxed)
    }

    /// The process id of the capture program.
    pub fn child_pid(&self) -> Option<u32> {
        self.child.as_ref().map(|child| child.id())
    }

    /// The process group of the capture.
    ///
    /// The child was started in its own process group, so the group id is
    /// the id of the child. It is `None` when the capture has no child any
    /// more. A caller that ends the run without unwinding the capture uses
    /// it to stop the group.
    pub fn process_group(&self) -> Option<u32> {
        self.child.as_ref().map(|child| child.id())
    }

    /// The diagnostics of the capture.
    pub fn diagnostics(&self) -> String {
        self.stderr.lock().unwrap().text()
    }

    /// Stop the capture.
    ///
    /// The capture's process group is asked to stop with `SIGTERM` and killed
    /// when it does not stop in time. The direct child is reaped and the
    /// threads that read it are joined. Descendants that leave the group are
    /// outside this cleanup guarantee.
    pub fn stop(&mut self) {
        self.stop_requested.store(true, Ordering::SeqCst);
        if let Some(mut child) = self.child.take() {
            stop_process_group(child.id(), Some(&mut child));
        }
        self.join_threads();
    }

    /// Wait for the threads that read the child and join them.
    fn join_threads(&mut self) {
        // The queue is dropped first: a reader that waits for room in it
        // then fails its send and returns, so the join cannot block.
        self.events = None;
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }

    /// Check the format the child reported against the format that was
    /// asked for.
    ///
    /// The check is what keeps a capture that delivers the wrong format
    /// from feeding misaligned audio into the pipeline.
    fn check_reported_format(&self) -> Result<()> {
        let (state, changed) = &*self.reported;
        let deadline = Instant::now() + FORMAT_REPORT_TIMEOUT;
        let mut state = state.lock().unwrap();
        while state.format.is_none() && !state.stderr_finished {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let (next, timeout) = changed.wait_timeout(state, deadline - now).unwrap();
            state = next;
            if timeout.timed_out() {
                break;
            }
        }
        let Some(reported) = state.format.clone() else {
            return Err(anyhow!(
                "the audio capture did not report the raw audio format"
            ));
        };
        drop(state);
        ensure!(
            reported.rate == self.config.sample_rate,
            "the audio capture delivers {} Hz, the pipeline works with {} Hz",
            reported.rate,
            self.config.sample_rate
        );
        ensure!(
            reported.channels == self.config.channels,
            "the audio capture delivers {} channels, the pipeline works with {} channel",
            reported.channels,
            self.config.channels
        );
        ensure!(
            reported.format == self.config.format.pw_record_name(),
            "the audio capture delivers the format '{}', the pipeline expects '{}'",
            reported.format,
            self.config.format.pw_record_name()
        );
        ensure!(
            reported.sample_size == self.config.format.sample_bytes(),
            "the audio capture delivers {} bytes per sample, the pipeline expects {}",
            reported.sample_size,
            self.config.format.sample_bytes()
        );
        Ok(())
    }

    /// The end of the capture.
    ///
    /// A capture that failed without being asked to stop is an error. A
    /// capture that was asked to stop is the end of the recording.
    fn end_of_capture(&mut self) -> Result<Option<AudioBlock>> {
        self.finished = true;
        let status = self.reap();
        // The format is checked after the child and stderr reader finish.
        // Otherwise the raw-format line could still be in the pipe when
        // the end of stdout is observed.
        self.join_threads();
        if self.stop_requested.load(Ordering::SeqCst) {
            return Ok(None);
        }
        match status? {
            Some(status) if !status.success() => {
                let detail = self.diagnostics();
                let detail = detail.trim();
                Err(anyhow!(
                    "the audio capture failed with exit code {}{}",
                    status.code().unwrap_or(-1),
                    if detail.is_empty() {
                        String::new()
                    } else {
                        format!(": {detail}")
                    }
                ))
            }
            Some(status) => {
                self.check_reported_format()?;
                // The child left without an error. The source it recorded
                // from is gone, so the recording ends here.
                crate::dlog!(
                    1,
                    "the audio capture ended on its own with exit code {}",
                    status.code().unwrap_or(-1)
                );
                Ok(None)
            }
            None => Ok(None),
        }
    }

    /// Wait for the child and return its exit status.
    ///
    /// The wait is bounded: a child that closed its output but keeps
    /// running is killed instead of blocking the pipeline.
    fn reap(&mut self) -> Result<Option<std::process::ExitStatus>> {
        let Some(process_group) = self.child.as_ref().map(Child::id) else {
            return Ok(None);
        };
        let deadline = Instant::now() + STOP_GRACE;
        loop {
            let waited = self
                .child
                .as_mut()
                .expect("the capture child remains present while it is reaped")
                .try_wait();
            match waited {
                Ok(Some(status)) => {
                    if process_group_exists(process_group) {
                        stop_process_group(process_group, None);
                    }
                    self.child.take();
                    return Ok(Some(status));
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        send_group_signal(process_group, libc::SIGKILL);
                        let waited = self
                            .child
                            .as_mut()
                            .expect("the capture child remains present while it is reaped")
                            .wait();
                        match waited {
                            Ok(status) => {
                                self.child.take();
                                return Ok(Some(status));
                            }
                            Err(error) => {
                                return self.reap_after_wait_error(process_group, error);
                            }
                        }
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return self.reap_after_wait_error(process_group, error),
            }
        }
    }

    fn reap_after_wait_error(
        &mut self,
        process_group: u32,
        error: std::io::Error,
    ) -> Result<Option<std::process::ExitStatus>> {
        if let Some(mut child) = self.child.take() {
            stop_process_group(process_group, Some(&mut child));
        }
        Err(error).context("the audio capture process could not be waited for")
    }
}

/// What the capture waits for.
enum Wait {
    /// The reader thread delivered something.
    Event(CaptureEvent),
    /// Nothing arrived in time.
    Timeout,
    /// The reader thread is gone, so the capture is over.
    Closed,
}

impl AudioSource for PipeWireCapture {
    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn channels(&self) -> u16 {
        self.config.channels
    }

    /// The next block of samples.
    ///
    /// A block without samples means that the capture has nothing to
    /// deliver right now. It is not the end of the recording: the pipeline
    /// asks again, and it notices a cancellation in between.
    fn next_block(&mut self) -> Result<Option<AudioBlock>> {
        if self.finished {
            return Ok(None);
        }
        // The borrow of the queue ends before the capture is finished, so
        // that the end of the capture can be handled.
        let waited = {
            let Some(receiver) = self.events.as_ref() else {
                // The capture was stopped.
                self.finished = true;
                return Ok(None);
            };
            if self.cancel.is_cancelled() {
                // The run was cancelled. An empty block lets the pipeline
                // notice the cancellation itself, so that it is reported as
                // a cancellation and not as the end of a recording.
                Wait::Timeout
            } else {
                match receiver.recv_timeout(READ_POLL) {
                    Ok(event) => Wait::Event(event),
                    Err(RecvTimeoutError::Timeout) => Wait::Timeout,
                    Err(RecvTimeoutError::Disconnected) => Wait::Closed,
                }
            }
        };
        match waited {
            Wait::Event(CaptureEvent::Audio(samples)) => {
                self.check_reported_format()?;
                Ok(Some(AudioBlock::new(samples)))
            }
            Wait::Event(CaptureEvent::Failed(message)) => {
                self.finished = true;
                Err(anyhow!("the audio capture failed: {message}"))
            }
            Wait::Timeout => Ok(Some(AudioBlock::new(Vec::new()))),
            Wait::Closed => self.end_of_capture(),
        }
    }
}

impl Drop for PipeWireCapture {
    fn drop(&mut self) {
        // Stop the capture group before releasing its pipes and reader threads.
        self.stop();
    }
}

/// Stop a process group and reap its direct child when one is provided.
fn stop_process_group(process_group: u32, mut child: Option<&mut Child>) {
    if !process_group_exists(process_group) {
        if let Some(child) = child {
            let _ = child.wait();
        }
        return;
    }

    send_group_signal(process_group, libc::SIGTERM);
    let deadline = Instant::now() + STOP_GRACE;
    loop {
        if let Some(child) = child.as_deref_mut() {
            let _ = child.try_wait();
        }
        if !process_group_exists(process_group) {
            break;
        }
        if Instant::now() >= deadline {
            send_group_signal(process_group, libc::SIGKILL);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    if let Some(child) = child {
        let _ = child.wait();
    }
}

/// Whether a process group still has processes.
fn process_group_exists(process_group: u32) -> bool {
    // SAFETY: signal zero only checks whether the process group exists.
    unsafe {
        if libc::kill(-(process_group as libc::pid_t), 0) == 0 {
            return true;
        }
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// Send a signal to the capture's process group.
fn send_group_signal(process_group: u32, signal: i32) {
    // SAFETY: the process group id was assigned to the capture child when
    // it was started. A group that has already exited makes this fail.
    unsafe {
        libc::kill(-(process_group as libc::pid_t), signal);
    }
}

/// Read the audio of the child and decode it.
fn spawn_reader(
    stdout: ChildStdout,
    sender: SyncSender<CaptureEvent>,
    format: PcmFormat,
    delivered: Arc<AtomicU64>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = BufReader::with_capacity(64 * 1024, stdout);
        let mut decoder = RawPcmDecoder::new(format);
        let mut buffer = vec![0u8; 16 * 1024];
        let mut pending: Vec<f32> = Vec::with_capacity(BLOCK_SAMPLES * 2);
        loop {
            let read = match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    let _ = sender.send(CaptureEvent::Failed(error.to_string()));
                    return;
                }
            };
            let samples = decoder.decode(&buffer[..read]);
            if samples.is_empty() {
                continue;
            }
            delivered.fetch_add(samples.len() as u64, Ordering::Relaxed);
            pending.extend_from_slice(&samples);
            while pending.len() >= BLOCK_SAMPLES {
                let block: Vec<f32> = pending.drain(..BLOCK_SAMPLES).collect();
                if sender.send(CaptureEvent::Audio(block)).is_err() {
                    // The pipeline is gone, so nobody reads the audio any
                    // more.
                    return;
                }
            }
        }
        // The last block is shorter than the others.
        if !pending.is_empty() {
            let _ = sender.send(CaptureEvent::Audio(pending));
        }
    })
}

/// Read the diagnostics of the child.
fn spawn_stderr_reader(
    stderr: ChildStderr,
    tail: Arc<Mutex<StderrTail>>,
    reported: Arc<(Mutex<ReportedFormatState>, Condvar)>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if let Some(format) = parse_reported_format(&line) {
                let (state, changed) = &*reported;
                let mut state = state.lock().unwrap();
                state.format = Some(format);
                changed.notify_all();
            }
            let mut tail = tail.lock().unwrap();
            tail.push(line.as_bytes());
            tail.push(b"\n");
        }
        let (state, changed) = &*reported;
        state.lock().unwrap().stderr_finished = true;
        changed.notify_all();
    })
}

/// Parse the line in which `pw-record` reports the format of its raw
/// output.
///
/// The line looks like this:
///
/// ```text
/// raw: rate=16000 channels=1 fmt=s16 samplesize=2 stride=2
/// ```
///
/// The marker is looked for instead of at the beginning of the line, so
/// that a build of `pw-record` that prefixes its messages still works.
fn parse_reported_format(line: &str) -> Option<ReportedFormat> {
    let (_, rest) = line.rsplit_once("raw:")?;
    let mut rate = None;
    let mut channels = None;
    let mut format = None;
    let mut sample_size = None;
    for field in rest.split_whitespace() {
        let Some((key, value)) = field.split_once('=') else {
            continue;
        };
        match key {
            "rate" => rate = value.parse().ok(),
            "channels" => channels = value.parse().ok(),
            "fmt" => format = Some(value.to_string()),
            "samplesize" => sample_size = value.parse().ok(),
            _ => {}
        }
    }
    Some(ReportedFormat {
        rate: rate?,
        channels: channels?,
        format: format?,
        sample_size: sample_size?,
    })
}

/// A source of the PipeWire session that audio can be recorded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureDevice {
    /// The name of the node. It is the value that `--target` expects.
    pub name: String,
    /// The description of the node.
    pub description: String,
    /// The media class of the node.
    pub media_class: String,
}

/// The sources of the running PipeWire session that audio can be recorded
/// from.
///
/// The list comes from `pw-cli`. On Arch `pw-record` belongs to the package
/// `pipewire-audio` and `pw-cli` to the package `pipewire`, and
/// `pipewire-audio` depends on `pipewire`. The check therefore needs no
/// tool that the capture does not already need.
///
/// The list is only used to check a device that was asked for: `pw-record`
/// falls back to the default source when it is given a node that does not
/// exist, and it does not say so.
pub fn list_devices() -> Result<Vec<CaptureDevice>> {
    let output = Command::new(DEVICE_PROGRAM)
        .arg("ls")
        .arg("Node")
        .output()
        .with_context(|| {
            format!("the devices of the session could not be listed with {DEVICE_PROGRAM}")
        })?;
    if !output.status.success() {
        return Err(anyhow!(
            "{DEVICE_PROGRAM} did not list the nodes of the session (exit code {})",
            output.status.code().unwrap_or(-1)
        ));
    }
    Ok(parse_devices(&String::from_utf8_lossy(&output.stdout)))
}

/// The sources of a session, from the output of `pw-cli ls Node`.
fn parse_devices(listing: &str) -> Vec<CaptureDevice> {
    let mut devices = Vec::new();
    let mut name = None;
    let mut description = None;
    let mut media_class = None;

    for line in listing.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // A new node starts with its id and its type.
        if trimmed.starts_with("id ") && trimmed.contains("type PipeWire:Interface:Node") {
            push_device(&mut devices, &mut name, &mut description, &mut media_class);
            continue;
        }
        let Some((key, value)) = trimmed.split_once(" = ") else {
            continue;
        };
        let value = value.trim().trim_matches('"').to_string();
        match key {
            "node.name" => name = Some(value),
            "node.description" => description = Some(value),
            "media.class" => media_class = Some(value),
            _ => {}
        }
    }
    push_device(&mut devices, &mut name, &mut description, &mut media_class);
    devices
}

/// Add the node that was read to the list of the sources.
fn push_device(
    devices: &mut Vec<CaptureDevice>,
    name: &mut Option<String>,
    description: &mut Option<String>,
    media_class: &mut Option<String>,
) {
    let (Some(name), Some(media_class)) = (name.take(), media_class.take()) else {
        description.take();
        return;
    };
    let is_source = media_class == "Audio/Source" || media_class == "Stream/Input/Audio";
    if is_source {
        devices.push(CaptureDevice {
            name,
            description: description.take().unwrap_or_default(),
            media_class,
        });
    }
    description.take();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decoder_turns_s16_into_samples() {
        let mut decoder = RawPcmDecoder::new(PcmFormat::S16Le);
        // 0x8000 is the smallest value of a signed 16 bit sample.
        let bytes = [0u8, 0, 0x00, 0x80, 0, 0, 0, 0];
        let samples = decoder.decode(&bytes);
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], -1.0);
        assert_eq!(samples[2], 0.0);
        assert_eq!(samples[3], 0.0);
    }

    #[test]
    fn the_decoder_keeps_a_partial_sample() {
        let mut decoder = RawPcmDecoder::new(PcmFormat::S16Le);
        // The first byte of a sample arrives alone.
        assert!(decoder.decode(&[0x00]).is_empty());
        // The rest of it arrives with the next sample.
        let samples = decoder.decode(&[0x80, 0x00, 0x00]);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], -1.0);
        assert_eq!(samples[1], 0.0);
    }

    #[test]
    fn the_decoder_keeps_the_order_of_the_samples() {
        let mut decoder = RawPcmDecoder::new(PcmFormat::F32Le);
        let mut bytes = Vec::new();
        for value in [0.25f32, -0.5, 1.0, -0.125] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let samples = decoder.decode(&bytes);
        assert_eq!(samples, vec![0.25, -0.5, 1.0, -0.125]);
    }

    #[test]
    fn the_decoder_survives_a_split_at_every_byte() {
        let mut bytes = Vec::new();
        let mut expected = Vec::new();
        for value in 0..64i32 {
            let sample = value as f32 / 64.0;
            expected.push(sample);
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        let mut decoder = RawPcmDecoder::new(PcmFormat::F32Le);
        let mut samples = Vec::new();
        for chunk in bytes.chunks(1) {
            samples.extend_from_slice(&decoder.decode(chunk));
        }
        assert_eq!(samples, expected);
    }

    #[test]
    fn the_decoder_ignores_a_trailing_partial_sample() {
        let mut decoder = RawPcmDecoder::new(PcmFormat::S16Le);
        let samples = decoder.decode(&[0x00, 0x00, 0x11]);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0], 0.0);
    }

    #[test]
    fn the_reported_format_is_parsed() {
        let format =
            parse_reported_format("raw: rate=16000 channels=1 fmt=s16 samplesize=2 stride=2")
                .expect("the line is parsed");
        assert_eq!(format.rate, 16000);
        assert_eq!(format.channels, 1);
        assert_eq!(format.format, "s16");
        assert_eq!(format.sample_size, 2);
    }

    #[test]
    fn a_line_that_is_not_a_format_is_ignored() {
        assert!(parse_reported_format("stream state changed paused -> streaming").is_none());
        assert!(parse_reported_format("").is_none());
    }

    #[test]
    fn a_prefixed_format_line_is_parsed() {
        // A build of pw-record that prefixes its messages must still be
        // understood, or the check of the format would never run.
        let format = parse_reported_format(
            "[pw-record.c(42):0:main] raw: rate=16000 channels=1 fmt=s16 samplesize=2 stride=2",
        )
        .expect("the line is parsed");
        assert_eq!(format.rate, 16000);
        assert_eq!(format.channels, 1);
        assert_eq!(format.format, "s16");
        assert_eq!(format.sample_size, 2);
    }

    #[test]
    fn the_devices_are_read_from_the_node_list() {
        let listing = "\
\tid 40, type PipeWire:Interface:Node/3
\t\tnode.name = \"ec_capture_usb\"
\t\tmedia.class = \"Stream/Input/Audio\"
\tid 41, type PipeWire:Interface:Node/3
\t\tnode.description = \"USB AEC stage\"
\t\tnode.name = \"source_ec_usb\"
\t\tmedia.class = \"Audio/Source\"
\tid 42, type PipeWire:Interface:Node/3
\t\tnode.name = \"a_sink\"
\t\tmedia.class = \"Audio/Sink\"
";
        let devices = parse_devices(listing);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].name, "ec_capture_usb");
        assert_eq!(devices[1].name, "source_ec_usb");
        assert_eq!(devices[1].description, "USB AEC stage");
    }

    #[test]
    fn the_stderr_tail_is_bounded() {
        let mut tail = StderrTail::default();
        let line = vec![b'x'; 1024];
        for _ in 0..32 {
            tail.push(&line);
        }
        assert!(tail.bytes.len() <= STDERR_KEEP_BYTES);
        assert!(tail.truncated);
    }
}
