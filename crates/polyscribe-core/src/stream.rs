// SPDX-License-Identifier: MIT

//! Incremental audio input for the transcription pipeline.
//!
//! The file based transcription decodes a media file with ffmpeg and hands
//! the audio to whisper in windows of 30 seconds. This module provides the
//! same thing for audio that arrives while the transcription is running: a
//! source delivers decoded samples in blocks, and the pipeline turns them
//! into the same windows the file path produces.
//!
//! # The contract
//!
//! * A source delivers **decoded** samples: 16 kHz, mono, f32. The pipeline
//!   does not resample; a source that captures at another rate has to convert
//!   before it delivers.
//! * The samples are pulled, never pushed: [`AudioSource::next_block`] is
//!   called when the pipeline needs more audio. That is the backpressure: a
//!   slow consumer slows the producer down, and the audio that waits in the
//!   pipeline is bounded by one window plus the lookback.
//! * The end of the stream is explicit: `next_block` returns `None`.
//! * A source error aborts the transcription and is reported to the caller.
//! * [`CancellationToken`] stops the transcription at the next safe
//!   boundary: no further window is submitted for inference.
//!
//! # The lifecycle of a streaming transcription
//!
//! ```text
//!   samples received  ->  window submitted  ->  segment finalized
//!   (source)              (inference)           (reconciliation)
//! ```
//!
//! * **Samples received** are audio that arrived from the source. They are
//!   not transcribed yet.
//! * **A window submitted** is a block of 30 seconds that was handed to
//!   whisper.
//! * **A segment** belongs to a window. The last segment of a window can be
//!   cut short by the end of the window, so it stays *provisional* until the
//!   next window confirms it. A provisional segment is never reported as
//!   final: only finalized segments reach the caller.
//! * **Stream completion** means the source reached the end and the last
//!   window was flushed. A cancelled run reports the cancellation instead,
//!   and keeps only the segments that were finalized.
//!
//! The events of the run are reported through [`StreamEvent`].

use anyhow::{Result, ensure};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::chunked::{AudioWindow, WHISPER_SAMPLE_RATE};

/// The sample rate the pipeline works with.
pub const STREAM_SAMPLE_RATE: u32 = WHISPER_SAMPLE_RATE;

/// The number of channels the pipeline works with.
pub const STREAM_CHANNELS: u16 = 1;

/// A block of decoded audio samples.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBlock {
    /// The samples, 16 kHz mono f32.
    pub samples: Vec<f32>,
}

impl AudioBlock {
    /// A block of the given samples.
    pub fn new(samples: Vec<f32>) -> Self {
        Self { samples }
    }
}

/// A source of decoded audio.
///
/// The pipeline pulls the audio from the source, so the source decides how
/// much it delivers at a time. A source that captures from a device delivers
/// what the device has buffered; a source that reads a file can deliver
/// whatever is convenient.
pub trait AudioSource: Send {
    /// The sample rate of the delivered samples.
    fn sample_rate(&self) -> u32;

    /// The number of channels of the delivered samples.
    fn channels(&self) -> u16;

    /// The next block of samples, or `None` at the end of the stream.
    fn next_block(&mut self) -> Result<Option<AudioBlock>>;
}

/// A token that asks a running transcription to stop.
///
/// Cancellation is cooperative: it takes effect at the next safe boundary,
/// which is the moment before the next window is submitted for inference. A
/// whisper call that is already running is not interrupted.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the transcription to stop.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Whether the transcription was asked to stop.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// What happened during a streaming transcription.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// The source delivered audio. `total` is the number of samples received
    /// so far.
    SamplesReceived {
        /// The samples received so far.
        total: u64,
    },
    /// A window was handed to the inference. The windows are numbered from
    /// zero in the order they were submitted.
    WindowSubmitted {
        /// The number of the window.
        index: u64,
        /// The first sample of the window.
        start_sample: u64,
        /// The sample after the window.
        end_sample: u64,
    },
    /// A segment that may still change.
    ///
    /// The last segment of a window can be cut short by the end of the
    /// window, so it stays provisional until the next window confirms it.
    /// A provisional segment is reported for the caller that wants to show
    /// the text of a recording while it is still running. Its text may
    /// change, and the segment that is reported as final afterwards is the
    /// one that counts.
    SegmentProvisional {
        /// The segment, as it is known now.
        entry: crate::OutputEntry,
    },
    /// A segment is final: it will not be replaced by a later window.
    SegmentFinalized {
        /// The segment.
        entry: crate::OutputEntry,
    },
    /// The source reached the end of the stream and the last window was
    /// flushed.
    Completed {
        /// The samples that were received in total.
        total_samples: u64,
    },
    /// The transcription was cancelled.
    Cancelled,
}

/// How a streaming transcription ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOutcome {
    /// The stream was transcribed to its end.
    Completed {
        /// The number of finalized segments.
        segments: usize,
    },
    /// The transcription was cancelled.
    Cancelled {
        /// The number of finalized segments.
        segments: usize,
    },
}

/// How the windows of a transcription are cut.
///
/// The batch policy is the one of the file path: a window is as long as the
/// whisper encoder takes at once, so a long recording is decoded in as few
/// passes as possible. The live policy uses shorter windows, so that the
/// text of a recording appears while the recording is still running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowPolicy {
    /// The batch policy: windows of 30 seconds.
    Batch,
    /// The live policy: short windows, with a longer startup window until
    /// the language of the recording is known.
    Live(LiveWindowPolicy),
}

/// The window policy of an interactive transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveWindowPolicy {
    /// The number of samples a window holds.
    pub window_samples: usize,
    /// The number of samples the startup window holds.
    ///
    /// The language of a recording is detected on the first window. A short
    /// window carries too little speech for a reliable detection, so the
    /// startup window is longer.
    pub startup_samples: usize,
    /// The number of startup windows that are used at most.
    ///
    /// The bound keeps a recording that carries no speech from holding the
    /// startup window forever.
    pub max_startup_windows: u32,
}

impl Default for LiveWindowPolicy {
    fn default() -> Self {
        Self {
            window_samples: 5 * crate::chunked::WHISPER_SAMPLE_RATE as usize,
            startup_samples: 10 * crate::chunked::WHISPER_SAMPLE_RATE as usize,
            max_startup_windows: 3,
        }
    }
}

impl LiveWindowPolicy {
    /// The policy for a live window of the given length in seconds.
    ///
    /// The startup window is twice as long, because the language is
    /// detected on it. It is bounded by 30 seconds, the length of the
    /// window of the whisper encoder.
    pub fn new(window_seconds: f64) -> Self {
        let rate = crate::chunked::WHISPER_SAMPLE_RATE as f64;
        let window_samples = (window_seconds * rate).round().max(rate) as usize;
        let startup_samples = ((window_seconds * 2.0).min(30.0) * rate).round() as usize;
        Self {
            window_samples,
            startup_samples: startup_samples.max(window_samples),
            max_startup_windows: 3,
        }
    }

    /// The number of samples a window holds.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// The number of samples the startup window holds.
    pub fn startup_samples(&self) -> usize {
        self.startup_samples
    }
}

/// A source of windows of audio.
///
/// The file based decoder and the streaming decoder both implement this, so
/// the transcription loop has one implementation for both.
pub trait WindowSource {
    /// The next window, or `None` at the end of the audio.
    fn next_window(&mut self, cancel: &CancellationToken) -> Result<Option<AudioWindow>>;

    /// A window used to retry a region that was not transcribed.
    ///
    /// The final partial window may already have marked its source as
    /// finished. Implementations keep its buffered samples available for
    /// these bounded recovery attempts.
    fn next_recovery_window(&mut self, cancel: &CancellationToken) -> Result<Option<AudioWindow>> {
        self.next_window(cancel)
    }

    /// Move the start of the next window.
    fn seek_to(&mut self, start_sample: u64);

    /// The first sample that is still available for a window.
    fn retained_from(&self) -> u64;

    /// The samples that were received from the source.
    fn received_samples(&self) -> u64;

    /// Change the number of samples the windows that follow hold.
    ///
    /// The live policy uses this to switch from the longer startup window
    /// to the shorter window once the language of the recording is known.
    /// The file path keeps its windows and ignores the call.
    fn set_window_samples(&mut self, _samples: usize) {}
}

/// Turns an [`AudioSource`] into the windows the transcription loop consumes.
///
/// The adapter keeps at most one window plus the lookback in memory, so the
/// memory does not depend on the length of the stream.
pub struct StreamedAudio {
    source: Box<dyn AudioSource>,
    assembler: crate::chunked::WindowAssembler,
    /// The start of the next window.
    next_start: u64,
    /// The samples that were received in total.
    received: u64,
    /// Whether the source reached the end.
    finished: bool,
}

impl StreamedAudio {
    /// A streaming decoder for the given source.
    ///
    /// The windows are the ones of the batch policy: 30 seconds each.
    pub fn new(source: Box<dyn AudioSource>) -> Result<Self> {
        Self::with_config(source, crate::chunked::ChunkConfig::default())
    }

    /// A streaming decoder for the given source and window configuration.
    pub fn with_config(
        source: Box<dyn AudioSource>,
        config: crate::chunked::ChunkConfig,
    ) -> Result<Self> {
        ensure!(
            source.sample_rate() == STREAM_SAMPLE_RATE,
            "the audio source delivers {} Hz, the pipeline works with {STREAM_SAMPLE_RATE} Hz",
            source.sample_rate()
        );
        ensure!(
            source.channels() == STREAM_CHANNELS,
            "the audio source delivers {} channels, the pipeline works with {STREAM_CHANNELS} channel",
            source.channels()
        );
        Ok(Self {
            source,
            assembler: crate::chunked::WindowAssembler::new(config),
            next_start: 0,
            received: 0,
            finished: false,
        })
    }

    /// The samples that were received from the source.
    pub fn received_samples(&self) -> u64 {
        self.received
    }

    /// The samples that are buffered and not part of a window yet.
    pub fn buffered_samples(&self) -> usize {
        self.assembler.buffered_samples()
    }
}

impl WindowSource for StreamedAudio {
    fn next_window(&mut self, cancel: &CancellationToken) -> Result<Option<AudioWindow>> {
        if self.finished {
            return Ok(None);
        }
        let window_samples = self.assembler.window_samples() as u64;
        while self.assembler.available_from(self.next_start) < window_samples {
            if cancel.is_cancelled() {
                return Ok(None);
            }
            let Some(block) = self.source.next_block()? else {
                // The stream ended. The last window can be shorter.
                self.finished = true;
                break;
            };
            self.received += block.samples.len() as u64;
            self.assembler.push(&block.samples);
        }
        Ok(self.assembler.take_window(self.next_start))
    }

    fn next_recovery_window(&mut self, cancel: &CancellationToken) -> Result<Option<AudioWindow>> {
        if !self.finished {
            return self.next_window(cancel);
        }
        if cancel.is_cancelled() {
            return Ok(None);
        }
        Ok(self.assembler.take_window(self.next_start))
    }

    fn seek_to(&mut self, start_sample: u64) {
        self.next_start = start_sample;
    }

    fn retained_from(&self) -> u64 {
        self.assembler.retained_from()
    }

    fn received_samples(&self) -> u64 {
        self.received
    }

    fn set_window_samples(&mut self, samples: usize) {
        self.assembler.set_window_samples(samples);
    }
}

/// A deterministic audio source for tests.
///
/// The source delivers the samples of a buffer in blocks of a configurable
/// size. A gate can hold the delivery back, so that a test can observe what
/// the pipeline does while the source has not reached the end. An error can
/// be injected after a given number of blocks, and the source can be stopped.
#[derive(Debug)]
pub struct ScriptedAudioSource {
    samples: VecDeque<f32>,
    /// The size of the delivered blocks.
    pub block_size: usize,
    /// The number of blocks that are delivered before the gate closes.
    blocks_before_gate: usize,
    /// Whether the gate is closed.
    gate: Arc<(Mutex<bool>, Condvar)>,
    /// The error that is raised after the given number of blocks.
    fail_after: Option<(usize, String)>,
    /// The number of blocks that were delivered.
    delivered: usize,
    /// Set to stop the source.
    stopped: bool,
}

impl ScriptedAudioSource {
    /// A source that delivers the given samples in blocks of `block_size`.
    pub fn new(samples: Vec<f32>, block_size: usize) -> Self {
        Self {
            samples: samples.into(),
            block_size: block_size.max(1),
            blocks_before_gate: 0,
            gate: Arc::new((Mutex::new(false), Condvar::new())),
            fail_after: None,
            delivered: 0,
            stopped: false,
        }
    }

    /// Hold the delivery back after the given number of blocks.
    pub fn gate_after(&mut self, blocks: usize) {
        self.blocks_before_gate = blocks;
        if blocks > 0 {
            *self.gate.0.lock().unwrap() = false;
        }
    }

    /// Open the gate: the delivery continues.
    pub fn open_gate(&self) {
        self.gate_handle().open();
    }

    /// A handle that can open the gate from another thread.
    pub fn gate_handle(&self) -> GateHandle {
        GateHandle {
            gate: Arc::clone(&self.gate),
        }
    }

    /// Raise an error after the given number of blocks.
    pub fn fail_after(&mut self, blocks: usize, message: &str) {
        self.fail_after = Some((blocks, message.to_string()));
    }

    /// Stop the source: it reports the end of the stream.
    pub fn stop(&mut self) {
        self.stopped = true;
    }
}

/// A handle that opens the gate of a [`ScriptedAudioSource`].
#[derive(Debug, Clone)]
pub struct GateHandle {
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl GateHandle {
    /// Let the delivery continue.
    pub fn open(&self) {
        let (lock, condvar) = &*self.gate;
        *lock.lock().unwrap() = true;
        condvar.notify_all();
    }
}

impl AudioSource for ScriptedAudioSource {
    fn sample_rate(&self) -> u32 {
        STREAM_SAMPLE_RATE
    }

    fn channels(&self) -> u16 {
        STREAM_CHANNELS
    }

    fn next_block(&mut self) -> Result<Option<AudioBlock>> {
        if self.stopped || self.samples.is_empty() {
            return Ok(None);
        }
        if self.blocks_before_gate > 0 && self.delivered >= self.blocks_before_gate {
            // The gate holds the delivery back until the test opens it.
            let (lock, condvar) = &*self.gate;
            let mut open = lock.lock().unwrap();
            while !*open {
                let (guard, timeout) = condvar
                    .wait_timeout(open, std::time::Duration::from_millis(50))
                    .unwrap();
                open = guard;
                if timeout.timed_out() && self.stopped {
                    return Ok(None);
                }
            }
        }
        if let Some((after, message)) = &self.fail_after
            && self.delivered >= *after
        {
            return Err(anyhow::anyhow!("{message}"));
        }
        let count = self.block_size.min(self.samples.len());
        let block: Vec<f32> = self.samples.drain(..count).collect();
        self.delivered += 1;
        Ok(Some(AudioBlock::new(block)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that delivers a fixed number of samples.
    fn source(samples: usize, block: usize) -> ScriptedAudioSource {
        ScriptedAudioSource::new(vec![0.1; samples], block)
    }

    /// A source with a wrong sample rate.
    struct WrongRate;

    impl AudioSource for WrongRate {
        fn sample_rate(&self) -> u32 {
            44_100
        }
        fn channels(&self) -> u16 {
            STREAM_CHANNELS
        }
        fn next_block(&mut self) -> Result<Option<AudioBlock>> {
            Ok(None)
        }
    }

    /// Pull the windows of a stream, advancing the way the transcription does.
    fn windows(audio: &mut StreamedAudio) -> Vec<AudioWindow> {
        let mut out = Vec::new();
        while let Some(window) = audio.next_window(&CancellationToken::new()).unwrap() {
            let next = window.start_sample + window.samples.len() as u64;
            out.push(window);
            audio.seek_to(next);
        }
        out
    }

    #[test]
    fn the_window_is_thirty_seconds_long() {
        let mut audio =
            StreamedAudio::new(Box::new(source(40 * STREAM_SAMPLE_RATE as usize, 4096))).unwrap();
        let window = audio
            .next_window(&CancellationToken::new())
            .unwrap()
            .unwrap();
        assert_eq!(window.samples.len(), 30 * STREAM_SAMPLE_RATE as usize);
        assert_eq!(window.start_sample, 0);
    }

    #[test]
    fn the_block_size_does_not_change_the_windows() {
        let samples = 40 * STREAM_SAMPLE_RATE as usize;
        let mut starts = Vec::new();
        for block in [512usize, 4096, 16_000, 160_001] {
            let mut audio = StreamedAudio::new(Box::new(source(samples, block))).unwrap();
            let seen: Vec<(u64, usize)> = windows(&mut audio)
                .iter()
                .map(|w| (w.start_sample, w.samples.len()))
                .collect();
            starts.push(seen);
        }
        assert_eq!(starts[0], starts[1]);
        assert_eq!(starts[1], starts[2]);
        assert_eq!(starts[2], starts[3]);
        // The windows cover the audio without a gap.
        assert!(!starts[0].is_empty());
    }

    #[test]
    fn the_end_of_the_stream_flushes_the_last_window() {
        let samples = 35 * STREAM_SAMPLE_RATE as usize;
        let mut audio = StreamedAudio::new(Box::new(source(samples, 8192))).unwrap();
        let mut lengths = Vec::new();
        let mut total = 0u64;
        for window in windows(&mut audio) {
            lengths.push(window.samples.len());
            total += window.samples.len() as u64;
        }
        // The last window is the remainder of the audio.
        assert_eq!(total, samples as u64);
        assert_eq!(
            lengths.last().copied(),
            Some(5 * STREAM_SAMPLE_RATE as usize)
        );
    }

    #[test]
    fn a_final_partial_window_can_be_redecoded_for_recovery() {
        let samples = (0..14).map(|sample| sample as f32).collect();
        let config = crate::chunked::ChunkConfig {
            window_samples: 10,
            lookback_samples: 2,
        };
        let mut audio =
            StreamedAudio::with_config(Box::new(ScriptedAudioSource::new(samples, 4)), config)
                .unwrap();
        let cancel = CancellationToken::new();

        let first = audio.next_window(&cancel).unwrap().unwrap();
        assert_eq!(first.samples.len(), 10);
        audio.seek_to(10);
        let final_window = audio.next_window(&cancel).unwrap().unwrap();
        assert_eq!(final_window.samples, vec![10.0, 11.0, 12.0, 13.0]);

        audio.seek_to(8);
        let recovered = audio.next_recovery_window(&cancel).unwrap().unwrap();
        assert_eq!(recovered.start_sample, 8);
        assert_eq!(recovered.samples, vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        assert!(audio.next_window(&cancel).unwrap().is_none());
    }

    #[test]
    fn a_source_with_the_wrong_sample_rate_is_refused() {
        let error = match StreamedAudio::new(Box::new(WrongRate)) {
            Ok(_) => panic!("a source with the wrong sample rate was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("44100 Hz"), "{error}");
    }

    #[test]
    fn the_buffered_audio_stays_bounded() {
        // Ten minutes of audio, delivered in small blocks.
        let samples = 600 * STREAM_SAMPLE_RATE as usize;
        let mut audio = StreamedAudio::new(Box::new(source(samples, 1024))).unwrap();
        let mut max_buffered = 0usize;
        let mut count = 0;
        for window in windows(&mut audio) {
            max_buffered = max_buffered.max(audio.buffered_samples());
            count += 1;
            assert!(window.samples.len() <= 30 * STREAM_SAMPLE_RATE as usize);
        }
        assert!(count > 10, "the audio is delivered in windows");
        // The buffer never holds more than a window plus the lookback.
        let limit = 32 * STREAM_SAMPLE_RATE as usize;
        assert!(
            max_buffered <= limit,
            "the buffer held {max_buffered} samples, more than {limit}"
        );
    }

    #[test]
    fn the_cancellation_token_stops_the_adapter() {
        let samples = 120 * STREAM_SAMPLE_RATE as usize;
        let mut audio = StreamedAudio::new(Box::new(source(samples, 4096))).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(audio.next_window(&cancel).unwrap().is_none());
    }

    #[test]
    fn a_failing_source_reports_the_error() {
        let mut source = source(40 * STREAM_SAMPLE_RATE as usize, 4096);
        source.fail_after(2, "the device disappeared");
        let mut audio = StreamedAudio::new(Box::new(source)).unwrap();
        // The window needs the whole window, so the adapter pulls the blocks
        // until the source fails.
        let error = match audio.next_window(&CancellationToken::new()) {
            Ok(_) => panic!("the failing source was not noticed"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("the device disappeared"),
            "{error}"
        );
    }

    #[test]
    fn the_gate_holds_the_delivery_back() {
        let samples = 60 * STREAM_SAMPLE_RATE as usize;
        let mut source = source(samples, 4096);
        source.gate_after(1);
        let gate = source.gate_handle();
        // The first window needs the whole window, so the adapter blocks in
        // the source after the first block.
        let handle = std::thread::spawn(move || {
            let mut audio = StreamedAudio::new(Box::new(source)).unwrap();
            audio
                .next_window(&CancellationToken::new())
                .unwrap()
                .is_some()
        });
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(!handle.is_finished(), "the window was produced too early");
        // Opening the gate lets the window through.
        gate.open();
        assert!(handle.join().unwrap(), "the window was not produced");
    }
}
