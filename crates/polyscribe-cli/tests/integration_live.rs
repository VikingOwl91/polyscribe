// SPDX-License-Identifier: MIT

//! Tests and measurements of the live window policy.
//!
//! The live policy uses shorter windows than the batch path, so that the
//! text of a recording appears while the recording is still running. The
//! tests check the contract of the policy: the segments are ordered, every
//! segment is finalized once, and the text arrives before the recording
//! ends.
//!
//! The measurement replays a recording at the speed of the recording and
//! reports when the text appears, how long the decoding takes and what the
//! transcript holds. It is ignored by default: it is a measurement, not a
//! check.
//!
//! The tests need a Whisper model, so they skip themselves without one,
//! like the other end-to-end tests.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use polyscribe_core::backend::{Language, ModelSession, transcribe_stream};
use polyscribe_core::stream::{
    AudioBlock, AudioSource, CancellationToken, LiveWindowPolicy, StreamEvent, WindowPolicy,
};

const SAMPLE_RATE: u32 = 16_000;

/// The text and audio files used to compose a measurement recording.
const MEASUREMENT_CLIPS: [(&str, &str); 5] = [
    ("clip1.wav", "the quick brown fox jumps over the lazy dog"),
    ("clip2.wav", "she sells sea shells by the sea shore"),
    ("clip3.wav", "how much wood would a woodchuck chuck"),
    ("clip4.wav", "the rain in spain stays mainly in the plain"),
    (
        "clip6.wav",
        "we promptly judged antique ivory buckles for the next prize",
    ),
];

/// The backend the tests run on.
fn test_backend() -> polyscribe_core::backend::BackendKind {
    match std::env::var("POLYSCRIBE_TEST_BACKEND").as_deref() {
        Ok("hip") => polyscribe_core::backend::BackendKind::Hip,
        Ok("cuda") => polyscribe_core::backend::BackendKind::Cuda,
        _ => polyscribe_core::backend::BackendKind::Cpu,
    }
}

/// The model the tests transcribe with.
fn e2e_model() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("POLYSCRIBE_E2E_MODEL").ok()?);
    path.is_file().then_some(path)
}

/// The samples of a WAV file.
fn wav_samples(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let mut pos = 12;
    let (start, len) = loop {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            break (pos + 8, size);
        }
        pos += 8 + size + (size % 2);
    };
    bytes[start..start + len]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

/// The committed speech fixture.
fn speech() -> Vec<f32> {
    wav_samples(&fixture("speech.wav"))
}

/// A recording of varied speech, with a pause between the clips.
///
/// The clips are the ones of the boundary fixtures: short utterances of
/// different text. A pause between them makes the recording look like a
/// dictation, with a window that holds more than one segment and a window
/// that holds silence.
///
/// The phase shifts the beginning of the recording, so that the speech
/// does not always start at the same place in a window. A window policy
/// has to work for every phase, not only for the one that suits it.
fn varied(seconds: f64, phase: usize) -> Vec<f32> {
    let clips: Vec<Vec<f32>> = [
        "clip1.wav",
        "clip2.wav",
        "clip3.wav",
        "clip4.wav",
        "clip6.wav",
    ]
    .iter()
    .map(|name| wav_samples(&fixture(&format!("boundary/{name}"))))
    .collect();
    let mut out: Vec<f32> = Vec::new();
    let pause = vec![0.0f32; 16_000 / 2];
    let target = (seconds * SAMPLE_RATE as f64) as usize;
    // The phase moves the speech away from the beginning of the window.
    let lead = (phase as f64 * 0.41 * SAMPLE_RATE as f64) as usize;
    out.extend(std::iter::repeat_n(0.0f32, lead));
    let mut index = phase % clips.len();
    while out.len() < target {
        let clip = &clips[index % clips.len()];
        index += 1;
        out.extend_from_slice(clip);
        out.extend_from_slice(&pause);
    }
    out.truncate(target);
    out
}

/// The recording the measurement uses.
///
/// The length can be changed with `POLYSCRIBE_LIVE_AUDIO_SECONDS`, the
/// phase with `POLYSCRIBE_LIVE_PHASE`.
fn measurement_audio() -> Vec<f32> {
    varied(measurement_seconds(), measurement_phase())
}

fn measurement_seconds() -> f64 {
    std::env::var("POLYSCRIBE_LIVE_AUDIO_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30.0)
}

fn measurement_phase() -> usize {
    std::env::var("POLYSCRIBE_LIVE_PHASE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// A fixture of the test directory.
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// A source that delivers the samples at the speed of the recording.
///
/// The source holds the delivery back until the block is due, so that the
/// measurement sees what a user sees: the text of a recording appears while
/// the recording is still running.
struct RealtimeSource {
    samples: Vec<f32>,
    position: usize,
    block: usize,
    /// The moment the recording started.
    origin: Instant,
    /// Whether the delivery waits for the block to be due.
    pace: bool,
}

impl RealtimeSource {
    fn new(samples: Vec<f32>, block: usize, pace: bool) -> Self {
        Self {
            samples,
            position: 0,
            block: block.max(1),
            origin: Instant::now(),
            pace,
        }
    }
}

impl AudioSource for RealtimeSource {
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn channels(&self) -> u16 {
        1
    }

    fn next_block(&mut self) -> anyhow::Result<Option<AudioBlock>> {
        if self.position >= self.samples.len() {
            return Ok(None);
        }
        if self.pace {
            let due =
                self.origin + Duration::from_secs_f64(self.position as f64 / SAMPLE_RATE as f64);
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
        }
        let count = self.block.min(self.samples.len() - self.position);
        let block: Vec<f32> = self.samples[self.position..self.position + count].to_vec();
        self.position += count;
        Ok(Some(AudioBlock::new(block)))
    }
}

/// The words of a text, in lower case.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect()
        })
        .filter(|word: &String| !word.is_empty())
        .collect()
}

/// The words of a transcript.
fn transcript_words(entries: &[polyscribe_core::OutputEntry]) -> Vec<String> {
    entries
        .iter()
        .flat_map(|entry| words(&entry.text))
        .collect()
}

/// The policy the measurement runs with.
fn measurement_policy() -> WindowPolicy {
    match std::env::var("POLYSCRIBE_LIVE_POLICY").as_deref() {
        Ok("batch") => WindowPolicy::Batch,
        _ => {
            let seconds: f64 = std::env::var("POLYSCRIBE_LIVE_WINDOW_SECONDS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(5.0);
            WindowPolicy::Live(LiveWindowPolicy::new(seconds))
        }
    }
}

/// The result of a measurement.
struct Measurement {
    /// The moment the recording started.
    origin: Instant,
    /// The moment the run ended.
    finished: Instant,
    /// The moment the first provisional segment was reported.
    first_provisional: Option<Duration>,
    /// The moment the first finalized segment was reported.
    first_finalized: Option<Duration>,
    /// The moment the last segment was reported.
    last_event: Option<Duration>,
    /// The number of windows that were submitted.
    windows: u64,
    /// The number of windows that were handed to the inference, including
    /// the windows that were decoded again to recover a region that
    /// whisper did not transcribe.
    decodes: u64,
    /// The finalized segments.
    entries: Vec<polyscribe_core::OutputEntry>,
    /// The provisional segments, in the order they were reported.
    provisional: Vec<polyscribe_core::OutputEntry>,
}

/// Run a measurement of the given samples.
///
/// With `pace` the audio is delivered at the speed of the recording, so the
/// measurement sees what a user sees. Without it the audio is delivered as
/// fast as it can be consumed, so the measurement shows the fast-path
/// wall time, including model loading.
fn measure(samples: Vec<f32>, policy: WindowPolicy, pace: bool, model: PathBuf) -> Measurement {
    let block = SAMPLE_RATE as usize / 10;
    let source = RealtimeSource::new(samples, block, pace);
    let origin = Instant::now();
    let mut session = ModelSession::with_model(model);
    let mut first_provisional = None;
    let mut first_finalized = None;
    let mut last_event = None;
    let mut windows = 0u64;
    let mut entries = Vec::new();
    let mut provisional = Vec::new();
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        policy,
        &mut |event: StreamEvent| {
            match &event {
                StreamEvent::SegmentProvisional { entry } => {
                    if first_provisional.is_none() {
                        first_provisional = Some(origin.elapsed());
                    }
                    provisional.push(entry.clone());
                }
                StreamEvent::SegmentFinalized { entry } => {
                    if first_finalized.is_none() {
                        first_finalized = Some(origin.elapsed());
                    }
                    entries.push(entry.clone());
                }
                StreamEvent::WindowSubmitted { index, .. } => {
                    windows = index + 1;
                }
                _ => {}
            }
            last_event = Some(origin.elapsed());
        },
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");
    assert_eq!(
        windows, result.windows,
        "every inference submission must emit WindowSubmitted"
    );
    assert_eq!(
        result.entries.len(),
        entries.len(),
        "the result and the events hold the same segments"
    );
    Measurement {
        origin,
        decodes: result.windows,
        finished: Instant::now(),
        first_provisional,
        first_finalized,
        last_event,
        windows,
        entries,
        provisional,
    }
}

/// The peak memory of the process, in bytes.
fn peak_rss() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| {
                    l.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse::<u64>().ok())
                })
        })
        .unwrap_or(0)
        * 1024
}

/// The number of segments that repeat another segment.
fn duplicate_count(entries: &[polyscribe_core::OutputEntry]) -> usize {
    let mut seen: Vec<(u64, u64, String)> = Vec::new();
    let mut duplicates = 0;
    for entry in entries {
        let key = (
            entry.start.to_bits(),
            entry.end.to_bits(),
            entry.text.clone(),
        );
        if seen.contains(&key) {
            duplicates += 1;
        }
        seen.push(key);
    }
    duplicates
}

/// The reference for the complete clips in a composed measurement recording.
///
/// A final clip can be cut by the end of the recording. It is excluded from
/// the reference because its full utterance was not present in the audio.
fn measurement_reference(seconds: f64, phase: usize) -> Vec<String> {
    let total = (seconds * SAMPLE_RATE as f64) as usize;
    let mut position = (phase as f64 * 0.41 * SAMPLE_RATE as f64) as usize;
    let mut index = phase % MEASUREMENT_CLIPS.len();
    let mut reference = Vec::new();

    while position < total {
        let (file, text) = MEASUREMENT_CLIPS[index % MEASUREMENT_CLIPS.len()];
        let samples = wav_samples(&fixture(&format!("boundary/{file}"))).len();
        if position + samples > total {
            break;
        }
        reference.extend(words(text));
        position += samples + SAMPLE_RATE as usize / 2;
        index += 1;
    }
    reference
}

/// Word-level Levenshtein distance.
fn edit_distance(reference: &[String], hypothesis: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    for (index, expected) in reference.iter().enumerate() {
        let mut current = vec![index + 1; hypothesis.len() + 1];
        for (column, actual) in hypothesis.iter().enumerate() {
            current[column + 1] = if expected == actual {
                previous[column]
            } else {
                1 + previous[column]
                    .min(previous[column + 1])
                    .min(current[column])
            };
        }
        previous = current;
    }
    previous[hypothesis.len()]
}

fn word_error_rate(reference: &[String], hypothesis: &[String]) -> f64 {
    edit_distance(reference, hypothesis) as f64 / reference.len().max(1) as f64
}

/// The live policy produces ordered segments that are finalized once.
#[test]
fn the_live_policy_produces_ordered_segments() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    let policy = WindowPolicy::Live(LiveWindowPolicy::new(5.0));
    let measurement = measure(speech(), policy, true, model);

    assert!(
        !measurement.entries.is_empty(),
        "the live policy produced no segment"
    );
    for pair in measurement.entries.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "the segments are not ordered: {pair:?}"
        );
        assert!(
            pair[0].end <= pair[1].start + 0.5,
            "the segments overlap: {pair:?}"
        );
    }
    // No segment is reported twice.
    assert_eq!(
        duplicate_count(&measurement.entries),
        0,
        "a segment was reported twice"
    );
}

#[test]
fn the_measurement_reference_matches_the_composed_audio() {
    assert_eq!(measurement_reference(4.0, 0), words(MEASUREMENT_CLIPS[0].1));
    assert_eq!(measurement_reference(3.0, 0), Vec::<String>::new());
    assert_eq!(measurement_reference(4.0, 1), words(MEASUREMENT_CLIPS[1].1));
    assert_eq!(edit_distance(&words("one two"), &words("one three")), 1);
}

#[test]
fn recovery_decodes_are_reported_as_window_submissions() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    let samples = wav_samples(&fixture("recovery/skipped-22s.wav"));
    let source = RealtimeSource::new(samples, SAMPLE_RATE as usize / 10, false);
    let mut session = ModelSession::with_model(model);
    let mut submitted = 0;
    let mut received = 0;
    let mut samples_reported_at_submission = Vec::new();
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        WindowPolicy::Batch,
        &mut |event: StreamEvent| match event {
            StreamEvent::SamplesReceived { total } => received = total,
            StreamEvent::WindowSubmitted { end_sample, .. } => {
                submitted += 1;
                samples_reported_at_submission.push((end_sample, received));
            }
            _ => {}
        },
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");

    assert!(
        result.windows > 2,
        "the recovery fixture did not trigger a recovery decode: {} windows",
        result.windows
    );
    assert_eq!(
        submitted, result.windows,
        "every inference window, including recovery decodes, must be reported"
    );
    for (end_sample, received) in samples_reported_at_submission {
        assert!(
            received >= end_sample,
            "window ending at sample {end_sample} was submitted after only {received} received samples were reported"
        );
    }
}

/// The live policy with the smallest supported window does not stall when a
/// speech hole cannot be recovered.
///
/// With a two-second window the minimum advance equals the largest recovery
/// lookback. Measuring the advance from the recovery window (which starts
/// earlier) could keep the main window at its original position. The advance
/// is measured from the main window instead, so progress is bounded.
#[test]
fn live_policy_with_minimum_window_makes_bounded_progress() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    let samples = wav_samples(&fixture("recovery/skipped-22s.wav"));
    let source = RealtimeSource::new(samples, SAMPLE_RATE as usize / 10, false);
    let mut session = ModelSession::with_model(model);
    let mut submitted = 0;
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        WindowPolicy::Live(LiveWindowPolicy::new(2.0)),
        &mut |event: StreamEvent| {
            if matches!(event, StreamEvent::WindowSubmitted { .. }) {
                submitted += 1;
            }
        },
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");

    // The recording is about 78 seconds long. With two-second windows the
    // run needs at least a few windows and must not loop forever. An
    // unbounded re-decode would produce thousands of windows.
    assert!(
        result.windows >= 3,
        "the live run produced too few windows: {}",
        result.windows
    );
    assert!(
        result.windows <= 200,
        "the live run did not make bounded progress: {} windows",
        result.windows
    );
    assert_eq!(
        submitted, result.windows,
        "every inference window must be reported, including recovery decodes"
    );
}

/// A whole-window segment does not trigger repeated re-decoding on the live
/// path.
///
/// The live policy already skips a segment that covers a whole window. This
/// test verifies that a recording of silence (which makes Whisper hallucinate
/// a phrase spanning a whole window) does not produce an excessive number of
/// windows.
#[test]
fn live_policy_does_not_decode_silence_over_and_over() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("SKIPPED: ffmpeg is not available on PATH; skipping the silence test");
        return;
    }

    let samples = vec![0.0f32; 30 * SAMPLE_RATE as usize];
    let source = RealtimeSource::new(samples, SAMPLE_RATE as usize / 10, false);
    let mut session = ModelSession::with_model(model);
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        WindowPolicy::Live(LiveWindowPolicy::new(5.0)),
        &mut |_| {},
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");

    // 30 seconds of silence with 5-second windows needs 6 windows. A
    // whole-window hallucination must not re-decode the same window over
    // and over. Before the fix this took dozens of windows.
    assert!(
        result.windows <= 12,
        "silence was decoded over and over: {} windows",
        result.windows
    );
}

/// The live policy reports the text while the recording is still running.
#[test]
fn the_live_policy_reports_the_text_before_the_source_ends() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    let samples = speech();
    let total = samples.len() as u64;
    // The source counts the samples it delivered, so that the test can see
    // whether the recording was still running when the text appeared.
    let delivered = Arc::new(AtomicU64::new(0));
    let source = RealtimeSource::new(samples, SAMPLE_RATE as usize, true);
    let source = CountingSource {
        inner: source,
        delivered: Arc::clone(&delivered),
    };

    let mut session = ModelSession::with_model(model);
    let mut first_finalized_at: Option<u64> = None;
    transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        WindowPolicy::Live(LiveWindowPolicy::new(3.0)),
        &mut |event: StreamEvent| {
            if let StreamEvent::SegmentFinalized { .. } = event
                && first_finalized_at.is_none()
            {
                first_finalized_at = Some(delivered.load(Ordering::Relaxed));
            }
        },
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");

    let delivered_at_first_segment =
        first_finalized_at.expect("the live policy finalized a segment");
    assert!(
        delivered_at_first_segment < total,
        "the recording delivered all {total} samples before the first segment was finalized"
    );
}

/// A source that counts the samples it delivered.
struct CountingSource {
    inner: RealtimeSource,
    delivered: Arc<AtomicU64>,
}

impl AudioSource for CountingSource {
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    fn channels(&self) -> u16 {
        self.inner.channels()
    }

    fn next_block(&mut self) -> anyhow::Result<Option<AudioBlock>> {
        let block = self.inner.next_block()?;
        if let Some(block) = &block {
            self.delivered
                .fetch_add(block.samples.len() as u64, Ordering::Relaxed);
        }
        Ok(block)
    }
}

/// The language of a recording is detected and pinned.
#[test]
fn the_language_is_detected() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    let source = RealtimeSource::new(speech(), SAMPLE_RATE as usize, false);
    let mut session = ModelSession::with_model(model);
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Auto,
        WindowPolicy::Live(LiveWindowPolicy::new(5.0)),
        &mut |_event: StreamEvent| {},
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");

    let detected = result.detected_language.clone();
    // The language of the recording is either detected or known from the
    // model: an English-only model can only transcribe English, so its
    // language is not detected at all.
    if session.is_english_only() {
        assert!(
            detected.is_none(),
            "an English-only model must not detect a language: {detected:?}"
        );
    } else {
        assert_eq!(
            detected.as_deref(),
            Some("en"),
            "the English recording was not detected as English"
        );
    }
    assert!(
        !result.entries.is_empty(),
        "the recording was not transcribed"
    );
}

/// Every segment of a live run is finalized, and the end of the stream
/// flushes the last one.
#[test]
fn every_segment_is_finalized_once() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    let source = RealtimeSource::new(speech(), SAMPLE_RATE as usize, false);
    let mut session = ModelSession::with_model(model);
    let mut finalized: Vec<polyscribe_core::OutputEntry> = Vec::new();
    let mut provisional: Vec<polyscribe_core::OutputEntry> = Vec::new();
    let mut completed = false;
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        WindowPolicy::Live(LiveWindowPolicy::new(3.0)),
        &mut |event: StreamEvent| match event {
            StreamEvent::SegmentFinalized { entry } => finalized.push(entry),
            StreamEvent::SegmentProvisional { entry } => provisional.push(entry),
            StreamEvent::Completed { .. } => completed = true,
            _ => {}
        },
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");

    assert!(completed, "the run did not complete");
    assert!(!finalized.is_empty(), "no segment was finalized");
    // The end of the stream flushed every segment of the result.
    assert_eq!(
        finalized.len(),
        result.entries.len(),
        "the finalized events and the result hold a different number of segments"
    );
    // No segment is reported twice.
    assert_eq!(
        duplicate_count(&finalized),
        0,
        "a segment was finalized twice"
    );
    // The timestamps move forward.
    for pair in finalized.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "the segments are not ordered: {pair:?}"
        );
    }
    // A provisional segment is not a final one.
    for entry in &provisional {
        assert!(
            !finalized
                .iter()
                .any(|final_entry| final_entry.start == entry.start && final_entry.end == entry.end),
            "a segment was reported as provisional and as final: {entry:?}"
        );
    }
}

/// The measurement of the live window policy.
///
/// The test is ignored by default: it is a measurement, not a check. Run it
/// with `cargo test -- --ignored --nocapture` and a model in
/// `POLYSCRIBE_E2E_MODEL`. The window is chosen with
/// `POLYSCRIBE_LIVE_WINDOW_SECONDS`, the policy with
/// `POLYSCRIBE_LIVE_POLICY` (`live` or `batch`).
#[test]
#[ignore = "a measurement, not a test"]
fn measure_the_live_window() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the measurement");
        return;
    };
    let policy = measurement_policy();
    let samples = measurement_audio();
    let audio_seconds = samples.len() as f64 / SAMPLE_RATE as f64;

    // The run at the speed of the recording: the latency of the text.
    let live = measure(samples.clone(), policy, true, model.clone());
    let wall = live.finished.duration_since(live.origin).as_secs_f64();

    // The fast run delivers audio as soon as it is consumed. Its wall time
    // includes loading the model.
    let fast = measure(samples, policy, false, model);
    let fast_wall = fast.finished.duration_since(fast.origin).as_secs_f64();

    let reference = measurement_reference(measurement_seconds(), measurement_phase());
    let hypothesis = transcript_words(&live.entries);
    let errors = edit_distance(&reference, &hypothesis);

    println!("live window measurement");
    println!("  policy:              {policy:?}");
    println!("  audio:               {audio_seconds:.2} s");
    println!("  windows submitted:   {}", live.windows);
    println!("  whisper calls:       {}", live.decodes);
    println!(
        "  recovery re-decodes: {}",
        live.decodes.saturating_sub(live.windows)
    );
    println!("  segments:            {}", live.entries.len());
    println!("  provisional events:  {}", live.provisional.len());
    println!(
        "  first provisional:   {}",
        live.first_provisional
            .map(|d| format!("{:.2} s", d.as_secs_f64()))
            .unwrap_or_else(|| "never".to_string())
    );
    println!(
        "  first finalized:     {}",
        live.first_finalized
            .map(|d| format!("{:.2} s", d.as_secs_f64()))
            .unwrap_or_else(|| "never".to_string())
    );
    println!(
        "  last event:          {}",
        live.last_event
            .map(|d| format!("{:.2} s", d.as_secs_f64()))
            .unwrap_or_else(|| "never".to_string())
    );
    println!("  wall clock:          {wall:.2} s");
    println!(
        "  real time factor:    {:.2} (audio seconds per wall second)",
        audio_seconds / wall.max(0.001)
    );
    println!("  fast-run wall:       {fast_wall:.2} s (includes model loading)");
    println!(
        "  fast-run per call:   {:.2} s",
        fast_wall / live.decodes.max(1) as f64
    );
    println!(
        "  calls per audio s:   {:.2}",
        live.decodes as f64 / audio_seconds.max(0.001)
    );
    println!(
        "  fast-path ratio:     {:.2} (audio seconds per fast-run wall second)",
        audio_seconds / fast_wall.max(0.001)
    );
    println!(
        "  fast-run / audio:    {:.2} (fast-run wall seconds per audio second)",
        fast_wall / audio_seconds.max(0.001)
    );
    println!("  duplicate segments:  {}", duplicate_count(&live.entries));
    println!("  reference words:     {}", reference.len());
    println!(
        "  WER:                 {:.3} ({errors} word errors)",
        word_error_rate(&reference, &hypothesis)
    );
    println!(
        "  peak RSS (both runs): {:.1} MB",
        peak_rss() as f64 / 1024.0 / 1024.0
    );
    println!("  transcript:");
    for entry in &live.entries {
        println!(
            "    [{:7.2} -> {:7.2}] {}",
            entry.start, entry.end, entry.text
        );
    }
}

/// The measurement of the language detection.
///
/// The recording is given with `POLYSCRIBE_LIVE_LANG_AUDIO`, the window
/// with `POLYSCRIBE_LIVE_WINDOW_SECONDS`. The detection runs on the first
/// window, so the recording has to carry speech.
#[test]
#[ignore = "a measurement, not a test"]
fn measure_the_language_detection() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the measurement");
        return;
    };
    let Ok(audio) = std::env::var("POLYSCRIBE_LIVE_LANG_AUDIO") else {
        eprintln!("SKIPPED: POLYSCRIBE_LIVE_LANG_AUDIO is not set");
        return;
    };

    let policy = measurement_policy();
    let samples = wav_samples(Path::new(&audio));
    let audio_seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    let source = RealtimeSource::new(samples, SAMPLE_RATE as usize / 10, false);
    let mut session = ModelSession::with_model(model);
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Auto,
        policy,
        &mut |_event: StreamEvent| {},
        &CancellationToken::new(),
    )
    .expect("the transcription does not fail");

    println!("language detection measurement");
    println!("  policy:              {policy:?}");
    println!("  audio:               {audio_seconds:.2} s");
    println!(
        "  detected language:   {}",
        result.detected_language.as_deref().unwrap_or("none")
    );
    println!("  segments:            {}", result.entries.len());
    for entry in &result.entries {
        println!(
            "    [{:7.2} -> {:7.2}] {}",
            entry.start, entry.end, entry.text
        );
    }
}
