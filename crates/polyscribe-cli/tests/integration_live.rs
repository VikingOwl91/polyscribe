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
use std::time::{Duration, Instant};

use polyscribe_core::backend::{Language, ModelSession, transcribe_stream};
use polyscribe_core::stream::{
    AudioBlock, AudioSource, CancellationToken, LiveWindowPolicy, StreamEvent, WindowPolicy,
};

const SAMPLE_RATE: u32 = 16_000;

/// The text of the committed fixture.
const REFERENCE_TEXT: &str =
    "this is a short test recording for the polyscribe end to end transcription test";

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
    let seconds: f64 = std::env::var("POLYSCRIBE_LIVE_AUDIO_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30.0);
    let phase: usize = std::env::var("POLYSCRIBE_LIVE_PHASE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    varied(seconds, phase)
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
/// fast as it can be consumed, so the measurement shows the cost of the
/// decoding alone.
fn measure(samples: Vec<f32>, policy: WindowPolicy, pace: bool) -> Measurement {
    let block = SAMPLE_RATE as usize / 10;
    let source = RealtimeSource::new(samples, block, pace);
    let origin = Instant::now();
    let mut session = ModelSession::new();
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

/// The words of the reference that the transcript does not hold.
fn missing_words(entries: &[polyscribe_core::OutputEntry]) -> Vec<String> {
    let reference = words(REFERENCE_TEXT);
    let transcript = transcript_words(entries);
    let mut missing = Vec::new();
    for word in reference {
        if !transcript.contains(&word) {
            missing.push(word);
        }
    }
    missing
}

/// The words of the transcript that the reference does not hold.
fn extra_words(entries: &[polyscribe_core::OutputEntry]) -> Vec<String> {
    let reference = words(REFERENCE_TEXT);
    let transcript = transcript_words(entries);
    let mut extra = Vec::new();
    for word in transcript {
        if !reference.contains(&word) {
            extra.push(word);
        }
    }
    extra
}

/// The live policy produces ordered segments that are finalized once.
#[test]
fn the_live_policy_produces_ordered_segments() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    // SAFETY: the test process sets the variable before the transcription
    // reads it.
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let policy = WindowPolicy::Live(LiveWindowPolicy::new(5.0));
    let measurement = measure(speech(), policy, true);

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

/// The live policy reports the text while the recording is still running.
#[test]
fn the_live_policy_reports_the_text_before_the_source_ends() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the live tests");
        return;
    };
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let policy = WindowPolicy::Live(LiveWindowPolicy::new(3.0));
    let measurement = measure(speech(), policy, true);

    let audio_seconds = speech().len() as f64 / SAMPLE_RATE as f64;
    let first = measurement
        .first_finalized
        .expect("the live policy finalized a segment");
    assert!(
        first.as_secs_f64() < audio_seconds,
        "the first segment was finalized after {} s, the recording is {audio_seconds} s long",
        first.as_secs_f64()
    );
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
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let policy = measurement_policy();
    let samples = measurement_audio();
    let audio_seconds = samples.len() as f64 / SAMPLE_RATE as f64;

    // The run at the speed of the recording: the latency of the text.
    let live = measure(samples.clone(), policy, true);
    let wall = live.finished.duration_since(live.origin).as_secs_f64();

    // The run as fast as the machine can: the cost of the decoding.
    let fast = measure(samples, policy, false);
    let decode = fast.finished.duration_since(fast.origin).as_secs_f64();

    let missing = missing_words(&live.entries);
    let extra = extra_words(&live.entries);

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
    println!("  decoding only:      {decode:.2} s");
    println!(
        "  decode per call:     {:.2} s",
        decode / live.decodes.max(1) as f64
    );
    println!(
        "  calls per audio s:   {:.2}",
        live.decodes as f64 / audio_seconds.max(0.001)
    );
    println!(
        "  decode throughput:   {:.2} (audio seconds per decode second)",
        audio_seconds / decode.max(0.001)
    );
    println!(
        "  headroom:            {:.2} (the live pace fits in the decode time)",
        decode / audio_seconds.max(0.001)
    );
    println!("  duplicate segments:  {}", duplicate_count(&live.entries));
    println!("  missing words:       {missing:?}");
    println!("  extra words:         {extra:?}");
    println!(
        "  peak RSS:            {:.1} MB",
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
