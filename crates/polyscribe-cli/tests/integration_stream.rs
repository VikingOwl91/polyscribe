// SPDX-License-Identifier: MIT

//! End-to-end tests of the streaming transcription.
//!
//! The tests feed a recording to the local whisper.cpp engine in blocks,
//! while the recording is still being delivered, and check that the
//! transcript grows before the source reached its end.
//!
//! The tests need a Whisper model, so they skip themselves without one,
//! like the other end-to-end tests.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use polyscribe_core::backend::{Language, ModelSession, transcribe_stream};
use polyscribe_core::stream::{CancellationToken, ScriptedAudioSource, StreamEvent};

const SAMPLE_RATE: u32 = 16_000;

/// The backend the tests run on.
///
/// The CI has no GPU, so the tests run on the CPU. A machine with a GPU can
/// run them on it with `POLYSCRIBE_TEST_BACKEND=hip`.
fn test_backend() -> polyscribe_core::backend::BackendKind {
    match std::env::var("POLYSCRIBE_TEST_BACKEND").as_deref() {
        Ok("hip") => polyscribe_core::backend::BackendKind::Hip,
        Ok("cuda") => polyscribe_core::backend::BackendKind::Cuda,
        _ => polyscribe_core::backend::BackendKind::Cpu,
    }
}

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

/// A recording of about 65 seconds, built from the committed fixture.
///
/// The length can be changed with `POLYSCRIBE_TEST_AUDIO_SECONDS` for the
/// measurements.
fn long_recording() -> Vec<f32> {
    let seconds: usize = std::env::var("POLYSCRIBE_TEST_AUDIO_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(65);
    let samples = wav_samples(&fixture("speech.wav"));
    let mut out = Vec::new();
    while out.len() < seconds * SAMPLE_RATE as usize {
        out.extend_from_slice(&samples);
    }
    out.truncate(seconds * SAMPLE_RATE as usize);
    out
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Write the samples as a WAV file, so that the file path and the stream
/// can be compared.
fn write_wav(path: &Path, samples: &[f32]) {
    let mut bytes: Vec<u8> = Vec::new();
    let data_len = (samples.len() * 2) as u32;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&((sample * 32768.0) as i16).to_le_bytes());
    }
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&bytes).unwrap();
}

/// A stream that is transcribed in a thread, so that the test can watch the
/// events while the source is still delivering audio.
struct Run {
    events: Receiver<StreamEvent>,
    cancel: CancellationToken,
    handle: Option<
        std::thread::JoinHandle<
            Result<polyscribe_core::backend::TranscriptionResult, polyscribe_core::Error>,
        >,
    >,
}

impl Run {
    /// Start a transcription of the given samples.
    fn start(samples: Vec<f32>, block: usize, gate_after: Option<usize>) -> Self {
        let (sender, events) = channel();
        let mut source = ScriptedAudioSource::new(samples, block);
        if let Some(blocks) = gate_after {
            source.gate_after(blocks);
        }
        let cancel = CancellationToken::new();
        let cancel_for_thread = cancel.clone();
        let handle = std::thread::spawn(move || {
            let mut session = ModelSession::new();
            transcribe_stream(
                &mut session,
                test_backend(),
                Box::new(source),
                "test",
                Language::Explicit("en".to_string()),
                &mut |event: StreamEvent| {
                    let _ = sender.send(event);
                },
                &cancel_for_thread,
            )
        });
        Self {
            events,
            cancel,
            handle: Some(handle),
        }
    }

    /// Wait for the next event of the given kind.
    fn wait_for(&self, timeout: Duration) -> Option<StreamEvent> {
        self.events.recv_timeout(timeout).ok()
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// A transcript that grows while the source is still delivering audio.
#[test]
fn the_transcript_grows_before_the_source_is_at_its_end() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the streaming tests");
        return;
    };
    // SAFETY: the test process sets the variable before the transcription
    // thread reads it.
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let samples = long_recording();
    // One second per block, the gate closes after 60 blocks: the source has
    // delivered the first 60 seconds and stops there.
    let run = Run::start(samples, SAMPLE_RATE as usize, Some(60));

    // The first window needs 30 seconds, the second one the next 30. The
    // segments of the first window become final when the second window has
    // been transcribed, which happens while the source is held back.
    let mut finalized = 0usize;
    let mut completed = false;
    let mut seen: Vec<String> = Vec::new();
    // The decoding of a window takes seconds on a free host and a lot more
    // on a loaded one, so the waits are generous.
    let deadline = std::time::Instant::now() + Duration::from_secs(1800);
    while std::time::Instant::now() < deadline {
        match run.wait_for(Duration::from_secs(180)) {
            Some(StreamEvent::SegmentFinalized { entry }) => {
                finalized += 1;
                seen.push(format!("segment at {:.2}", entry.start));
                // The first finalized segment proves that the transcript
                // grows before the source reached its end.
                if finalized == 1 {
                    break;
                }
            }
            Some(StreamEvent::Completed { .. }) => {
                completed = true;
                seen.push("completed".to_string());
                break;
            }
            Some(event) => seen.push(
                format!("{event:?}")
                    .split_whitespace()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            None => {
                seen.push("timeout".to_string());
                break;
            }
        }
    }
    assert!(
        finalized > 0,
        "no segment was finalized while the source was held back: {seen:?}"
    );
    assert!(
        !completed,
        "the stream was completed before the first segment was finalized"
    );
}

/// The stream and the file produce the same transcript.
#[test]
fn a_stream_and_the_same_file_produce_the_same_transcript() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the streaming tests");
        return;
    };
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let samples = long_recording();
    let tmp = tempfile::tempdir().unwrap();
    let audio = tmp.path().join("recording.wav");
    write_wav(&audio, &samples);

    // The file path.
    let mut selection =
        polyscribe_core::backend::select_backend(polyscribe_core::backend::BackendKind::Cpu, false)
            .unwrap();
    let file_entries = polyscribe_core::backend::collect_entries_for_input(
        selection.backend.as_mut(),
        &audio,
        "test",
        Some("en"),
        None,
        None,
    )
    .unwrap();

    // The same audio, delivered in blocks.
    let mut session = ModelSession::new();
    let mut events = Vec::new();
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(ScriptedAudioSource::new(samples.clone(), 4096)),
        "test",
        Language::Explicit("en".to_string()),
        &mut |event: StreamEvent| events.push(event),
        &CancellationToken::new(),
    )
    .unwrap_or_else(|error| panic!("the stream failed: {error}"));

    let stream_entries: Vec<polyscribe_core::OutputEntry> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::SegmentFinalized { entry } => Some(entry.clone()),
            _ => None,
        })
        .collect();

    assert!(!file_entries.is_empty(), "the file was not transcribed");
    assert_eq!(
        result.outcome,
        polyscribe_core::stream::StreamOutcome::Completed {
            segments: stream_entries.len()
        },
        "the stream did not complete"
    );
    assert_eq!(
        file_entries.len(),
        stream_entries.len(),
        "the file and the stream produced a different number of segments"
    );
    for (file, stream) in file_entries.iter().zip(stream_entries.iter()) {
        assert_eq!(file.start, stream.start, "the segment starts differ");
        assert_eq!(file.end, stream.end, "the segment ends differ");
        assert_eq!(file.text, stream.text, "the segment texts differ");
    }
}

/// The segments of a stream are ordered and are not reported twice.
#[test]
fn the_segments_of_a_stream_are_ordered_and_unique() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the streaming tests");
        return;
    };
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let mut session = ModelSession::new();
    let mut events = Vec::new();
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(ScriptedAudioSource::new(long_recording(), 8192)),
        "test",
        Language::Explicit("en".to_string()),
        &mut |event: StreamEvent| events.push(event),
        &CancellationToken::new(),
    )
    .unwrap_or_else(|error| panic!("the stream failed: {error}"));

    let segments: Vec<polyscribe_core::OutputEntry> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::SegmentFinalized { entry } => Some(entry.clone()),
            _ => None,
        })
        .collect();

    assert!(!segments.is_empty(), "the stream was not transcribed");
    for pair in segments.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "the segments are not ordered: {:?} then {:?}",
            pair[0],
            pair[1]
        );
        assert!(
            pair[0].end <= pair[1].start + 0.2,
            "the segments overlap: {:?} then {:?}",
            pair[0],
            pair[1]
        );
    }
    // The same segment is not reported twice.
    let mut seen: Vec<(u64, u64)> = segments
        .iter()
        .map(|s| (s.start.to_bits(), s.end.to_bits()))
        .collect();
    let count = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), count, "a segment was reported twice");
    assert_eq!(result.windows, 3, "the recording has three windows");
    assert_eq!(
        result.outcome,
        polyscribe_core::stream::StreamOutcome::Completed {
            segments: segments.len()
        },
        "the stream did not complete"
    );
}

/// A cancelled stream stops at the next window and reports no provisional
/// segment as final.
#[test]
fn a_cancelled_stream_stops_and_reports_nothing_provisional() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the streaming tests");
        return;
    };
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    // The source delivers the first window and then holds the audio back.
    let mut source = ScriptedAudioSource::new(long_recording(), SAMPLE_RATE as usize);
    source.gate_after(30);
    let gate = source.gate_handle();
    let cancel = CancellationToken::new();
    let cancel_for_thread = cancel.clone();
    let (sender, events) = channel();

    let handle = std::thread::spawn(move || {
        let mut session = ModelSession::new();
        transcribe_stream(
            &mut session,
            polyscribe_core::backend::BackendKind::Cpu,
            Box::new(source),
            "test",
            Language::Explicit("en".to_string()),
            &mut |event: StreamEvent| {
                let _ = sender.send(event);
            },
            &cancel_for_thread,
        )
    });

    // Wait for the first window, then stop the run.
    let mut submitted = false;
    while let Ok(event) = events.recv_timeout(Duration::from_secs(300)) {
        if let StreamEvent::WindowSubmitted { .. } = event {
            submitted = true;
            break;
        }
    }
    assert!(submitted, "the first window was not submitted");
    cancel.cancel();
    // The source is held back; opening the gate lets the run notice the
    // cancellation.
    gate.open();

    let result = handle.join().unwrap().expect("the run did not fail");
    assert_eq!(
        result.outcome,
        polyscribe_core::stream::StreamOutcome::Cancelled {
            segments: result.entries.len()
        },
        "the run did not report a cancellation"
    );
    // The run stops before the whole recording is transcribed.
    assert!(
        result.windows < 3,
        "the run transcribed every window of the recording: {}",
        result.windows
    );
}

/// A failing source ends the run with its error.
#[test]
fn a_failing_source_ends_the_run() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the streaming tests");
        return;
    };
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let mut source = ScriptedAudioSource::new(long_recording(), SAMPLE_RATE as usize);
    source.fail_after(2, "the capture device disappeared");
    let mut session = ModelSession::new();
    let error = match transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        &mut |_event: StreamEvent| {},
        &CancellationToken::new(),
    ) {
        Ok(_) => panic!("the failing source was not noticed"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("the capture device disappeared"),
        "{error}"
    );
}

/// Measures the streaming transcription.
///
/// The test is ignored by default: it is a measurement, not a check. Run it
/// with `cargo test -- --ignored --nocapture` and a model in
/// `POLYSCRIBE_E2E_MODEL`.
#[test]
#[ignore = "a measurement, not a test"]
fn measure_the_streaming_transcription() {
    let Some(model) = e2e_model() else {
        eprintln!("SKIPPED: POLYSCRIBE_E2E_MODEL is not set; skipping the measurement");
        return;
    };
    unsafe { std::env::set_var("WHISPER_MODEL", &model) };

    let samples = long_recording();
    let audio_seconds = samples.len() as f64 / SAMPLE_RATE as f64;
    let source = ScriptedAudioSource::new(samples, 4096);
    let mut session = ModelSession::new();
    let started = std::time::Instant::now();
    let mut first_window = None;
    let mut first_segment = None;
    let mut max_buffered = 0u64;
    let mut windows = 0u64;
    let result = transcribe_stream(
        &mut session,
        test_backend(),
        Box::new(source),
        "test",
        Language::Explicit("en".to_string()),
        &mut |event: StreamEvent| match event {
            StreamEvent::SamplesReceived { total } => {
                // The audio that is not part of a window yet.
                let buffered = total.saturating_sub(windows * 30 * SAMPLE_RATE as u64);
                max_buffered = max_buffered.max(buffered);
            }
            StreamEvent::WindowSubmitted { index, .. } => {
                windows = index + 1;
                first_window.get_or_insert(started.elapsed());
            }
            StreamEvent::SegmentFinalized { .. } => {
                first_segment.get_or_insert(started.elapsed());
            }
            _ => {}
        },
        &CancellationToken::new(),
    )
    .unwrap();

    let rss = std::fs::read_to_string("/proc/self/status")
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
        .unwrap_or(0);

    println!("streaming measurement");
    println!("  audio:              {audio_seconds:.1} s");
    println!("  windows:            {}", result.windows);
    println!("  segments:           {}", result.entries.len());
    println!(
        "  time to the first window:      {:?}",
        first_window.unwrap_or_default()
    );
    println!(
        "  time to the first segment:    {:?}",
        first_segment.unwrap_or_default()
    );
    println!(
        "  max buffered audio:  {:.2} s",
        max_buffered as f64 / SAMPLE_RATE as f64
    );
    println!("  peak RSS:            {:.1} MB", rss as f64 / 1024.0);
    println!("  model loads:         1 (the session loads the model once)");
}
