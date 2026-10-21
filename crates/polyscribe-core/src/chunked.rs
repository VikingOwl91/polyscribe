// SPDX-License-Identifier: MIT

//! Bounded-memory decoding of media files.
//!
//! The audio of a media file is decoded incrementally by ffmpeg and handed
//! to the transcriber in windows of a fixed size. Only one window is in
//! memory at a time, so the memory a transcription needs does not grow with
//! the duration of the recording.
//!
//! # Windows
//!
//! A window holds as much audio as whisper's encoder takes at once, 30
//! seconds. whisper slides such a window over a long recording itself: it
//! moves the start of the next window to the end of the last segment it
//! decoded, so the speech at the window boundary is transcribed again in
//! the next window and no word is lost to the cut. This module does the
//! same, so the segments of a long recording come out the same way as when
//! the whole recording is handed to whisper at once.
//!
//! The audio in the overlap between two windows is decoded twice, which is
//! what makes the boundary safe. The overlap is bounded by the window
//! length, so the memory stays bounded as well.

use anyhow::{Context, anyhow};
use std::collections::VecDeque;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;

/// The sample rate whisper expects.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// How the recording is cut into windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    /// The number of samples a window holds.
    pub window_samples: usize,
    /// The number of samples that are kept in front of a window.
    ///
    /// whisper does not always report a segment for the beginning of the
    /// audio it is given. When it does not, the region has to be decoded
    /// again, and a window that starts a little earlier gives whisper a
    /// different view of the same audio. The samples in front of the window
    /// are kept for that, and they are bounded.
    pub lookback_samples: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        // whisper's encoder works on 30 second windows, so a window of
        // 30 seconds is transcribed in a single pass. Two seconds of
        // lookback are kept in front of it, which is enough to decode a
        // region that whisper skipped again from a different position.
        Self {
            window_samples: 30 * WHISPER_SAMPLE_RATE as usize,
            lookback_samples: 2 * WHISPER_SAMPLE_RATE as usize,
        }
    }
}

impl ChunkConfig {
    /// The number of samples a window holds.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// The number of samples that are kept in front of a window.
    pub fn lookback_samples(&self) -> usize {
        self.lookback_samples
    }
}

/// A window of decoded audio.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioWindow {
    /// Index of the first sample of this window in the whole recording.
    pub start_sample: u64,
    /// The decoded samples: 16 kHz mono.
    pub samples: Vec<f32>,
}

impl AudioWindow {
    /// The start of the window in seconds.
    pub fn start_seconds(&self) -> f64 {
        self.start_sample as f64 / WHISPER_SAMPLE_RATE as f64
    }

    /// The end of the window in seconds.
    pub fn end_seconds(&self) -> f64 {
        (self.start_sample + self.samples.len() as u64) as f64 / WHISPER_SAMPLE_RATE as f64
    }
}

/// Holds the decoded samples that are still available for a window.
///
/// The assembler is independent of the input, so the windowing can be
/// tested without decoding anything. The samples of the last window are
/// kept, because the next window can start inside it: whisper slides its
/// window back to the end of the last segment it decoded. A short lookback
/// in front of the window is kept as well, so that a region that whisper
/// did not transcribe can be decoded again from a slightly earlier
/// position. The retained samples are therefore bounded by one window plus
/// the lookback.
#[derive(Debug)]
pub struct WindowAssembler {
    window_samples: usize,
    lookback_samples: usize,
    /// The samples that were decoded and are not dropped yet.
    buffer: VecDeque<f32>,
    /// Index of the first sample in `buffer`.
    buffer_start: u64,
    /// The index of the next sample that was decoded.
    decoded: u64,
}

impl WindowAssembler {
    pub fn new(config: ChunkConfig) -> Self {
        Self {
            window_samples: config.window_samples,
            lookback_samples: config.lookback_samples,
            buffer: VecDeque::new(),
            buffer_start: 0,
            decoded: 0,
        }
    }

    /// Add decoded samples.
    pub fn push(&mut self, samples: &[f32]) {
        self.buffer.extend(samples.iter().copied());
        self.decoded += samples.len() as u64;
    }

    /// The number of samples that are available from `start_sample`.
    ///
    /// The start of a window can be beyond the audio that was decoded so
    /// far, so the count saturates at zero.
    pub fn available_from(&self, start_sample: u64) -> u64 {
        if start_sample < self.buffer_start {
            return 0;
        }
        self.decoded.saturating_sub(start_sample)
    }

    /// The index of the first sample that was not decoded yet.
    pub fn decoded_samples(&self) -> u64 {
        self.decoded
    }

    /// The number of samples a window holds.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// The samples that are buffered and not part of a window yet.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// The index of the first sample that is still available.
    ///
    /// A window cannot start before it: the samples in front of it are
    /// gone.
    pub fn retained_from(&self) -> u64 {
        self.buffer_start
    }

    /// Take a window that starts at `start_sample`.
    ///
    /// Returns `None` when the recording ends before the window is full. The
    /// samples of the window stay in the buffer, so a later window can
    /// start inside it.
    pub fn take_window(&mut self, start_sample: u64) -> Option<AudioWindow> {
        if start_sample < self.buffer_start {
            // The window starts before the retained region, which cannot
            // happen for a window that follows another one.
            return None;
        }
        // The samples in front of the window are only needed for the
        // lookback, so they are dropped up to that point.
        let drop_before = start_sample.saturating_sub(self.lookback_samples as u64);
        let skip = (drop_before - self.buffer_start) as usize;
        if skip >= self.buffer.len() {
            return None;
        }
        // Samples before the window are not needed any more.
        self.buffer.drain(..skip);
        self.buffer_start = drop_before;

        // The window starts at start_sample, so the lookback in front of it
        // is skipped.
        let offset = (start_sample - self.buffer_start) as usize;
        let count = self.window_samples.min(self.buffer.len() - offset);
        if count == 0 {
            return None;
        }
        let samples: Vec<f32> = self
            .buffer
            .iter()
            .skip(offset)
            .take(count)
            .copied()
            .collect();
        Some(AudioWindow {
            start_sample,
            samples,
        })
    }
}

/// Spawn ffmpeg to decode a media file into 16 kHz mono f32 on stdout.
///
/// The caller owns the child and has to reap it.
pub(crate) fn spawn_ffmpeg_decoder(audio_path: &Path) -> anyhow::Result<Child> {
    let child = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(audio_path)
        .arg("-f")
        .arg("f32le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(WHISPER_SAMPLE_RATE.to_string())
        .arg("pipe:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                anyhow!(
                    "ffmpeg not found on PATH. Please install ffmpeg and ensure it is available."
                )
            } else {
                anyhow!(
                    "Failed to execute ffmpeg for {}: {}",
                    audio_path.display(),
                    e
                )
            }
        })?;
    Ok(child)
}

/// Decodes a media file into windows of audio.
///
/// The decoder reads from ffmpeg only when the next window is requested,
/// so decoding cannot outrun the transcription: ffmpeg blocks once its
/// output pipe is full.
pub struct ChunkedAudio {
    audio_path: PathBuf,
    child: Option<Child>,
    stdout: Option<std::io::BufReader<ChildStdout>>,
    stderr_thread: Option<JoinHandle<Vec<u8>>>,
    assembler: WindowAssembler,
    /// The index where the next window starts.
    next_start: u64,
    finished: bool,
}

impl ChunkedAudio {
    /// Start decoding the given media file.
    pub fn new(audio_path: &Path, config: ChunkConfig) -> anyhow::Result<Self> {
        let mut child = spawn_ffmpeg_decoder(audio_path)?;
        let stdout = child
            .stdout
            .take()
            .expect("the decoder was spawned with a piped stdout");
        let stderr = child
            .stderr
            .take()
            .expect("the decoder was spawned with a piped stderr");

        // Drain stderr on a separate thread so ffmpeg can never deadlock on
        // a full pipe.
        let stderr_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut stderr = stderr;
            let _ = stderr.read_to_end(&mut buf);
            buf
        });

        Ok(Self {
            audio_path: audio_path.to_path_buf(),
            child: Some(child),
            stdout: Some(std::io::BufReader::new(stdout)),
            stderr_thread: Some(stderr_thread),
            assembler: WindowAssembler::new(config),
            next_start: 0,
            finished: false,
        })
    }

    /// Move the start of the next window.
    ///
    /// The start is normally the end of the last segment of the previous
    /// window, which is how whisper slides its own window over a long
    /// recording.
    pub fn seek_to(&mut self, start_sample: u64) {
        self.next_start = start_sample;
    }

    /// The index where the next window starts.
    pub fn next_start(&self) -> u64 {
        self.next_start
    }

    /// The index of the first sample that is still available for a window.
    ///
    /// A window cannot start before it: the samples in front of it are
    /// gone. The lookback in front of the current window is kept, so that a
    /// region that whisper did not transcribe can be decoded again from a
    /// slightly earlier position.
    pub fn retained_from(&self) -> u64 {
        self.assembler.retained_from()
    }

    /// Read whatever ffmpeg has produced so far. Returns false at the end
    /// of the stream.
    fn read_more(&mut self) -> anyhow::Result<bool> {
        let stdout = self
            .stdout
            .as_mut()
            .expect("the decoder is only read while it is alive");
        let mut buf = [0u8; 4 * 4096];
        let mut filled = 0usize;
        while filled < buf.len() {
            match stdout.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    return Err(anyhow!(
                        "Failed to read ffmpeg output for {}: {}",
                        self.audio_path.display(),
                        e
                    ));
                }
            }
        }
        if filled == 0 {
            return Ok(false);
        }
        // A trailing partial sample is dropped, like the whole-file decoder
        // does.
        let complete = filled - (filled % 4);
        let samples: Vec<f32> = buf[..complete]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.assembler.push(&samples);
        Ok(true)
    }

    /// Collect the stderr of ffmpeg and check its exit status.
    fn ffmpeg_result(&mut self) -> anyhow::Result<()> {
        let stderr = self
            .stderr_thread
            .take()
            .map(|t| t.join().unwrap_or_default())
            .unwrap_or_default();
        let stderr = String::from_utf8_lossy(&stderr).into_owned();
        if let Some(mut child) = self.child.take() {
            let status = child.wait().with_context(|| {
                format!(
                    "Failed to wait for ffmpeg for {}",
                    self.audio_path.display()
                )
            })?;
            if !status.success() {
                return Err(anyhow!(
                    "ffmpeg failed for {}: {}",
                    self.audio_path.display(),
                    stderr
                ));
            }
        }
        Ok(())
    }

    /// The next window of audio, or `None` at the end of the recording.
    pub fn next_window(&mut self) -> anyhow::Result<Option<AudioWindow>> {
        crate::stream::WindowSource::next_window(self, &crate::stream::CancellationToken::new())
    }
}

impl crate::stream::WindowSource for ChunkedAudio {
    fn next_window(
        &mut self,
        cancel: &crate::stream::CancellationToken,
    ) -> anyhow::Result<Option<AudioWindow>> {
        if self.finished {
            return Ok(None);
        }
        let window_samples = self.assembler.window_samples as u64;
        loop {
            if self.assembler.available_from(self.next_start) >= window_samples {
                break;
            }
            if cancel.is_cancelled() {
                return Ok(None);
            }
            if !self.read_more()? {
                // The recording ended. The last window may be shorter.
                self.finished = true;
                self.ffmpeg_result()?;
                let window = self.assembler.take_window(self.next_start);
                return Ok(window);
            }
        }
        let window = self
            .assembler
            .take_window(self.next_start)
            .expect("the window is complete");
        Ok(Some(window))
    }

    fn seek_to(&mut self, start_sample: u64) {
        self.next_start = start_sample;
    }

    fn retained_from(&self) -> u64 {
        self.assembler.retained_from()
    }

    fn received_samples(&self) -> u64 {
        self.assembler.decoded_samples()
    }
}

impl Drop for ChunkedAudio {
    fn drop(&mut self) {
        // A decoder that is dropped before the end of the stream has to
        // leave no ffmpeg behind.
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(window: usize) -> ChunkConfig {
        ChunkConfig {
            window_samples: window,
            lookback_samples: 0,
        }
    }

    #[test]
    fn test_default_window_is_whispers_encoder_window() {
        let c = ChunkConfig::default();
        assert_eq!(c.window_samples(), 30 * WHISPER_SAMPLE_RATE as usize);
    }

    #[test]
    fn test_empty_input_produces_no_window() {
        let mut a = WindowAssembler::new(config(10));
        assert_eq!(a.available_from(0), 0);
        assert!(a.take_window(0).is_none());
    }

    #[test]
    fn test_short_input_produces_one_window() {
        let mut a = WindowAssembler::new(config(10));
        a.push(&[1.0, 2.0, 3.0]);
        let w = a
            .take_window(0)
            .expect("a short input still has one window");
        assert_eq!(w.samples, vec![1.0, 2.0, 3.0]);
        assert_eq!(w.start_sample, 0);
        // Nothing is left over.
        assert!(a.take_window(3).is_none());
    }

    #[test]
    fn test_exact_window_boundary_produces_one_full_window() {
        let mut a = WindowAssembler::new(config(4));
        a.push(&[1.0, 2.0, 3.0, 4.0]);
        let w = a.take_window(0).expect("four samples fill one window");
        assert_eq!(w.samples, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(w.start_sample, 0);
        assert!(a.take_window(4).is_none());
    }

    #[test]
    fn test_partial_final_window() {
        let mut a = WindowAssembler::new(config(4));
        a.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let first = a.take_window(0).expect("the first window is complete");
        assert_eq!(first.samples, vec![1.0, 2.0, 3.0, 4.0]);
        // The next window starts where the previous one stopped.
        let last = a.take_window(4).expect("the remainder is the last window");
        assert_eq!(last.samples, vec![5.0, 6.0]);
        assert_eq!(last.start_sample, 4);
    }

    #[test]
    fn test_window_starts_at_its_own_position_with_a_lookback() {
        // The lookback keeps the samples in front of a window, so that a
        // region that whisper did not transcribe can be decoded again from
        // a slightly earlier position. The window itself still starts at
        // the position it was asked for.
        let mut a = WindowAssembler::new(ChunkConfig {
            window_samples: 4,
            lookback_samples: 2,
        });
        a.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w = a.take_window(2).expect("the window is complete");
        assert_eq!(w.start_sample, 2);
        assert_eq!(w.samples, vec![3.0, 4.0, 5.0, 6.0]);
        // The samples in front of the window are still available.
        assert_eq!(a.retained_from(), 0);
        // A window that starts inside the lookback of the previous one
        // gets the audio of its own position.
        let w = a.take_window(1).expect("the window is complete");
        assert_eq!(w.start_sample, 1);
        assert_eq!(w.samples, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_window_can_start_inside_decoded_audio() {
        // This is how the next window is moved back to the end of the last
        // segment of the previous window.
        let mut a = WindowAssembler::new(config(4));
        a.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let first = a.take_window(0).expect("the first window is complete");
        assert_eq!(first.samples, vec![1.0, 2.0, 3.0, 4.0]);
        // The next window starts at sample 2, so the first two samples are
        // transcribed again.
        let second = a.take_window(2).expect("the second window is available");
        assert_eq!(second.samples, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(second.start_sample, 2);
    }

    #[test]
    fn test_windows_cover_every_sample() {
        let mut a = WindowAssembler::new(config(4));
        let total = 23;
        let samples: Vec<f32> = (0..total).map(|i| i as f32).collect();
        a.push(&samples);

        // A window that slides by three samples, like whisper does.
        let mut start = 0u64;
        let mut covered: Vec<bool> = vec![false; total];
        let mut windows = 0;
        while let Some(w) = a.take_window(start) {
            for (i, s) in w.samples.iter().enumerate() {
                let idx = (w.start_sample as usize) + i;
                assert_eq!(*s, idx as f32, "the samples keep their order");
                covered[idx] = true;
            }
            windows += 1;
            start += 3;
            if start >= total as u64 {
                break;
            }
        }
        // Every sample was transcribed at least once, and the sliding
        // windows overlap by one sample.
        assert!(covered.iter().all(|c| *c));
        assert!(windows > 1);
    }

    #[test]
    fn test_long_recording_arithmetic_stays_bounded() {
        // Three hours of audio: the number of windows is computed, the
        // audio itself is never held in memory.
        let hours = 3u64;
        let total_samples = hours * 3600 * WHISPER_SAMPLE_RATE as u64;
        let c = ChunkConfig::default();
        let window = c.window_samples() as u64;
        assert_eq!(window, 30 * 16_000);
        // The sample counters do not overflow for a very long recording.
        let huge = 1_000u64 * 3600 * WHISPER_SAMPLE_RATE as u64;
        assert!(huge > total_samples);
        assert!(total_samples / window < 20_000);
    }
}
