// SPDX-License-Identifier: MIT

use std::io::IsTerminal as _;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct FileProgress {
    enabled: bool,
    file_bars: Vec<cliclack::ProgressBar>,
    total_bar: Option<cliclack::ProgressBar>,
    completed: usize,
    total_file_count: usize,
}

impl FileProgress {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            file_bars: Vec::new(),
            total_bar: None,
            completed: 0,
            total_file_count: 0,
        }
    }

    pub fn default_for_files(file_count: usize) -> Self {
        let enabled = file_count > 1
            && std::io::stderr().is_terminal()
            && !crate::is_quiet()
            && !crate::is_no_progress();
        Self::new(enabled)
    }

    pub fn init_files(&mut self, labels: &[String]) {
        self.total_file_count = labels.len();
        if !self.enabled || labels.len() <= 1 {
            self.enabled = false;
            return;
        }
        let total = cliclack::progress_bar(labels.len() as u64);
        total.start("Total");
        self.total_bar = Some(total);
        for label in labels {
            let pb = cliclack::progress_bar(100);
            pb.start(label);
            self.file_bars.push(pb);
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_file_message(&mut self, idx: usize, message: &str) {
        if !self.enabled {
            return;
        }
        if let Some(pb) = self.file_bars.get_mut(idx) {
            pb.set_message(message);
        }
    }

    pub fn set_file_percent(&mut self, idx: usize, percent: u64) {
        if !self.enabled {
            return;
        }
        if let Some(pb) = self.file_bars.get_mut(idx) {
            let p = percent.min(100);
            pb.set_message(format!("{p}%"));
        }
    }

    pub fn mark_file_done(&mut self, idx: usize) {
        if !self.enabled {
            return;
        }
        if let Some(pb) = self.file_bars.get_mut(idx) {
            pb.stop("done");
        }
        self.completed += 1;
        if let Some(total) = &mut self.total_bar {
            total.inc(1);
            if self.completed >= self.total_file_count {
                total.stop("all done");
            }
        }
    }

    pub fn finish_total(&mut self, message: &str) {
        if !self.enabled {
            return;
        }
        if let Some(total) = &mut self.total_bar {
            total.stop(message);
        }
    }
}

#[derive(Debug)]
pub struct ProgressReporter {
    non_interactive: bool,
}

impl ProgressReporter {
    pub fn new(non_interactive: bool) -> Self {
        Self { non_interactive }
    }

    pub fn step(&mut self, message: &str) {
        if crate::is_quiet() {
            return;
        }
        if self.non_interactive {
            let _ = cliclack::log::info(format!("[..] {message}"));
        } else {
            let _ = cliclack::log::info(format!("• {message}"));
        }
    }

    pub fn finish_with_message(&mut self, message: &str) {
        if crate::is_quiet() {
            return;
        }
        if self.non_interactive {
            let _ = cliclack::log::info(format!("[ok] {message}"));
        } else {
            let _ = cliclack::log::info(format!("✓ {message}"));
        }
    }
}

struct TranscriptProgressState {
    bar: Option<cliclack::ProgressBar>,
    started: Instant,
    inference_started: Option<Instant>,
    last_percent: i32,
    last_draw: Instant,
}

/// A stderr-only progress bar for one transcription.
///
/// It shows the current stage, elapsed time, and an ETA derived from the
/// percentage reported by the backend. It is silent unless stderr is a
/// terminal and neither `--quiet` nor `--no-progress` is active, so stdout
/// stays transcript-only and pipes stay clean.
pub struct TranscriptProgress {
    state: Mutex<TranscriptProgressState>,
}

impl TranscriptProgress {
    pub fn start(label: impl AsRef<str>) -> Self {
        let enabled =
            !crate::is_quiet() && !crate::is_no_progress() && std::io::stderr().is_terminal();
        let bar = enabled.then(|| {
            let bar =
                cliclack::progress_bar(100).with_template("{msg} {bar:30.magenta} {percent}%");
            bar.start(label.as_ref());
            bar
        });
        Self {
            state: Mutex::new(TranscriptProgressState {
                bar,
                started: Instant::now(),
                inference_started: None,
                last_percent: -1,
                last_draw: Instant::now(),
            }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.bar.is_some())
            .unwrap_or(false)
    }

    pub fn update(&self, percent: i32) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let Some(bar) = state.bar.clone() else {
            return;
        };
        let now = Instant::now();
        let percent = percent.clamp(0, 100);
        if percent == state.last_percent
            && now.duration_since(state.last_draw) < Duration::from_millis(100)
        {
            return;
        }
        state.last_percent = percent;
        state.last_draw = now;
        if percent >= 40 && state.inference_started.is_none() {
            state.inference_started = Some(now);
        }
        let elapsed = state.started.elapsed();
        let inference_elapsed = state
            .inference_started
            .map(|started| now.duration_since(started));
        bar.set_position(percent as u64);
        bar.set_message(transcript_message(percent, elapsed, inference_elapsed));
    }

    pub fn finish(&self, message: impl AsRef<str>) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let elapsed = state.started.elapsed();
        if let Some(bar) = state.bar.take() {
            bar.set_position(100);
            bar.stop(format!(
                "{} ({})",
                message.as_ref(),
                format_elapsed(elapsed)
            ));
        }
    }
}

fn transcript_message(
    percent: i32,
    elapsed: Duration,
    inference_elapsed: Option<Duration>,
) -> String {
    let stage = match percent {
        0..=4 => "preparing audio",
        5..=19 => "audio decoded; loading model",
        20..=29 => "model loaded; creating state",
        30..=39 => "starting inference",
        40..=89 => "transcribing",
        90..=99 => "finalizing",
        _ => "done",
    };
    match estimate_remaining(percent, inference_elapsed) {
        Some(eta) => format!(
            "{stage} | elapsed {} | ETA {}",
            format_elapsed(elapsed),
            format_elapsed(eta)
        ),
        None => format!("{stage} | elapsed {}", format_elapsed(elapsed)),
    }
}

fn estimate_remaining(percent: i32, inference_elapsed: Option<Duration>) -> Option<Duration> {
    let inference_elapsed = inference_elapsed?;
    if !(40..90).contains(&percent) {
        return None;
    }
    let fraction = (percent - 40) as f64 / 50.0;
    if fraction <= 0.0 {
        return None;
    }
    let elapsed = inference_elapsed.as_secs_f64();
    let remaining = elapsed * (1.0 - fraction) / fraction;
    remaining
        .is_finite()
        .then(|| Duration::from_secs_f64(remaining.max(0.0)))
}

fn format_elapsed(duration: Duration) -> String {
    let total = duration.as_secs();
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_elapsed_time() {
        assert_eq!(format_elapsed(Duration::from_secs(5)), "00:05");
        assert_eq!(format_elapsed(Duration::from_secs(65)), "01:05");
        assert_eq!(format_elapsed(Duration::from_secs(3661)), "1:01:01");
    }

    #[test]
    fn estimates_remaining_from_inference_progress() {
        assert_eq!(estimate_remaining(5, Some(Duration::from_secs(10))), None);
        assert_eq!(estimate_remaining(40, Some(Duration::from_secs(10))), None);
        let eta = estimate_remaining(65, Some(Duration::from_secs(10))).unwrap();
        assert!((eta.as_secs_f64() - 10.0 * 0.5 / 0.5).abs() < 0.01);
        assert_eq!(estimate_remaining(100, Some(Duration::from_secs(10))), None);
    }

    #[test]
    fn transcript_message_names_the_stage_and_eta() {
        let message =
            transcript_message(65, Duration::from_secs(10), Some(Duration::from_secs(10)));
        assert!(message.contains("transcribing"));
        assert!(message.contains("elapsed 00:10"));
        assert!(message.contains("ETA 00:10"));
    }
}
