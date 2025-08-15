// SPDX-License-Identifier: MIT

use std::io::IsTerminal as _;

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
        if self.non_interactive {
            let _ = cliclack::log::info(format!("[..] {message}"));
        } else {
            let _ = cliclack::log::info(format!("• {message}"));
        }
    }

    pub fn finish_with_message(&mut self, message: &str) {
        if self.non_interactive {
            let _ = cliclack::log::info(format!("[ok] {message}"));
        } else {
            let _ = cliclack::log::info(format!("✓ {message}"));
        }
    }
}
