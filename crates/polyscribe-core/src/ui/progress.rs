// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

use std::io::IsTerminal as _;

/// Manages a set of per-file progress bars plus a top aggregate bar using cliclack.
pub struct ProgressManager {
    enabled: bool,
    per: Vec<cliclack::ProgressBar>,
    total: Option<cliclack::ProgressBar>,
    completed: usize,
    total_len: usize,
}

impl ProgressManager {
    /// Create a new manager with the given enabled flag.
    pub fn new(enabled: bool) -> Self {
        Self { enabled, per: Vec::new(), total: None, completed: 0, total_len: 0 }
    }

    /// Create a manager that enables bars when `n > 1`, stderr is a TTY, and not quiet.
    pub fn default_for_files(n: usize) -> Self {
        let enabled = n > 1 && std::io::stderr().is_terminal() && !crate::is_quiet() && !crate::is_no_progress();
        Self::new(enabled)
    }

    /// Initialize bars for the given file labels. If disabled or single file, no-op.
    pub fn init_files(&mut self, labels: &[String]) {
        self.total_len = labels.len();
        if !self.enabled || labels.len() <= 1 {
            // No bars in single-file mode or when disabled
            self.enabled = false;
            return;
        }
        // Aggregate bar at the top
        let mut total = cliclack::progress_bar(labels.len() as u64);
        total.start("Total");
        self.total = Some(total);
        // Per-file bars (100% scale for each)
        for label in labels {
            let mut pb = cliclack::progress_bar(100);
            pb.start(label);
            self.per.push(pb);
        }
    }

    /// Returns true when bars are enabled (multi-file TTY mode).
    pub fn is_enabled(&self) -> bool { self.enabled }

    /// Update a per-file bar message.
    pub fn set_per_message(&mut self, idx: usize, message: &str) {
        if !self.enabled { return; }
        if let Some(pb) = self.per.get_mut(idx) {
            pb.set_message(message);
        }
    }

    /// Update a per-file bar percent (0..=100).
    pub fn set_per_percent(&mut self, idx: usize, percent: u64) {
        if !self.enabled { return; }
        if let Some(pb) = self.per.get_mut(idx) {
            let p = percent.min(100);
            pb.set_message(&format!("{p}%"));
        }
    }

    /// Mark a file as finished (set to 100% and update total counter).
    pub fn mark_file_done(&mut self, idx: usize) {
        if !self.enabled { return; }
        if let Some(pb) = self.per.get_mut(idx) {
            pb.stop("done");
        }
        self.completed += 1;
        if let Some(total) = &mut self.total {
            total.inc(1);
            if self.completed >= self.total_len {
                total.stop("all done");
            }
        }
    }

    /// Finish the aggregate bar with a custom message.
    pub fn finish_total(&mut self, message: &str) {
        if !self.enabled { return; }
        if let Some(total) = &mut self.total {
            total.stop(message);
        }
    }
}

/// A simple reporter for displaying progress messages using cliclack logging.
#[derive(Debug)]
pub struct ProgressReporter {
    non_interactive: bool,
}

impl ProgressReporter {
    /// Creates a new progress reporter.
    pub fn new(non_interactive: bool) -> Self { Self { non_interactive } }

    /// Displays a progress step message.
    pub fn step(&mut self, message: &str) {
        if self.non_interactive {
            let _ = cliclack::log::info(format!("[..] {message}"));
        } else {
            let _ = cliclack::log::info(format!("• {message}"));
        }
    }

    /// Displays a completion message.
    pub fn finish_with_message(&mut self, message: &str) {
        if self.non_interactive {
            let _ = cliclack::log::info(format!("[ok] {message}"));
        } else {
            let _ = cliclack::log::info(format!("✓ {message}"));
        }
    }
}
