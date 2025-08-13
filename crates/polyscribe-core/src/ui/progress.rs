// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::io::IsTerminal as _;

/// Manages a set of per-file progress bars plus a top aggregate bar.
pub struct ProgressManager {
    enabled: bool,
    mp: Option<MultiProgress>,
    per: Vec<ProgressBar>,
    total: Option<ProgressBar>,
    completed: usize,
}

impl ProgressManager {
    /// Create a new manager with the given enabled flag.
    pub fn new(enabled: bool) -> Self {
        Self { enabled, mp: None, per: Vec::new(), total: None, completed: 0 }
    }

    /// Create a manager that enables bars when `n > 1`, stderr is a TTY, and not quiet.
    pub fn default_for_files(n: usize) -> Self {
        let enabled = n > 1 && std::io::stderr().is_terminal() && !crate::is_quiet() && !crate::is_no_progress();
        Self::new(enabled)
    }

    /// Initialize bars for the given file labels. If disabled or single file, no-op.
    pub fn init_files(&mut self, labels: &[String]) {
        if !self.enabled || labels.len() <= 1 {
            // No bars in single-file mode or when disabled
            self.enabled = false;
            return;
        }
        let mp = MultiProgress::new();
        // Aggregate bar at the top
        let total = mp.add(ProgressBar::new(labels.len() as u64));
        total.set_style(ProgressStyle::with_template("{prefix} [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("=>-"));
        total.set_prefix("Total");
        self.total = Some(total);
        // Per-file bars
        for label in labels {
            let pb = mp.add(ProgressBar::new(100));
            pb.set_style(ProgressStyle::with_template("{prefix} [{bar:40.green/black}] {pos}% {msg}")
                .unwrap()
                .progress_chars("=>-"));
            pb.set_position(0);
            pb.set_prefix(label.clone());
            self.per.push(pb);
        }
        self.mp = Some(mp);
    }

    /// Returns true when bars are enabled (multi-file TTY mode).
    pub fn is_enabled(&self) -> bool { self.enabled }

    /// Get a clone of the per-file progress bar at index, if enabled.
    pub fn per_bar(&self, idx: usize) -> Option<ProgressBar> {
        if !self.enabled { return None; }
        self.per.get(idx).cloned()
    }

    /// Get a clone of the aggregate (total) progress bar, if enabled.
    pub fn total_bar(&self) -> Option<ProgressBar> {
        if !self.enabled { return None; }
        self.total.as_ref().cloned()
    }

    /// Mark a file as finished (set to 100% and update total counter).
    pub fn mark_file_done(&mut self, idx: usize) {
        if !self.enabled { return; }
        if let Some(pb) = self.per.get(idx) {
            pb.set_position(100);
            pb.finish_with_message("done");
        }
        self.completed += 1;
        if let Some(total) = &self.total { total.set_position(self.completed as u64); }
    }
}

/// A simple reporter for displaying progress messages in the terminal.
/// Provides different output formatting based on whether the environment is interactive or not.
#[derive(Debug)]
pub struct ProgressReporter {
    non_interactive: bool,
}

impl ProgressReporter {
    /// Creates a new progress reporter.
    ///
    /// # Arguments
    ///
    /// * `non_interactive` - Whether the output should be formatted for non-interactive environments.
    pub fn new(non_interactive: bool) -> Self {
        Self { non_interactive }
    }

    /// Displays a progress step message.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to display for this progress step.
    pub fn step(&mut self, message: &str) {
        if self.non_interactive {
            eprintln!("[..] {message}");
        } else {
            eprintln!("• {message}");
        }
    }

    /// Displays a completion message.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to display when a task is completed.
    pub fn finish_with_message(&mut self, message: &str) {
        if self.non_interactive {
            eprintln!("[ok] {message}");
        } else {
            eprintln!("✓ {message}");
        }
    }
}
