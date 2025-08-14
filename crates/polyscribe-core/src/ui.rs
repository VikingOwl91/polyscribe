// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! UI helpers powered by cliclack for interactive console experiences.
//! Centralizes prompts, logging, and progress primitives.

/// Progress indicators and reporting tools for displaying task completion.
pub mod progress;

use std::io;

/// Log an informational message.
pub fn info(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::info(m);
}

/// Log a warning message.
pub fn warn(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::warning(m);
}

/// Log an error message.
pub fn error(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::error(m);
}

/// Log a success message.
pub fn success(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::success(m);
}

/// Log a note message with a prompt and a message.
pub fn note(prompt: impl AsRef<str>, message: impl AsRef<str>) {
    let _ = cliclack::note(prompt.as_ref(), message.as_ref());
}

/// Print a short intro header.
pub fn intro(title: impl AsRef<str>) {
    let _ = cliclack::intro(title.as_ref());
}

/// Print a short outro footer.
pub fn outro(msg: impl AsRef<str>) {
    let _ = cliclack::outro(msg.as_ref());
}

/// Print a line that should appear above any progress indicators.
pub fn println_above_bars(line: impl AsRef<str>) {
    let _ = cliclack::log::info(line.as_ref());
}

/// Prompt for input on stdin using cliclack's input component.
/// Returns default if provided and user enters empty string.
/// In non-interactive workflows, callers should skip prompt based on their flags.
pub fn prompt_input(prompt: &str, default: Option<&str>) -> io::Result<String> {
    let mut q = cliclack::input(prompt);
    if let Some(def) = default { q = q.default_input(def); }
    q.interact().map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
}

/// Present a single-choice selector and return the selected index.
pub fn prompt_select<'a>(prompt: &str, items: &[&'a str]) -> io::Result<usize> {
    let mut sel = cliclack::select::<usize>(prompt);
    for (idx, label) in items.iter().enumerate() {
        sel = sel.item(idx, *label, "");
    }
    sel.interact()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
}

/// Present a multi-choice selector and return indices of selected items.
pub fn prompt_multi_select<'a>(prompt: &str, items: &[&'a str], defaults: Option<&[bool]>) -> io::Result<Vec<usize>> {
    let mut ms = cliclack::multiselect::<usize>(prompt);
    for (idx, label) in items.iter().enumerate() {
        ms = ms.item(idx, *label, "");
    }
    if let Some(def) = defaults {
        let selected: Vec<usize> = def
            .iter()
            .enumerate()
            .filter_map(|(i, &on)| if on { Some(i) } else { None })
            .collect();
        if !selected.is_empty() {
            ms = ms.initial_values(selected);
        }
    }
    ms.interact()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
}

/// A simple spinner wrapper built on top of `cliclack::spinner()`.
///
/// This wrapper provides a minimal API with start/stop/success/error methods
/// to standardize spinner usage across the project.
pub struct Spinner(cliclack::ProgressBar);

impl Spinner {
    /// Creates and starts a new spinner with the provided status text.
    pub fn start(text: impl AsRef<str>) -> Self {
        let s = cliclack::spinner();
        s.start(text.as_ref());
        Self(s)
    }
    /// Stops the spinner with a submitted/completed style and message.
    pub fn stop(self, text: impl AsRef<str>) {
        let s = self.0;
        s.stop(text.as_ref());
    }
    /// Marks the spinner as successfully finished (alias for `stop`).
    pub fn success(self, text: impl AsRef<str>) {
        let s = self.0;
        // cliclack progress bar uses `stop` for successful completion styling
        s.stop(text.as_ref());
    }
    /// Marks the spinner as failed with an error style and message.
    pub fn error(self, text: impl AsRef<str>) {
        let s = self.0;
        s.error(text.as_ref());
    }
}
