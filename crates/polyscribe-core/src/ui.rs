// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! Minimal UI helpers used across the core crate.
//! This keeps interactive bits centralized and easy to stub in tests.

/// Progress indicators and reporting tools for displaying task completion.
pub mod progress;

use std::io::{self, Write};

/// Print an informational line to stderr (suppressed when quiet mode is enabled by callers).
pub fn info(msg: impl AsRef<str>) {
    eprintln!("{}", msg.as_ref());
}

/// Print a warning line to stderr.
pub fn warn(msg: impl AsRef<str>) {
    eprintln!("WARNING: {}", msg.as_ref());
}

/// Print an error line to stderr.
pub fn error(msg: impl AsRef<str>) {
    eprintln!("ERROR: {}", msg.as_ref());
}

/// Print a short intro header (non-fancy).
pub fn intro(title: impl AsRef<str>) {
    eprintln!("== {} ==", title.as_ref());
}

/// Print a short outro footer (non-fancy).
pub fn outro(msg: impl AsRef<str>) {
    eprintln!("{}", msg.as_ref());
}

/// Print a line that should appear above any progress indicators (plain for now).
pub fn println_above_bars(line: impl AsRef<str>) {
    eprintln!("{}", line.as_ref());
}

/// Prompt for input on stdin. Returns default if provided and user enters empty string.
/// In non-interactive workflows, callers should skip prompt based on their flags.
pub fn prompt_input(prompt: &str, default: Option<&str>) -> io::Result<String> {
    let mut stdout = io::stdout();
    match default {
        Some(def) => {
            write!(stdout, "{} [{}]: ", prompt, def)?;
        }
        None => {
            write!(stdout, "{}: ", prompt)?;
        }
    }
    stdout.flush()?;

    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        Ok(default.unwrap_or_default().to_string())
    } else {
        Ok(trimmed.to_string())
    }
}
