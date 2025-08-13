// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! Centralized UI helpers (TTY-aware, quiet/verbose-aware)

use std::io;

/// Startup intro/banner (suppressed when quiet).
pub fn intro(msg: impl AsRef<str>) {
    let _ = cliclack::intro(msg.as_ref());
}

/// Final outro/summary printed below any progress indicators (suppressed when quiet).
pub fn outro(msg: impl AsRef<str>) {
    let _ = cliclack::outro(msg.as_ref());
}

/// Info message (TTY-aware; suppressed by --quiet is handled by outer callers if needed)
pub fn info(msg: impl AsRef<str>) {
    let _ = cliclack::log::info(msg.as_ref());
}

/// Print a warning (always printed).
pub fn warn(msg: impl AsRef<str>) {
    // cliclack provides a warning-level log utility
    let _ = cliclack::log::warning(msg.as_ref());
}

/// Print an error (always printed).
pub fn error(msg: impl AsRef<str>) {
    let _ = cliclack::log::error(msg.as_ref());
}

/// Print a line above any progress bars (maps to cliclack log; synchronized).
pub fn println_above_bars(msg: impl AsRef<str>) {
    if crate::is_quiet() { return; }
    // cliclack logs are synchronized with its spinners/bars
    let _ = cliclack::log::info(msg.as_ref());
}

/// Input prompt with a question: returns Ok(None) if non-interactive or canceled
pub fn prompt_input(question: impl AsRef<str>, default: Option<&str>) -> anyhow::Result<Option<String>> {
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        return Ok(None);
    }
    let mut p = cliclack::input(question.as_ref());
    if let Some(d) = default {
        // Use default_input when available in 0.3.x
        p = p.default_input(d);
    }
    match p.interact() {
        Ok(s) => Ok(Some(s)),
        Err(_) => Ok(None),
    }
}

/// Confirmation prompt; returns Ok(None) if non-interactive or canceled
pub fn prompt_confirm(question: impl AsRef<str>, default_yes: bool) -> anyhow::Result<Option<bool>> {
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        return Ok(None);
    }
    let res = cliclack::confirm(question.as_ref())
        .initial_value(default_yes)
        .interact();
    match res {
        Ok(v) => Ok(Some(v)),
        Err(_) => Ok(None),
    }
}

/// Prompt the user (TTY-aware via cliclack) and read a line from stdin. Returns the raw line with trailing newline removed.
pub fn prompt_line(prompt: &str) -> io::Result<String> {
    // Route prompt through cliclack to keep consistent styling and avoid direct eprint!/println!
    let _ = cliclack::log::info(prompt);
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    Ok(s)
}

/// TTY-aware progress UI built on `indicatif` for per-file and aggregate progress bars.
///
/// This small helper encapsulates a `MultiProgress` with one aggregate (total) bar and
/// one per-file bar. It is intentionally minimal to keep integration lightweight.
pub mod progress {
    // The submodule is defined in a separate file for clarity.
    include!("ui/progress.rs");
}
