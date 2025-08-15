// SPDX-License-Identifier: MIT

pub mod progress;

use std::io;
use std::io::IsTerminal;

pub fn info(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::info(m);
}

pub fn warn(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::warning(m);
}

pub fn error(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::error(m);
}

pub fn success(msg: impl AsRef<str>) {
    let m = msg.as_ref();
    let _ = cliclack::log::success(m);
}

pub fn note(prompt: impl AsRef<str>, message: impl AsRef<str>) {
    let _ = cliclack::note(prompt.as_ref(), message.as_ref());
}

pub fn intro(title: impl AsRef<str>) {
    let _ = cliclack::intro(title.as_ref());
}

pub fn outro(msg: impl AsRef<str>) {
    let _ = cliclack::outro(msg.as_ref());
}

pub fn println_above_bars(line: impl AsRef<str>) {
    let _ = cliclack::log::info(line.as_ref());
}

pub fn prompt_input(prompt: &str, default: Option<&str>) -> io::Result<String> {
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        return Ok(default.unwrap_or("").to_string());
    }
    let mut q = cliclack::input(prompt);
    if let Some(def) = default {
        q = q.default_input(def);
    }
    q.interact().map_err(|e| io::Error::other(e.to_string()))
}

pub fn prompt_select(prompt: &str, items: &[&str]) -> io::Result<usize> {
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        return Err(io::Error::other("interactive prompt disabled"));
    }
    let mut sel = cliclack::select::<usize>(prompt);
    for (idx, label) in items.iter().enumerate() {
        sel = sel.item(idx, *label, "");
    }
    sel.interact().map_err(|e| io::Error::other(e.to_string()))
}

pub fn prompt_multi_select(
    prompt: &str,
    items: &[&str],
    defaults: Option<&[bool]>,
) -> io::Result<Vec<usize>> {
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        return Err(io::Error::other("interactive prompt disabled"));
    }
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
    ms.interact().map_err(|e| io::Error::other(e.to_string()))
}

pub fn prompt_confirm(prompt: &str, default: bool) -> io::Result<bool> {
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        return Ok(default);
    }
    let mut q = cliclack::confirm(prompt);
    q.interact().map_err(|e| io::Error::other(e.to_string()))
}

pub fn prompt_password(prompt: &str) -> io::Result<String> {
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        return Err(io::Error::other(
            "password prompt disabled in non-interactive mode",
        ));
    }
    let mut q = cliclack::password(prompt);
    q.interact().map_err(|e| io::Error::other(e.to_string()))
}

pub fn prompt_input_validated<F>(
    prompt: &str,
    default: Option<&str>,
    validate: F,
) -> io::Result<String>
where
    F: Fn(&str) -> Result<(), String> + 'static,
{
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        if let Some(def) = default {
            return Ok(def.to_string());
        }
        return Err(io::Error::other("interactive prompt disabled"));
    }
    let mut q = cliclack::input(prompt);
    if let Some(def) = default {
        q = q.default_input(def);
    }
    q.validate(move |s: &String| validate(s))
        .interact()
        .map_err(|e| io::Error::other(e.to_string()))
}

pub struct Spinner(cliclack::ProgressBar);

impl Spinner {
    pub fn start(text: impl AsRef<str>) -> Self {
        if crate::is_no_progress() || crate::is_no_interaction() || !std::io::stderr().is_terminal()
        {
            let _ = cliclack::log::info(text.as_ref());
            let s = cliclack::spinner();
            Self(s)
        } else {
            let s = cliclack::spinner();
            s.start(text.as_ref());
            Self(s)
        }
    }
    pub fn stop(self, text: impl AsRef<str>) {
        let s = self.0;
        if crate::is_no_progress() {
            let _ = cliclack::log::info(text.as_ref());
        } else {
            s.stop(text.as_ref());
        }
    }
    pub fn success(self, text: impl AsRef<str>) {
        let s = self.0;
        if crate::is_no_progress() {
            let _ = cliclack::log::success(text.as_ref());
        } else {
            s.stop(text.as_ref());
        }
    }
    pub fn error(self, text: impl AsRef<str>) {
        let s = self.0;
        if crate::is_no_progress() {
            let _ = cliclack::log::error(text.as_ref());
        } else {
            s.error(text.as_ref());
        }
    }
}

pub struct BytesProgress(Option<cliclack::ProgressBar>);

impl BytesProgress {
    pub fn start(total: u64, text: &str, initial: u64) -> Self {
        if crate::is_no_progress()
            || crate::is_no_interaction()
            || !std::io::stderr().is_terminal()
            || total == 0
        {
            let _ = cliclack::log::info(text);
            return Self(None);
        }
        let b = cliclack::progress_bar(total);
        b.start(text);
        if initial > 0 {
            b.inc(initial);
        }
        Self(Some(b))
    }

    pub fn inc(&mut self, delta: u64) {
        if let Some(b) = self.0.as_mut() {
            b.inc(delta);
        }
    }

    pub fn stop(mut self, text: &str) {
        if let Some(b) = self.0.take() {
            b.stop(text);
        } else {
            let _ = cliclack::log::info(text);
        }
    }

    pub fn error(mut self, text: &str) {
        if let Some(b) = self.0.take() {
            b.error(text);
        } else {
            let _ = cliclack::log::error(text);
        }
    }
}
