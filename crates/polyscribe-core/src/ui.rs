// SPDX-License-Identifier: MIT

pub mod progress;

use std::io;
use std::io::IsTerminal;
use std::io::Write as _;
use std::time::{Duration, Instant};

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

pub struct BytesProgress {
    enabled: bool,
    total: u64,
    current: u64,
    started: Instant,
    last_msg: Instant,
    width: usize,
    // Sticky ETA to carry through zero-speed stalls
    last_eta_secs: Option<f64>,
}

impl BytesProgress {
    pub fn start(total: u64, text: &str, initial: u64) -> Self {
        let enabled = !(crate::is_no_progress()
            || crate::is_no_interaction()
            || !std::io::stderr().is_terminal()
            || total == 0);
        if !enabled {
            let _ = cliclack::log::info(text);
        }
        let mut me = Self {
            enabled,
            total,
            current: initial.min(total),
            started: Instant::now(),
            last_msg: Instant::now(),
            width: 40,
            last_eta_secs: None,
        };
        me.draw();
        me
    }

    fn human_bytes(n: u64) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = 1024.0 * KB;
        const GB: f64 = 1024.0 * MB;
        let x = n as f64;
        if x >= GB {
            format!("{:.2} GiB", x / GB)
        } else if x >= MB {
            format!("{:.2} MiB", x / MB)
        } else if x >= KB {
            format!("{:.2} KiB", x / KB)
        } else {
            format!("{} B", n)
        }
    }

    // Elapsed formatting is used for stable, finite durations. For ETA, we guard
    // against zero-speed or unstable estimates separately via `format_eta`.

    fn refresh_allowed(&mut self) -> (f64, f64) {
        let now = Instant::now();
        let since_last = now.duration_since(self.last_msg);
        if since_last < Duration::from_millis(100) {
            // Too soon to refresh; keep previous ETA if any
            let eta = self.last_eta_secs.unwrap_or(f64::INFINITY);
            return (0.0, eta);
        }
        self.last_msg = now;
        let elapsed = now.duration_since(self.started).as_secs_f64().max(0.001);
        let speed = (self.current as f64) / elapsed;
        let remaining = self.total.saturating_sub(self.current) as f64;

        // If speed is effectively zero, carry ETA forward and add wall time.
        const EPS: f64 = 1e-6;
        let eta = if speed <= EPS {
            let prev = self.last_eta_secs.unwrap_or(f64::INFINITY);
            if prev.is_finite() {
                prev + since_last.as_secs_f64()
            } else {
                prev
            }
        } else {
            remaining / speed
        };
        // Remember only finite ETAs to use during stalls
        if eta.is_finite() {
            self.last_eta_secs = Some(eta);
        }
        (speed, eta)
    }

    fn format_elapsed(seconds: f64) -> String {
        let total = seconds.round() as u64;
        let h = total / 3600;
        let m = (total % 3600) / 60;
        let s = total % 60;
        if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
    }

    fn format_eta(seconds: f64) -> String {
        // If ETA is not finite (e.g., divide-by-zero speed) or unreasonably large,
        // show a placeholder rather than overflowing into huge values.
        if !seconds.is_finite() {
            return "—".to_string();
        }
        // Cap ETA display to 99:59:59 to avoid silly numbers; beyond that, show placeholder.
        const CAP_SECS: f64 = 99.0 * 3600.0 + 59.0 * 60.0 + 59.0;
        if seconds > CAP_SECS {
            return "—".to_string();
        }
        Self::format_elapsed(seconds)
    }

    fn draw(&mut self) {
        if !self.enabled { return; }
        let (speed, eta) = self.refresh_allowed();
        let elapsed = Instant::now().duration_since(self.started).as_secs_f64();
        // Build bar
        let width = self.width.max(10);
        let filled = ((self.current as f64 / self.total.max(1) as f64) * width as f64).round() as usize;
        let filled = filled.min(width);
        let mut bar = String::with_capacity(width);
        for _ in 0..filled { bar.push('■'); }
        for _ in filled..width { bar.push('□'); }

        let line = format!(
            "[{}] {} [{}] ({}/{} at {}/s)",
            Self::format_elapsed(elapsed),
            bar,
            Self::format_eta(eta),
            Self::human_bytes(self.current),
            Self::human_bytes(self.total),
            Self::human_bytes(speed.max(0.0) as u64),
        );
        eprint!("\r{}\x1b[K", line);
        let _ = io::stderr().flush();
    }

    pub fn inc(&mut self, delta: u64) {
        self.current = self.current.saturating_add(delta).min(self.total);
        self.draw();
    }

    pub fn stop(mut self, text: &str) {
        if self.enabled {
            self.draw();
            eprintln!();
        } else {
            let _ = cliclack::log::info(text);
        }
    }

    pub fn error(mut self, text: &str) {
        if self.enabled {
            self.draw();
            eprintln!();
            let _ = cliclack::log::error(text);
        } else {
            let _ = cliclack::log::error(text);
        }
    }
}
