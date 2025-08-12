// Progress abstraction for STDERR-only, TTY-aware progress bars.
// Centralizes progress logic so it can be swapped or disabled easily.

use std::env;
use std::io::IsTerminal;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

// Global hook to route logs through the active progress manager so they render within
// the same cliclack/indicatif area instead of raw stderr.
static GLOBAL_PM: std::sync::Mutex<Option<ProgressManager>> = std::sync::Mutex::new(None);

/// Install a global ProgressManager used for printing log lines above bars.
pub fn set_global_progress_manager(pm: &ProgressManager) {
    if let Ok(mut g) = GLOBAL_PM.lock() {
        *g = Some(pm.clone());
    }
}

/// Remove the global ProgressManager hook.
pub fn clear_global_progress_manager() {
    if let Ok(mut g) = GLOBAL_PM.lock() {
        *g = None;
    }
}

/// Try to print a line via the global ProgressManager, returning true if handled.
pub fn log_line_via_global(line: &str) -> bool {
    if let Ok(g) = GLOBAL_PM.lock() {
        if let Some(pm) = g.as_ref() {
            pm.println_above_bars(line);
            return true;
        }
    }
    false
}

const NAME_WIDTH: usize = 28;

#[derive(Debug, Clone)]
/// Progress message sent from worker threads to the UI/main thread.
/// fraction: 0.0..1.0 progress value; stage/message are optional labels.
pub struct ProgressMessage {
    /// Fractional progress in range 0.0..=1.0.
    pub fraction: f32,
    /// Optional stage label (e.g., "load_model", "encode", "decode", "done").
    pub stage: Option<String>,
    /// Optional human-readable note.
    pub note: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Mode describing how progress should be displayed.
///
/// - None: progress is disabled or not supported.
/// - Single: one spinner for the current item only.
/// - Multi: a total progress bar plus a current-item spinner.
pub enum ProgressMode {
    /// No progress output.
    None,
    /// Single spinner for the currently processed item.
    Single,
    /// Multi-bar progress including a total counter of all inputs.
    Multi {
        /// Total number of inputs to process when using multi-bar mode.
        total_inputs: u64,
    },
}

fn stderr_is_tty() -> bool {
    // Prefer std IsTerminal when available
    std::io::stderr().is_terminal()
}

fn progress_disabled_by_env() -> bool {
    matches!(env::var("NO_PROGRESS"), Ok(ref v) if v == "1" || v.eq_ignore_ascii_case("true"))
}

#[derive(Clone)]
/// Factory that decides progress mode and produces a ProgressManager bound to stderr.
pub struct ProgressFactory {
    enabled: bool,
    mp: Option<Arc<MultiProgress>>,
}

impl ProgressFactory {
    /// Create a factory that enables progress when stderr is a TTY and neither
    /// the NO_PROGRESS env var nor the force_disable flag are set.
    pub fn new(force_disable: bool) -> Self {
        let tty = stderr_is_tty();
        let env_off = progress_disabled_by_env();
        let enabled = !(force_disable || env_off) && tty;
        if enabled {
            let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(20));
            // Render tick even if nothing changes periodically for spinner feel
            mp.set_move_cursor(true);
            Self {
                enabled,
                mp: Some(Arc::new(mp)),
            }
        } else {
            Self {
                enabled: false,
                mp: None,
            }
        }
    }

    /// Decide a suitable ProgressMode for the given number of inputs,
    /// respecting whether progress is globally enabled.
    pub fn decide_mode(&self, inputs_len: usize) -> ProgressMode {
        if !self.enabled {
            return ProgressMode::None;
        }
        if inputs_len == 0 {
            ProgressMode::None
        } else if inputs_len == 1 {
            ProgressMode::Single
        } else {
            ProgressMode::Multi {
                total_inputs: inputs_len as u64,
            }
        }
    }

    /// Construct a ProgressManager for the previously decided mode. Returns
    /// a no-op manager when progress is disabled.
    pub fn make_manager(&self, mode: ProgressMode) -> ProgressManager {
        match (self.enabled, &self.mp, mode) {
            (true, Some(mp), ProgressMode::Single) => ProgressManager::with_single(mp.clone()),
            (true, Some(mp), ProgressMode::Multi { total_inputs }) => {
                ProgressManager::with_multi(mp.clone(), total_inputs)
            }
            _ => ProgressManager::noop(),
        }
    }

    /// Preferred constructor using Config. Respects config.no_progress and TTY.
    pub fn from_config(config: &crate::Config) -> Self {
        // Prefer Config.no_progress over manual flag; still honor NO_PROGRESS env var.
        let force_disable = config.no_progress;
        Self::new(force_disable)
    }
}

#[derive(Clone)]
/// Handle for updating and finishing progress bars or a no-op when disabled.
pub struct ProgressManager {
    inner: ProgressInner,
}

#[derive(Clone)]
enum ProgressInner {
    Noop,
    Single(Arc<SingleBars>),
    Multi(Arc<MultiBars>),
}

#[derive(Debug)]
struct SingleBars {
    header: ProgressBar,
    info: ProgressBar,
    current: ProgressBar,
    // keep MultiProgress alive for suspend/println behavior
    _mp: Arc<MultiProgress>,
}

#[derive(Debug)]
struct MultiBars {
    // Header row shown above bars
    header: ProgressBar,
    // Single info/status row shown under header and above bars
    info: ProgressBar,
    // Bars: current file and total
    current: ProgressBar,
    total: ProgressBar,
    // Optional per-file bars and aggregated total percent bar (unused in new UX)
    files: Mutex<Option<Vec<ProgressBar>>>, // each length 100
    total_pct: Mutex<Option<ProgressBar>>,   // length 100
    // Metadata for aggregation
    sizes: Mutex<Option<Vec<Option<u64>>>>,
    fractions: Mutex<Option<Vec<f32>>>, // 0..=1 per file
    last_total_draw_ms: Mutex<Instant>,
    // keep MultiProgress alive
    _mp: Arc<MultiProgress>,
}

#[derive(Clone)]
/// Handle for per-item progress updates. Safe to clone and send across threads to update
/// the currently active item's progress without affecting the global total counter.
pub struct ItemHandle {
    pb: ProgressBar,
}

impl ItemHandle {
    /// Update the determinate progress for this item using a fraction in 0.0..=1.0.
    /// Internally mapped to 0..100 units.
    pub fn set_progress(&self, fraction: f32) {
        let f = if fraction.is_nan() { 0.0 } else { fraction.clamp(0.0, 1.0) };
        let pos = (f * 100.0).round() as u64;
        if self.pb.length().unwrap_or(0) == 0 {
            self.pb.set_length(100);
        }
        if self.pb.position() != pos {
            self.pb.set_position(pos);
        }
    }
    /// Set a human-readable message for this item (e.g., current stage name).
    pub fn set_message(&self, message: &str) {
        self.pb.set_message(message.to_string());
    }
    /// Finish this item by prefixing "done " to the currently displayed message.
    /// The provided message parameter is ignored to preserve stable width and avoid flicker.
    pub fn finish_with(&self, _message: &str) {
        if !self.pb.is_finished() {
            self.pb.finish_with_message(_message.to_string());
        }
    }
}

impl ProgressManager {
    /// Test helper: create a Multi-mode manager with a hidden draw target, safe for tests
    /// even when not attached to a TTY.
    pub fn new_for_tests_multi_hidden(total: usize) -> Self {
        let mp = Arc::new(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()));
        Self::with_multi(mp, total as u64)
    }

    /// Test helper: create a Single-mode manager with a hidden draw target, safe for tests
    /// even when not attached to a TTY.
    pub fn new_for_tests_single_hidden() -> Self {
        let mp = Arc::new(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()));
        Self::with_single(mp)
    }

    /// Backwards-compatible constructor used by older tests: same as new_for_tests_multi_hidden.
    pub fn test_new_multi(total: usize) -> Self {
        Self::new_for_tests_multi_hidden(total)
    }

    /// Test helper: return (completed, total) for the global bar if present.
    pub fn total_state_for_tests(&self) -> Option<(u64, u64)> {
        match &self.inner {
            ProgressInner::Multi(m) => Some((m.total.position(), m.total.length().unwrap_or(0))),
            _ => None,
        }
    }

    /// Test helper: return the number of visible bars managed initially.
    /// Single mode: 3 (header, info, current). Multi mode: 4 (header, info, current, total).
    pub fn testing_bar_count(&self) -> usize {
        match &self.inner {
            ProgressInner::Noop => 0,
            ProgressInner::Single(_) => 3,
            ProgressInner::Multi(m) => {
                // Base bars always present
                let mut count = 4;
                // If per-file bars were initialized, include them as well
                if let Ok(files) = m.files.lock() { if let Some(v) = &*files { count += v.len(); } }
                if let Ok(t) = m.total_pct.lock() { if t.is_some() { count += 1; } }
                count
            }
        }
    }

    /// Test helper: get state of the current item bar (position, length, finished, message).
    pub fn current_state_for_tests(&self) -> Option<(u64, u64, bool, String)> {
        match &self.inner {
            ProgressInner::Single(s) => Some((
                s.current.position(),
                s.current.length().unwrap_or(0),
                s.current.is_finished(),
                s.current.message().to_string(),
            )),
            ProgressInner::Multi(m) => Some((
                m.current.position(),
                m.current.length().unwrap_or(0),
                m.current.is_finished(),
                m.current.message().to_string(),
            )),
            ProgressInner::Noop => None,
        }
    }

    fn noop() -> Self {
        Self {
            inner: ProgressInner::Noop,
        }
    }

    fn with_single(mp: Arc<MultiProgress>) -> Self {
        // Order: header, info row, then current file bar
        let header = mp.add(ProgressBar::new(0));
        header.set_style(info_style());
        let info = mp.add(ProgressBar::new(0));
        info.set_style(info_style());
        let current = mp.add(ProgressBar::new(100));
        current.set_style(current_style());
        Self {
            inner: ProgressInner::Single(Arc::new(SingleBars { header, info, current, _mp: mp })),
        }
    }

    fn with_multi(mp: Arc<MultiProgress>, total_inputs: u64) -> Self {
        // Order: header, info row, then current file bar, then total bar at the bottom
        let header = mp.add(ProgressBar::new(0));
        header.set_style(info_style());
        let info = mp.add(ProgressBar::new(0));
        info.set_style(info_style());
        let current = mp.add(ProgressBar::new(100));
        current.set_style(current_style());
        let total = mp.add(ProgressBar::new(total_inputs));
        total.set_style(total_style());
        Self {
            inner: ProgressInner::Multi(Arc::new(MultiBars {
                header,
                info,
                current,
                total,
                files: Mutex::new(None),
                total_pct: Mutex::new(None),
                sizes: Mutex::new(None),
                fractions: Mutex::new(None),
                last_total_draw_ms: Mutex::new(Instant::now()),
                _mp: mp,
            })),
        }
    }

    /// Set the total number of items for the global progress (multi mode).
    pub fn set_total(&self, n: usize) {
        match &self.inner {
            ProgressInner::Multi(m) => {
                m.total.set_length(n as u64);
            }
            _ => {}
        }
    }

    /// Mark exactly one completed item (clamped to not exceed total).
    pub fn inc_completed(&self) {
        match &self.inner {
            ProgressInner::Multi(m) => {
                let len = m.total.length().unwrap_or(0);
                let pos = m.total.position();
                if pos < len {
                    m.total.inc(1);
                }
            }
            _ => {}
        }
    }

    /// Start a new item handle with an optional label.
    pub fn start_item(&self, label: &str) -> ItemHandle {
        match &self.inner {
            ProgressInner::Noop => ItemHandle { pb: ProgressBar::hidden() },
            ProgressInner::Single(s) => {
                s.current.set_message(label.to_string());
                ItemHandle { pb: s.current.clone() }
            }
            ProgressInner::Multi(m) => {
                m.current.set_message(label.to_string());
                ItemHandle { pb: m.current.clone() }
            }
        }
    }

    /// Pause progress rendering to allow a clean prompt line to be printed.
    pub fn pause_for_prompt(&self) {
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(s) => {
                let _ = s._mp.suspend(|| {});
            }
            ProgressInner::Multi(m) => {
                let _ = m._mp.suspend(|| {});
            }
        }
    }

    /// Print a line above the bars safely (TTY-aware). Falls back to eprintln! when disabled.
    pub fn println_above_bars(&self, line: &str) {
        // Try to interpret certain INFO lines as a stable title + dynamic message.
        // Examples to match:
        //  - "INFO: Fetching online data: listing models from ggerganov/whisper.cpp..."
        //     -> header = "INFO: Fetching online data"; info = "listing models from ..."
        //  - "INFO: Downloading tiny.en-q5_1 (252 MiB | https://...)..."
        //     -> header = "INFO: Downloading"; info = rest
        //  - "INFO: Total 1/3" (defensive): header = "INFO: Total"; info = rest
        let parsed: Option<(String, String)> = {
            let s = line.trim();
            if let Some(rest) = s.strip_prefix("INFO: ") {
                // Case A: explicit title followed by colon
                if let Some((title, body)) = rest.split_once(':') {
                    let title_clean = format!("INFO: {}", title.trim());
                    let body_clean = body.trim().to_string();
                    Some((title_clean, body_clean))
                } else if let Some(rest2) = rest.strip_prefix("Downloading ") {
                    Some(("INFO: Downloading".to_string(), rest2.trim().to_string()))
                } else if let Some(rest2) = rest.strip_prefix("Total") {
                    Some(("INFO: Total".to_string(), rest2.trim().to_string()))
                } else {
                    // Fallback: use first word as title, remainder as body
                    let mut it = rest.splitn(2, ' ');
                    let first = it.next().unwrap_or("").trim();
                    let remainder = it.next().unwrap_or("").trim();
                    if !first.is_empty() {
                        Some((format!("INFO: {}", first), remainder.to_string()))
                    } else {
                        None
                    }
                }
            } else {
                None
            }
        };

        match &self.inner {
            ProgressInner::Noop => eprintln!("{}", line),
            ProgressInner::Single(s) => {
                if let Some((title, body)) = parsed.as_ref() {
                    s.header.set_message(title.clone());
                    s.info.set_message(body.clone());
                } else {
                    let _ = s._mp.println(line);
                }
            }
            ProgressInner::Multi(m) => {
                if let Some((title, body)) = parsed.as_ref() {
                    m.header.set_message(title.clone());
                    m.info.set_message(body.clone());
                } else {
                    let _ = m._mp.println(line);
                }
            }
        }
    }

    /// Resume progress after a prompt (currently a no-op; redraw continues automatically).
    pub fn resume_after_prompt(&self) {}

    /// Set the message for the current-item spinner.
    pub fn set_current_message(&self, msg: &str) {
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(s) => s.current.set_message(msg.to_string()),
            ProgressInner::Multi(m) => m.current.set_message(msg.to_string()),
        }
    }

    /// Set an explicit length for the current-item spinner (useful when it becomes a determinate bar).
    pub fn set_current_length(&self, len: u64) {
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(s) => s.current.set_length(len),
            ProgressInner::Multi(m) => m.current.set_length(len),
        }
    }

    /// Increment the current-item spinner by the given delta.
    pub fn inc_current(&self, delta: u64) {
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(s) => s.current.inc(delta),
            ProgressInner::Multi(m) => m.current.inc(delta),
        }
    }

    /// Finish the current-item spinner by prefixing "done " to its current message.
    pub fn finish_current_with(&self, _msg: &str) {
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(s) => {
                let orig = s.current.message().to_string();
                s.current.finish_with_message(format!("done {}", orig));
            }
            ProgressInner::Multi(m) => {
                let orig = m.current.message().to_string();
                m.current.finish_with_message(format!("done {}", orig));
            }
        }
    }

    /// Increment the total progress bar by the given delta (multi-bar mode only).
    pub fn inc_total(&self, delta: u64) {
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(_) => {}
            ProgressInner::Multi(m) => m.total.inc(delta),
        }
    }

    /// Finish progress bars. Keep total bar visible with a final message and prefix "done " for items.
    pub fn finish_all(&self) {
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(s) => {
                if !s.current.is_finished() {
                    let orig = s.current.message().to_string();
                    s.current.finish_with_message(format!("done {}", orig));
                }
            }
            ProgressInner::Multi(m) => {
                // If per-file bars are active, finish each with stable "done <msg>"
                let mut had_files = false;
                if let Ok(g) = m.files.lock() {
                    if let Some(files) = g.as_ref() {
                        had_files = true;
                        for pb in files.iter() {
                            if !pb.is_finished() {
                                let orig = pb.message().to_string();
                                pb.finish_with_message(format!("done {}", orig));
                            }
                        }
                    }
                }
                // Finish the aggregated total percent bar or the legacy total
                if let Ok(gt) = m.total_pct.lock() {
                    if let Some(tpb) = gt.as_ref() {
                        if !tpb.is_finished() {
                            tpb.finish_with_message("100% total".to_string());
                        }
                    }
                }
                if !had_files {
                    // Legacy total/current bars: keep total visible too
                    let len = m.total.length().unwrap_or(0);
                    if !m.current.is_finished() {
                        m.current.finish_and_clear();
                    }
                    if !m.total.is_finished() {
                        m.total.finish_with_message(format!("{}/{} total", len, len));
                    }
                }
            }
        }
    }

    /// Set determinate progress of the current item using a fractional value 0.0..=1.0.
    pub fn set_progress(&self, fraction: f32) {
        let f = if fraction.is_nan() { 0.0 } else { fraction.clamp(0.0, 1.0) };
        let pos = (f * 100.0).round() as u64;
        match &self.inner {
            ProgressInner::Noop => {}
            ProgressInner::Single(s) => {
                if s.current.length().unwrap_or(0) == 0 {
                    s.current.set_length(100);
                }
                if s.current.position() != pos {
                    s.current.set_position(pos);
                }
            }
            ProgressInner::Multi(m) => {
                if m.current.length().unwrap_or(0) == 0 {
                    m.current.set_length(100);
                }
                if m.current.position() != pos {
                    m.current.set_position(pos);
                }
            }
        }
    }

    /// Set a message/label for the current item (alias for set_current_message).
    pub fn set_message(&self, message: &str) {
        self.set_current_message(message);
    }
}

fn current_style() -> ProgressStyle {
    // Per-item determinate progress: show 0..100 as pos/len with a simple bar
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {pos}/{len} {bar:40.cyan/blue} {msg}")
        .expect("invalid progress template in current_style()")
}

fn info_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg}").unwrap()
}

fn total_style() -> ProgressStyle {
    // Bottom total bar with elapsed time
    ProgressStyle::with_template("Total [{bar:28}]  {pos}/{len} [{elapsed_precise}]")
        .unwrap()
        .progress_chars("=> ")
}

#[derive(Debug, Clone, Copy)]
/// Inputs used to determine progress enablement and mode.
pub struct SelectionInput {
    /// Number of inputs to process (used to choose single vs multi mode).
    pub inputs_len: usize,
    /// Whether progress was explicitly disabled via a CLI flag.
    pub no_progress_flag: bool,
    /// Optional override for whether stderr is a TTY; if None, auto-detect.
    pub stderr_tty_override: Option<bool>,
    /// Whether progress was disabled via the NO_PROGRESS environment variable.
    pub env_no_progress: bool,
}

/// Decide whether progress is enabled and which mode to use based on SelectionInput.
pub fn select_mode(si: SelectionInput) -> (bool, ProgressMode) {
    // Compute effective enablement
    let tty = si.stderr_tty_override.unwrap_or_else(stderr_is_tty);
    let disabled = si.no_progress_flag || si.env_no_progress;
    let enabled = tty && !disabled;
    let mode = if !enabled || si.inputs_len == 0 {
        ProgressMode::None
    } else if si.inputs_len == 1 {
        ProgressMode::Single
    } else {
        ProgressMode::Multi {
            total_inputs: si.inputs_len as u64,
        }
    };
    (enabled, mode)
}

/// Optional Ctrl-C cleanup: clears progress bars and removes temporary files before exiting on SIGINT.
pub fn install_ctrlc_cleanup(pm: ProgressManager) {
    let state = Arc::new(Mutex::new(Some(pm.clone())));
    let state_clone = state.clone();
    if let Err(e) = ctrlc::set_handler(move || {
        // Clear any visible progress bars
        if let Ok(mut guard) = state_clone.lock() {
            if let Some(pm) = guard.take() {
                pm.finish_all();
            }
        }
        // Best-effort removal of the last-model cache so it doesn't persist after Ctrl-C
        let models_dir = crate::models_dir_path();
        let last_path = models_dir.join(".last_model");
        let _ = std::fs::remove_file(&last_path);
        // Also remove any unfinished model downloads ("*.part")
        if let Ok(rd) = std::fs::read_dir(&models_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if name.ends_with(".part") {
                        let _ = std::fs::remove_file(&p);
                    }
                }
            }
        }
        // Exit with 130 to reflect SIGINT
        std::process::exit(130);
    }) {
        // Warn if we failed to install the handler; without it, Ctrl-C won't trigger cleanup
        crate::wlog!("Failed to install Ctrl-C handler: {}", e);
    }
}


// --- New: Per-file progress bars API for Multi mode ---
impl ProgressManager {
    /// Initialize per-file bars and an aggregated total percent bar using indicatif::MultiProgress.
    /// Each bar has length 100 and shows a truncated filename as message.
    /// This replaces the legacy current/total display with fixed per-file lines.
    pub fn init_files<I, S>(&self, labels_and_sizes: I)
    where
        I: IntoIterator<Item = (S, Option<u64>)>,
        S: Into<String>,
    {
        if let ProgressInner::Multi(m) = &self.inner {
            // Clear legacy bars from display to avoid duplication
            m.current.finish_and_clear();
            m.total.finish_and_clear();
            let mut files: Vec<ProgressBar> = Vec::new();
            let mut sizes: Vec<Option<u64>> = Vec::new();
            let mut fractions: Vec<f32> = Vec::new();
            for (label_in, size_opt) in labels_and_sizes {
                let label: String = label_in.into();
                let pb = m._mp.add(ProgressBar::new(100));
                pb.set_style(current_style());
                let short = truncate_label(&label, NAME_WIDTH);
                pb.set_message(format!("{:<width$}", short, width = NAME_WIDTH));
                files.push(pb);
                sizes.push(size_opt);
                fractions.push(0.0);
            }
            let total_pct = m._mp.add(ProgressBar::new(100));
            total_pct
                .set_style(ProgressStyle::with_template("{bar:40.cyan/blue} {percent:>3}% total").unwrap());
            // Store
            if let Ok(mut gf) = m.files.lock() { *gf = Some(files); }
            if let Ok(mut gt) = m.total_pct.lock() { *gt = Some(total_pct); }
            if let Ok(mut gs) = m.sizes.lock() { *gs = Some(sizes); }
            if let Ok(mut gfr) = m.fractions.lock() { *gfr = Some(fractions); }
            if let Ok(mut t) = m.last_total_draw_ms.lock() { *t = Instant::now(); }
        }
    }

    /// Return whether per-file bars are active (Multi mode only)
    pub fn has_file_bars(&self) -> bool {
        match &self.inner {
            ProgressInner::Multi(m) => m.files.lock().map(|g| g.is_some()).unwrap_or(false),
            _ => false,
        }
    }

    /// Get an item handle for a specific file index (Multi mode with file bars). Falls back to legacy current.
    pub fn item_handle_at(&self, index: usize) -> ItemHandle {
        match &self.inner {
            ProgressInner::Multi(m) => {
                if let Ok(g) = m.files.lock() {
                    if let Some(vec) = g.as_ref() {
                        if let Some(pb) = vec.get(index) {
                            return ItemHandle { pb: pb.clone() };
                        }
                    }
                }
                ItemHandle { pb: m.current.clone() }
            }
            ProgressInner::Single(s) => ItemHandle { pb: s.current.clone() },
            ProgressInner::Noop => ItemHandle { pb: ProgressBar::hidden() },
        }
    }

    /// Update a specific file's progress (0.0..=1.0) and recompute the aggregated total percent.
    pub fn set_file_progress(&self, index: usize, fraction: f32) {
        let f = if fraction.is_nan() { 0.0 } else { fraction.clamp(0.0, 1.0) };
        if let ProgressInner::Multi(m) = &self.inner {
            if let Ok(gf) = m.files.lock() {
                if let Some(files) = gf.as_ref() {
                    if index < files.len() {
                        let pb = &files[index];
                        pb.set_length(100);
                        let pos = (f * 100.0).round() as u64;
                        if pb.position() != pos {
                            pb.set_position(pos);
                        }
                    }
                }
            }
            if let Ok(mut gfr) = m.fractions.lock() {
                if let Some(fracs) = gfr.as_mut() {
                    if index < fracs.len() {
                        fracs[index] = f;
                    }
                }
            }
            self.recompute_total_pct();
        }
    }

    fn recompute_total_pct(&self) {
        if let ProgressInner::Multi(m) = &self.inner {
            let has_total = m.total_pct.lock().map(|g| g.is_some()).unwrap_or(false);
            if !has_total {
                return;
            }
            let now = Instant::now();
            let do_draw = if let Ok(mut last) = m.last_total_draw_ms.lock() {
                if now.duration_since(*last).as_millis() >= 50 {
                    *last = now;
                    true
                } else {
                    false
                }
            } else {
                true
            };
            if !do_draw {
                return;
            }
            let fractions = match m.fractions.lock().ok().and_then(|g| g.clone()) {
                Some(v) => v,
                None => return,
            };
            let sizes_opt = m.sizes.lock().ok().and_then(|g| g.clone());
            let pct = if let Some(sizes) = sizes_opt.as_ref() {
                if !sizes.is_empty() && sizes.iter().all(|o| o.is_some()) {
                    let mut num: f64 = 0.0;
                    let mut den: f64 = 0.0;
                    for (f, s) in fractions.iter().zip(sizes.iter()) {
                        let sz = s.unwrap_or(0) as f64;
                        num += (*f as f64) * sz;
                        den += sz;
                    }
                    if den > 0.0 { (num / den) as f32 } else { 0.0 }
                } else {
                    // Fallback to unweighted average
                    if fractions.is_empty() { 0.0 } else { (fractions.iter().sum::<f32>()) / (fractions.len() as f32) }
                }
            } else {
                if fractions.is_empty() { 0.0 } else { (fractions.iter().sum::<f32>()) / (fractions.len() as f32) }
            };
            let pos = (pct.clamp(0.0, 1.0) * 100.0).round() as u64;
            if let Ok(gt) = m.total_pct.lock() {
                if let Some(total_pb) = gt.as_ref() {
                    total_pb.set_length(100);
                    if total_pb.position() != pos {
                        total_pb.set_position(pos);
                    }
                }
            }
        }
    }
}

fn truncate_label(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        if max <= 3 {
            return ".".repeat(max);
        }
        let keep = max - 3;
        let truncated = s.chars().take(keep).collect::<String>();
        format!("{}...", truncated)
    }
}


#[cfg(test)]
mod tests {
    use super::truncate_label;

    #[test]
    fn truncate_keeps_short_and_exact() {
        assert_eq!(truncate_label("short", 10), "short");
        assert_eq!(truncate_label("short", 5), "short");
    }

    #[test]
    fn truncate_long_adds_ellipsis() {
        assert_eq!(truncate_label("abcdefghij", 8), "abcde...");
        assert_eq!(truncate_label("filename_long.flac", 12), "filename_...");
    }

    #[test]
    fn truncate_small_max_returns_dots() {
        assert_eq!(truncate_label("anything", 3), "...");
        assert_eq!(truncate_label("anything", 2), "..");
        assert_eq!(truncate_label("anything", 1), ".");
        assert_eq!(truncate_label("anything", 0), "");
    }

    #[test]
    fn truncate_handles_unicode_by_char_boundary() {
        // Using chars().take(keep) prevents splitting code points; not grapheme-perfect but safe.
        // "é" is 2 bytes but 1 char; keep=2 should keep "Aé" then add dots
        let s = "AéBCD"; // chars: A, é, B, C, D
        assert_eq!(truncate_label(s, 5), "Aé..."); // keep 2 chars + ...
    }
}
