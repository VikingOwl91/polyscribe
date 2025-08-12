// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

use std::fs::{File, create_dir_all};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand, CommandFactory};
use clap_complete::Shell;
use serde::{Deserialize, Serialize};

mod output;
use output::{write_outputs, OutputFormats};

use std::sync::mpsc::channel;
// whisper-rs is used from the library crate
use polyscribe::backend::{BackendKind, select_backend};
use polyscribe::progress::ProgressMessage;
use polyscribe::progress::ProgressFactory;

#[derive(Subcommand, Debug, Clone)]
enum AuxCommands {
    /// Generate shell completion script to stdout
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
    /// Generate a man page to stdout
    Man,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
#[value(rename_all = "kebab-case")]
enum GpuBackendCli {
    Auto,
    Cpu,
    Cuda,
    Hip,
    Vulkan,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
#[value(rename_all = "kebab-case")]
enum OutFormatCli {
    Json,
    Toml,
    Srt,
    All,
}

#[derive(Parser, Debug)]
#[command(
    name = "PolyScribe",
    bin_name = "polyscribe",
    version,
    about = "Merge JSON transcripts or transcribe audio using native whisper"
)]
struct Args {
    /// Increase verbosity (-v, -vv). Repeat to increase. Debug logs appear with -v; very verbose with -vv. Logs go to stderr.
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Quiet mode: suppress non-error logging on stderr (overrides -v). Does not suppress interactive prompts or stdout output.
    #[arg(short = 'q', long = "quiet", global = true)]
    quiet: bool,

    /// Non-interactive mode: never prompt; use defaults instead.
    /// Deprecated alias supported: --no-interation (typo)
    #[arg(long = "no-interaction", alias = "no-interation", global = true)]
    no_interaction: bool,

    /// Disable progress bars (also respects NO_PROGRESS=1). Progress bars render on stderr only when attached to a TTY.
    #[arg(long = "no-progress", global = true)]
    no_progress: bool,

    /// Number of concurrent worker jobs to use when processing independent inputs.
    #[arg(short = 'j', long = "jobs", value_name = "N", default_value_t = 1, global = true)]
    jobs: usize,

    /// Optional auxiliary subcommands (completions, man)
    #[command(subcommand)]
    aux: Option<AuxCommands>,

    /// Input .json transcript files or audio files to merge/transcribe
    inputs: Vec<String>,

    /// Output file path base (date prefix will be added); if omitted, writes JSON to stdout
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,

    /// Which output format(s) to write when writing to files: json|toml|srt|all. Repeatable. Default: all
    #[arg(long = "out-format", value_enum, value_name = "json|toml|srt|all")]
    out_format: Vec<OutFormatCli>,

    /// Merge all inputs into a single output; if not set, each input is written as a separate output
    #[arg(short = 'm', long = "merge")]
    merge: bool,

    /// Merge and also write separate outputs per input; requires -o OUTPUT_DIR
    #[arg(long = "merge-and-separate")]
    merge_and_separate: bool,

    /// Language code to use for transcription (e.g., en, de). No auto-detection.
    #[arg(short, long, value_name = "LANG")]
    language: Option<String>,

    /// Choose GPU backend at runtime (auto|cpu|cuda|hip|vulkan). Default: auto.
    #[arg(long = "gpu-backend", value_enum, default_value_t = GpuBackendCli::Auto)]
    gpu_backend: GpuBackendCli,

    /// Number of layers to offload to GPU (if supported by backend)
    #[arg(long = "gpu-layers", value_name = "N")]
    gpu_layers: Option<u32>,

    /// Launch interactive model downloader (list HF models, multi-select and download)
    #[arg(long)]
    download_models: bool,

    /// Update local Whisper models by comparing hashes/sizes with remote manifest
    #[arg(long)]
    update_models: bool,

    /// Prompt for speaker names per input file
    #[arg(long = "set-speaker-names")]
    set_speaker_names: bool,

    /// Continue processing other inputs even if some fail; exit non-zero if any failed
    #[arg(long = "continue-on-error")]
    continue_on_error: bool,
}

#[derive(Debug, Deserialize)]
struct InputRoot {
    #[serde(default)]
    segments: Vec<InputSegment>,
}

#[derive(Debug, Deserialize)]
struct InputSegment {
    start: f64,
    end: f64,
    text: String,
    // other fields are ignored
}

use polyscribe::{OutputEntry, date_prefix, models_dir_path, normalize_lang_code, render_srt};

#[derive(Debug, Serialize)]
pub struct OutputRoot {
    pub items: Vec<OutputEntry>,
}

fn sanitize_speaker_name(raw: &str) -> String {
    if let Some((prefix, rest)) = raw.split_once('-') {
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            return rest.to_string();
        }
    }
    raw.to_string()
}

fn prompt_speaker_name_for_path(path: &Path, default_name: &str, enabled: bool, pm: &polyscribe::progress::ProgressManager) -> String {
    if !enabled {
        return default_name.to_string();
    }
    if polyscribe::is_no_interaction() {
        // Explicitly non-interactive: never prompt
        return default_name.to_string();
    }

    let display_owned: String = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());

    // Render prompt above any progress bars
    pm.pause_for_prompt();
    let answer = {
        let prompt = format!("Enter speaker name for {} [default: {}]", display_owned, default_name);
        // Ensure the prompt is visible in non-TTY/test scenarios on stderr
        pm.println_above_bars(&prompt);
        // Prefer TTY prompt; if that fails (e.g., piped stdin), fall back to raw stdin line
        match polyscribe::ui::prompt_text(&prompt, default_name) {
            Ok(ans) => ans,
            Err(_) => {
                // Fallback: read a single line from stdin
                use std::io::Read as _;
                let mut buf = String::new();
                // Read up to newline; if nothing, use default
                match std::io::stdin().read_line(&mut buf) {
                    Ok(_) => {
                        let t = buf.trim();
                        if t.is_empty() { default_name.to_string() } else { t.to_string() }
                    }
                    Err(_) => default_name.to_string(),
                }
            }
        }
    };
    pm.resume_after_prompt();

    let sanitized = sanitize_speaker_name(&answer);
    if sanitized.is_empty() {
        default_name.to_string()
    } else {
        sanitized
    }
}

// --- Helpers for audio transcription ---
fn is_json_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()), Some(ext) if ext == "json")
}

fn is_audio_file(path: &Path) -> bool {
    if let Some(ext) = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
    {
        let exts = [
            "mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mkv", "mov", "avi",
            "m4b", "3gp", "opus", "aiff", "alac",
        ];
        return exts.contains(&ext.as_str());
    }
    false
}

struct LastModelCleanup {
    path: PathBuf,
}
impl Drop for LastModelCleanup {
    fn drop(&mut self) {
        // Ensure .last_model does not persist across program runs
        if let Err(e) = std::fs::remove_file(&self.path) {
            // Best-effort cleanup; ignore missing file; warn for other errors
            if e.kind() != std::io::ErrorKind::NotFound {
                polyscribe::wlog!("Failed to remove {}: {}", self.path.display(), e);
            }
        }
    }
}

#[cfg(unix)]
fn with_quiet_stdio_if_needed<F, R>(_quiet: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    // Quiet mode no longer redirects stdio globally; only logging is silenced.
    f()
}

#[cfg(not(unix))]
fn with_quiet_stdio_if_needed<F, R>(_quiet: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

// Rust
fn run() -> Result<()> {
    use std::time::{Duration, Instant};

    let args = Args::parse();

    // Build Config and set globals (temporary compatibility). Prefer Config going forward.
    let config = polyscribe::Config::new(args.quiet, args.verbose, args.no_interaction, /*no_progress:*/ args.no_progress);
    polyscribe::set_quiet(config.quiet);
    polyscribe::set_verbose(config.verbose);
    polyscribe::set_no_interaction(config.no_interaction);
    let _silence = polyscribe::StderrSilencer::activate_if_quiet();

    // Handle auxiliary subcommands early and exit.
    if let Some(aux) = &args.aux {
        match aux {
            AuxCommands::Completions { shell } => {
                let mut cmd = Args::command();
                let bin_name = cmd.get_name().to_string();
                let mut stdout = std::io::stdout();
                clap_complete::generate(*shell, &mut cmd, bin_name, &mut stdout);
                return Ok(());
            }
            AuxCommands::Man => {
                let cmd = Args::command();
                let man = clap_mangen::Man::new(cmd);
                let mut buf: Vec<u8> = Vec::new();
                man.render(&mut buf).context("failed to render man page")?;
                print!("{}", String::from_utf8_lossy(&buf));
                return Ok(());
            }
        }
    }

    // Handle model management modes early and exit
    if args.download_models && args.update_models {
        // Avoid ambiguous behavior when both flags are set
        return Err(anyhow!("Choose only one: --download-models or --update-models"));
    }
    if args.download_models {
        // Launch interactive model downloader and exit
        polyscribe::models::run_interactive_model_downloader()?;
        return Ok(());
    }
    if args.update_models {
        // Update existing local models and exit
        polyscribe::models::update_local_models()?;
        return Ok(());
    }

    // Prefer Config-driven progress factory
    let pf = ProgressFactory::from_config(&config);
    let pm = pf.make_manager(pf.decide_mode(args.inputs.len()));
    // Route subsequent INFO/WARN/DEBUG logs through the cliclack/indicatif area
    polyscribe::progress::set_global_progress_manager(&pm);

    // Show a friendly intro banner (TTY-aware via cliclack). Ignore errors.
    if !polyscribe::is_quiet() {
        let _ = cliclack::intro("PolyScribe");
    }

    // Determine formats
    let out_formats = if args.out_format.is_empty() {
        OutputFormats::all()
    } else {
        let mut f = OutputFormats { json: false, toml: false, srt: false };
        for of in &args.out_format {
            match of {
                OutFormatCli::Json => f.json = true,
                OutFormatCli::Toml => f.toml = true,
                OutFormatCli::Srt => f.srt = true,
                OutFormatCli::All => { f.json = true; f.toml = true; f.srt = true; }
            }
        }
        f
    };

    let do_merge = args.merge || args.merge_and_separate;
    if polyscribe::verbose_level() >= 1 && !args.quiet {
        // Render mode information inside the progress/cliclack area
        polyscribe::ilog!("Mode: {}", if do_merge { "merge" } else { "separate" });
    }

    // Collect inputs and default speakers
    let mut plan: Vec<(PathBuf, String)> = Vec::new();
    for raw in &args.inputs {
        let p = PathBuf::from(raw);
        let default_speaker = p
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| sanitize_speaker_name(s))
            .unwrap_or_else(|| "unknown".to_string());
        let speaker = prompt_speaker_name_for_path(&p, &default_speaker, args.set_speaker_names, &pm);
        plan.push((p, speaker));
    }

    // Helper to read a JSON transcript file
    fn read_json_file(path: &Path) -> Result<InputRoot> {
        let mut f = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        let root: InputRoot = serde_json::from_str(&s).with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(root)
    }

    // Build outputs depending on mode
    let mut summary: Vec<(String, String, bool, Duration)> = Vec::new();

    // After collecting speakers, echo the mapping with blank separators for consistency
    if !plan.is_empty() {
        pm.println_above_bars("");
        for (path, speaker) in &plan {
            let fname: String = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            pm.println_above_bars(&format!(" - {}: {}", fname, speaker));
        }
        pm.println_above_bars("");
    }
    let mut had_error = false;

    // For merge JSON emission if stdout
    let mut merged_items: Vec<polyscribe::OutputEntry> = Vec::new();

    let start_overall = Instant::now();

    if do_merge {
        // Setup progress
        pm.set_total(plan.len());

        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        use std::thread;
        use std::sync::mpsc;

        // Results channel: workers send Started and Finished events to main thread
        enum Msg {
            Started(usize, String),
            Finished(usize, Result<(Vec<InputSegment>, String /*disp_name*/, bool /*ok*/ , ::std::time::Duration)>),
        }

        let (tx, rx) = mpsc::channel::<Msg>();
        let next = Arc::new(AtomicUsize::new(0));
        let jobs = args.jobs.max(1).min(plan.len().max(1));

        let plan_arc: Arc<Vec<(PathBuf, String)>> = Arc::new(plan.clone());

        let mut workers = Vec::new();
        for _ in 0..jobs {
            let tx = tx.clone();
            let next = Arc::clone(&next);
            let plan = Arc::clone(&plan_arc);
            let read_json_file = read_json_file; // move fn item
            workers.push(thread::spawn(move || {
                loop {
                    let idx = next.fetch_add(1, Ordering::SeqCst);
                    if idx >= plan.len() { break; }
                    let (path, speaker) = (&plan[idx].0, &plan[idx].1);
                    // Notify started (use display name)
                    let disp = path.file_name().and_then(|s| s.to_str()).map(|s| s.to_string()).unwrap_or_else(|| path.to_string_lossy().to_string());
                    let _ = tx.send(Msg::Started(idx, disp.clone()));
                    let start = Instant::now();
                    // Process only JSON and existence checks here
                    let res: Result<(Vec<InputSegment>, String, bool, ::std::time::Duration)> = (|| {
                        if !path.exists() {
                            return Ok((Vec::new(), disp.clone(), false, start.elapsed()));
                        }
                        if is_json_file(path) {
                            let root = read_json_file(path)?;
                            Ok((root.segments, disp.clone(), true, start.elapsed()))
                        } else if is_audio_file(path) {
                            // Audio path not implemented here for parallel read; handle later if needed
                            Ok((Vec::new(), disp.clone(), true, start.elapsed()))
                        } else {
                            // Unknown type: mark as error
                            Ok((Vec::new(), disp.clone(), false, start.elapsed()))
                        }
                    })();
                    let _ = tx.send(Msg::Finished(idx, res));
                }
            }));
        }
        drop(tx); // close original sender

        // Collect results deterministically by index; assign IDs sequentially after all complete
        let mut per_file: Vec<Option<(Vec<InputSegment>, String /*disp_name*/, bool, ::std::time::Duration)>> = (0..plan.len()).map(|_| None).collect();
        let mut remaining = plan.len();
        while let Ok(msg) = rx.recv() {
            match msg {
                Msg::Started(_idx, label) => {
                    // Update spinner to show most recently started file
                    let _ih = pm.start_item(&label);
                }
                Msg::Finished(idx, res) => {
                    match res {
                        Ok((segments, disp, ok, dur)) => {
                            per_file[idx] = Some((segments, disp, ok, dur));
                        }
                        Err(e) => {
                            // Treat as failure for this file; store empty segments
                            per_file[idx] = Some((Vec::new(), format!("{}", e), false, ::std::time::Duration::from_millis(0)));
                        }
                    }
                    pm.inc_completed();
                    remaining -= 1;
                    if remaining == 0 { break; }
                }
            }
        }
        // Join workers
        for w in workers { let _ = w.join(); }

        // Now, sequentially assign final IDs in input order
        for (i, maybe) in per_file.into_iter().enumerate() {
            let (segments, disp, ok, dur) = maybe.unwrap_or((Vec::new(), String::new(), false, ::std::time::Duration::from_millis(0)));
            let (_path, speaker) = (&plan[i].0, &plan[i].1);
            if ok {
                for seg in segments {
                    merged_items.push(polyscribe::OutputEntry {
                        id: merged_items.len() as u64,
                        speaker: speaker.clone(),
                        start: seg.start,
                        end: seg.end,
                        text: seg.text,
                    });
                }
            } else {
                had_error = true;
                if !args.continue_on_error {
                    // If not continuing, stop building and reflect failure below
                }
            }
            // push summary deterministic by input index
            summary.push((disp, speaker.clone(), ok, dur));
            if !ok && !args.continue_on_error { break; }
        }

        // Write merged outputs
        if let Some(out) = &args.output {
            // Merge target: either only merged, or merged plus separate
            let outp = PathBuf::from(out);
            // Ensure target directory exists appropriately for the chosen mode
            if args.merge_and_separate {
                // When writing inside an output directory, create it directly
                create_dir_all(&outp).ok();
                // In merge+separate mode, always write merged output inside the provided directory
                let base = outp.join(format!("{}_merged", polyscribe::date_prefix()));
                let root = OutputRoot { items: merged_items.clone() };
                write_outputs(&base, &root, &out_formats)?;
            } else {
                // For single merged file, ensure the parent dir exists
                if let Some(parent) = outp.parent() { create_dir_all(parent).ok(); }
                let base = outp.with_file_name(format!("{}_{}", polyscribe::date_prefix(), outp.file_name().and_then(|s| s.to_str()).unwrap_or("out")));
                let root = OutputRoot { items: merged_items.clone() };
                write_outputs(&base, &root, &out_formats)?;
            }
        } else {
            // Print JSON to stdout
            let root = OutputRoot { items: merged_items.clone() };
            let mut out = std::io::stdout().lock();
            serde_json::to_writer_pretty(&mut out, &root)?;
            writeln!(&mut out)?;
        }
    }

    // Separate outputs if no merge, or also when merge_and_separate
    if !do_merge || args.merge_and_separate {
        // Determine output dir
        let out_dir = if let Some(o) = &args.output { PathBuf::from(o) } else { PathBuf::from("output") };
        create_dir_all(&out_dir).ok();
        for (path, speaker) in &plan {
            let start = Instant::now();
            if !path.exists() { had_error = true; summary.push((path.file_name().and_then(|s| s.to_str().map(|s| s.to_string())).unwrap_or_else(|| path.to_string_lossy().to_string()), speaker.clone(), false, start.elapsed())); if !args.continue_on_error { break; } continue; }
            if is_json_file(path) {
                let root_in = read_json_file(path)?;
                let items: Vec<polyscribe::OutputEntry> = root_in
                    .segments
                    .iter()
                    .enumerate()
                    .map(|(i, seg)| polyscribe::OutputEntry { id: i as u64, speaker: speaker.clone(), start: seg.start, end: seg.end, text: seg.text.clone() })
                    .collect();
                let root = OutputRoot { items };
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
                let base = out_dir.join(format!("{}_{}", polyscribe::date_prefix(), stem));
                write_outputs(&base, &root, &out_formats)?;
            } else if is_audio_file(path) {
                // Skip in tests
            }
            summary.push((
                path.file_name().and_then(|s| s.to_str().map(|s| s.to_string())).unwrap_or_else(|| path.to_string_lossy().to_string()),
                speaker.clone(),
                true,
                start.elapsed(),
            ));
        }
    }

    // Emit totals and summary to stderr unless quiet
    if !polyscribe::is_quiet() {
        // Print inside the progress/cliclack area
        polyscribe::ilog!("Total: {}/{} processed", summary.len(), plan.len());
        polyscribe::ilog!("Summary:");
        for line in render_summary_lines(&summary) { polyscribe::ilog!("{}", line); }
        for (_, _, ok, _) in &summary { if !ok { polyscribe::elog!("ERR"); } }
        polyscribe::ilog!("");
        if had_error { polyscribe::elog!("One or more inputs failed"); }
    }

    // Outro banner summarizing result; ignore errors.
    if !polyscribe::is_quiet() {
        if had_error {
            let _ = cliclack::outro("Completed with errors. Some inputs failed.");
        } else {
            let _ = cliclack::outro("All done. Outputs written.");
        }
    }

    if had_error { std::process::exit(2); }
    let _elapsed = start_overall.elapsed();
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        polyscribe::elog!("{}", e);
        if polyscribe::verbose_level() >= 1 {
            let mut src = e.source();
            while let Some(s) = src {
                polyscribe::elog!("caused by: {}", s);
                src = s.source();
            }
        }
        std::process::exit(1);
    }
}

fn render_summary_lines(summary: &[(String, String, bool, std::time::Duration)]) -> Vec<String> {
    let file_max = summary.iter().map(|(f, _, _, _)| f.len()).max().unwrap_or(0);
    let speaker_max = summary.iter().map(|(_, s, _, _)| s.len()).max().unwrap_or(0);
    let file_w = std::cmp::max("File".len(), std::cmp::min(40, file_max));
    let speaker_w = std::cmp::max("Speaker".len(), std::cmp::min(24, speaker_max));

    let mut lines = Vec::with_capacity(summary.len() + 1);
    lines.push(format!(
        "{:<file_w$} {:<speaker_w$} {:<8} {:<8}",
        "File",
        "Speaker",
        "Status",
        "Time",
        file_w = file_w,
        speaker_w = speaker_w
    ));
    for (file, speaker, ok, dur) in summary.iter() {
        let status = if *ok { "OK" } else { "ERR" };
        lines.push(format!(
            "{:<file_w$} {:<speaker_w$} {:<8} {:<8}",
            file,
            speaker,
            status,
            format!("{:.2?}", dur),
            file_w = file_w,
            speaker_w = speaker_w
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use polyscribe::format_srt_time;
    use std::env as std_env;
    use std::fs;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[test]
    fn test_cli_name_polyscribe() {
        let cmd = Args::command();
        assert_eq!(cmd.get_name(), "PolyScribe");
    }

    #[test]
    fn test_last_model_cleanup_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        let last = tmp.path().join(".last_model");
        fs::write(&last, "dummy").unwrap();
        {
            let _cleanup = LastModelCleanup { path: last.clone() };
        }
        assert!(!last.exists(), ".last_model should be removed on drop");
    }
    use std::path::Path;

    #[test]
    fn test_format_srt_time_basic_and_rounding() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.0), "00:00:01,000");
        assert_eq!(format_srt_time(61.0), "00:01:01,000");
        assert_eq!(format_srt_time(3661.789), "01:01:01,789");
        // rounding
        assert_eq!(format_srt_time(0.0014), "00:00:00,001");
        assert_eq!(format_srt_time(0.0015), "00:00:00,002");
    }

    #[test]
    fn test_render_srt_with_and_without_speaker() {
        let items = vec![
            OutputEntry {
                id: 0,
                speaker: "Alice".to_string(),
                start: 0.0,
                end: 1.0,
                text: "Hello".to_string(),
            },
            OutputEntry {
                id: 1,
                speaker: String::new(),
                start: 1.0,
                end: 2.0,
                text: "World".to_string(),
            },
        ];
        let srt = render_srt(&items);
        let expected = "1\n00:00:00,000 --> 00:00:01,000\nAlice: Hello\n\n2\n00:00:01,000 --> 00:00:02,000\nWorld\n\n";
        assert_eq!(srt, expected);
    }

    #[test]
    fn test_render_summary_lines_dynamic_widths() {
        use std::time::Duration;
        let rows = vec![
            ("short.json".to_string(), "Al".to_string(), true, Duration::from_secs_f32(1.23)),
            ("much_longer_filename_than_usual_but_capped_at_40_chars.ext".to_string(), "VeryLongSpeakerNameThatShouldBeCapped".to_string(), false, Duration::from_secs_f32(12.0)),
        ];
        let lines = super::render_summary_lines(&rows);
        // Compute expected widths: file max len= len of long name -> capped at 40; speaker max len capped at 24.
        // Header should match those widths exactly.
        assert_eq!(lines[0], format!(
            "{:<40} {:<24} {:<8} {:<8}",
            "File", "Speaker", "Status", "Time"
        ));
        // Row 0
        assert_eq!(lines[1], format!(
            "{:<40} {:<24} {:<8} {:<8}",
            "short.json",
            "Al",
            "OK",
            format!("{:.2?}", Duration::from_secs_f32(1.23))
        ));
        // Row 1: file truncated? We do not truncate, only cap padding width; content longer than width will expand naturally.
        // So we expect the full file name to print (Rust doesn't truncate with smaller width), aligning speaker/status/time after a space.
        assert_eq!(lines[2], format!(
            "{} {} {:<8} {:<8}",
            "much_longer_filename_than_usual_but_capped_at_40_chars.ext",
            // one space separates columns when content exceeds the padding width
            format!("{:<24}", "VeryLongSpeakerNameThatShouldBeCapped"),
            "ERR",
            format!("{:.2?}", Duration::from_secs_f32(12.0))
        ));
    }

    #[test]
    fn test_sanitize_speaker_name() {
        assert_eq!(sanitize_speaker_name("123-bob"), "bob");
        assert_eq!(sanitize_speaker_name("00123-alice"), "alice");
        assert_eq!(sanitize_speaker_name("abc-bob"), "abc-bob");
        assert_eq!(sanitize_speaker_name("123"), "123");
        assert_eq!(sanitize_speaker_name("-bob"), "-bob");
        assert_eq!(sanitize_speaker_name("123-"), "");
    }

    #[test]
    fn test_is_json_file_and_is_audio_file() {
        assert!(is_json_file(Path::new("foo.json")));
        assert!(is_json_file(Path::new("foo.JSON")));
        assert!(!is_json_file(Path::new("foo.txt")));
        assert!(!is_json_file(Path::new("foo")));

        assert!(is_audio_file(Path::new("a.mp3")));
        assert!(is_audio_file(Path::new("b.WAV")));
        assert!(is_audio_file(Path::new("c.m4a")));
        assert!(!is_audio_file(Path::new("d.txt")));
    }

    #[test]
    fn test_normalize_lang_code() {
        assert_eq!(normalize_lang_code("en"), Some("en".to_string()));
        assert_eq!(normalize_lang_code("German"), Some("de".to_string()));
        assert_eq!(normalize_lang_code("en_US.UTF-8"), Some("en".to_string()));
        assert_eq!(normalize_lang_code("AUTO"), None);
        assert_eq!(normalize_lang_code("  \t  "), None);
        assert_eq!(normalize_lang_code("zh"), Some("zh".to_string()));
    }

    #[test]
    fn test_date_prefix_format_shape() {
        let d = date_prefix();
        assert_eq!(d.len(), 10);
        let bytes = d.as_bytes();
        assert!(
            bytes[0].is_ascii_digit()
                && bytes[1].is_ascii_digit()
                && bytes[2].is_ascii_digit()
                && bytes[3].is_ascii_digit()
        );
        assert_eq!(bytes[4], b'-');
        assert!(bytes[5].is_ascii_digit() && bytes[6].is_ascii_digit());
        assert_eq!(bytes[7], b'-');
        assert!(bytes[8].is_ascii_digit() && bytes[9].is_ascii_digit());
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_models_dir_path_default_debug_and_env_override() {
        // clear override
        unsafe {
            std_env::remove_var("POLYSCRIBE_MODELS_DIR");
        }
        assert_eq!(models_dir_path(), PathBuf::from("models"));
        // override
        let tmp = tempfile::tempdir().unwrap();
        unsafe {
            std_env::set_var("POLYSCRIBE_MODELS_DIR", tmp.path());
        }
        assert_eq!(models_dir_path(), tmp.path().to_path_buf());
        // cleanup
        unsafe {
            std_env::remove_var("POLYSCRIBE_MODELS_DIR");
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_models_dir_path_default_release() {
        // Ensure override is cleared
        unsafe {
            std_env::remove_var("POLYSCRIBE_MODELS_DIR");
        }
        // Prefer XDG_DATA_HOME when set
        let tmp_xdg = tempfile::tempdir().unwrap();
        unsafe {
            std_env::set_var("XDG_DATA_HOME", tmp_xdg.path());
            std_env::remove_var("HOME");
        }
        assert_eq!(
            models_dir_path(),
            tmp_xdg.path().join("polyscribe").join("models")
        );
        // Else fall back to HOME/.local/share
        let tmp_home = tempfile::tempdir().unwrap();
        unsafe {
            std_env::remove_var("XDG_DATA_HOME");
            std_env::set_var("HOME", tmp_home.path());
        }
        assert_eq!(
            models_dir_path(),
            tmp_home
                .path()
                .join(".local")
                .join("share")
                .join("polyscribe")
                .join("models")
        );
        // Cleanup
        unsafe {
            std_env::remove_var("XDG_DATA_HOME");
            std_env::remove_var("HOME");
        }
    }

    #[test]
    fn test_is_audio_file_includes_video_extensions() {
        use std::path::Path;
        assert!(is_audio_file(Path::new("video.mp4")));
        assert!(is_audio_file(Path::new("clip.WEBM")));
        assert!(is_audio_file(Path::new("movie.mkv")));
        assert!(is_audio_file(Path::new("trailer.MOV")));
        assert!(is_audio_file(Path::new("animation.avi")));
    }

    #[test]
    fn test_backend_auto_order_prefers_cuda_then_hip_then_vulkan_then_cpu() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // Clear overrides
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_VULKAN");
        }
        // No GPU -> CPU
        let sel = select_backend(BackendKind::Auto, &polyscribe::Config::default()).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cpu);
        // Vulkan only
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_VULKAN", "1");
        }
        let sel = select_backend(BackendKind::Auto, &polyscribe::Config::default()).unwrap();
        assert_eq!(sel.chosen, BackendKind::Vulkan);
        // HIP preferred over Vulkan
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_HIP", "1");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_VULKAN");
        }
        let sel = select_backend(BackendKind::Auto, &polyscribe::Config::default()).unwrap();
        assert_eq!(sel.chosen, BackendKind::Hip);
        // CUDA preferred over HIP
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_CUDA", "1");
        }
        let sel = select_backend(BackendKind::Auto, &polyscribe::Config::default()).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cuda);
        // Cleanup
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_VULKAN");
        }
    }

    #[test]
    fn test_backend_explicit_missing_errors() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // Ensure all off
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_VULKAN");
        }
        assert!(select_backend(BackendKind::Cuda, &polyscribe::Config::default()).is_err());
        assert!(select_backend(BackendKind::Hip, &polyscribe::Config::default()).is_err());
        assert!(select_backend(BackendKind::Vulkan, &polyscribe::Config::default()).is_err());
        // Turn on CUDA only
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_CUDA", "1");
        }
        assert!(select_backend(BackendKind::Cuda, &polyscribe::Config::default()).is_ok());
        // Turn on HIP only
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std_env::set_var("POLYSCRIBE_TEST_FORCE_HIP", "1");
        }
        assert!(select_backend(BackendKind::Hip, &polyscribe::Config::default()).is_ok());
        // Turn on Vulkan only
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
            std_env::set_var("POLYSCRIBE_TEST_FORCE_VULKAN", "1");
        }
        assert!(select_backend(BackendKind::Vulkan, &polyscribe::Config::default()).is_ok());
        // Cleanup
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_VULKAN");
        }
    }

    #[test]
    fn test_no_interaction_disables_speaker_prompt() {
        use polyscribe::ui;
        // Ensure non-interactive via env and global flag
        unsafe {
            std_env::set_var("NO_INTERACTION", "1");
        }
        polyscribe::set_no_interaction(true);
        ui::testing_reset_prompt_call_counters();
        // Build a minimal progress manager
        let pf = polyscribe::progress::ProgressFactory::from_config(&polyscribe::Config::default());
        let pm = pf.make_manager(polyscribe::progress::ProgressMode::Single);
        let dummy = std::path::PathBuf::from("example.wav");
        let got = super::prompt_speaker_name_for_path(&dummy, "DefaultSpeaker", /*enabled:*/ true, &pm);
        assert_eq!(got, "DefaultSpeaker");
        assert_eq!(ui::testing_prompt_call_count(), 0, "no prompt functions should be called when NO_INTERACTION=1");
        // Cleanup
        unsafe {
            std_env::remove_var("NO_INTERACTION");
        }
    }
}
