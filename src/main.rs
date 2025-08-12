// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

use std::fs::{File, create_dir_all};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
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
    #[arg(long = "no-interaction", global = true)]
    no_interaction: bool,

    /// Disable progress bars (also respects NO_PROGRESS=1). Progress bars render on stderr only when attached to a TTY.
    #[arg(long = "no-progress", global = true)]
    no_progress: bool,

    /// Optional auxiliary subcommands (completions, man)
    #[command(subcommand)]
    aux: Option<AuxCommands>,

    /// Input .json transcript files or audio files to merge/transcribe
    inputs: Vec<String>,

    /// Output file path base (date prefix will be added); if omitted, writes JSON to stdout
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,

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
        match polyscribe::ui::prompt_text(&prompt, default_name) {
            Ok(ans) => ans,
            Err(_) => default_name.to_string(),
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

fn run() -> Result<()> {
    use polyscribe::progress::ProgressFactory;
    // Parse CLI
    let args = Args::parse();

    // Initialize runtime flags
    polyscribe::set_verbose(args.verbose);
    polyscribe::set_quiet(args.quiet);
    polyscribe::set_no_interaction(args.no_interaction);

    // Handle auxiliary subcommands that write to stdout and exit early
    if let Some(aux) = &args.aux {
        use clap::CommandFactory;
        match aux {
            AuxCommands::Completions { shell } => {
                let mut cmd = Args::command();
                let bin_name = cmd.get_name().to_string();
                clap_complete::generate(*shell, &mut cmd, bin_name, &mut io::stdout());
                return Ok(());
            }
            AuxCommands::Man => {
                let cmd = Args::command();
                let man = clap_mangen::Man::new(cmd);
                let mut out = Vec::new();
                man.render(&mut out)?;
                io::stdout().write_all(&out)?;
                return Ok(());
            }
        }
    }

    // Defer cleanup of .last_model until program exit
    let models_dir_buf = models_dir_path();
    let last_model_path = models_dir_buf.join(".last_model");
    // Ensure cleanup at end of program, regardless of exit path
    let _last_model_cleanup = LastModelCleanup {
        path: last_model_path.clone(),
    };

    // Also ensure cleanup on panic: install a panic hook that removes .last_model, then chains
    {
        let last_for_panic = last_model_path.clone();
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = std::fs::remove_file(&last_for_panic);
            // chain to default/previous hook for normal panic reporting
            prev_hook(info);
        }));
    }

    // Select backend
    let requested = match args.gpu_backend {
        GpuBackendCli::Auto => BackendKind::Auto,
        GpuBackendCli::Cpu => BackendKind::Cpu,
        GpuBackendCli::Cuda => BackendKind::Cuda,
        GpuBackendCli::Hip => BackendKind::Hip,
        GpuBackendCli::Vulkan => BackendKind::Vulkan,
    };
    let sel = select_backend(requested, args.verbose > 0)?;
    polyscribe::dlog!(1, "Using backend: {:?}", sel.chosen);

    // If requested, run the interactive model downloader first. If no inputs were provided, exit after downloading.
    if args.download_models {
        if let Err(e) = polyscribe::models::run_interactive_model_downloader() {
            polyscribe::elog!("Model downloader failed: {:#}", e);
        }
        if args.inputs.is_empty() {
            return Ok(());
        }
    }

    // If requested, update local models and exit unless inputs provided to continue
    if args.update_models {
        if let Err(e) = polyscribe::models::update_local_models() {
            polyscribe::elog!("Model update failed: {:#}", e);
            return Err(e);
        }
        // if only updating models and no inputs, exit
        if args.inputs.is_empty() {
            return Ok(());
        }
    }

    // Determine inputs and optional output path
    polyscribe::dlog!(1, "Parsed {} input(s)", args.inputs.len());

    // Progress will be initialized after all prompts are completed
    // Install Ctrl-C cleanup that removes .last_model and exits 130 on SIGINT
    let last_for_ctrlc = last_model_path.clone();
    ctrlc::set_handler(move || {
        let _ = std::fs::remove_file(&last_for_ctrlc);
        std::process::exit(130);
    })
    .expect("failed to set ctrlc handler");

    let mut inputs = args.inputs;
    let mut output_path = args.output;
    if output_path.is_none() && inputs.len() >= 2 {
        if let Some(last) = inputs.last().cloned() {
            if !Path::new(&last).exists() {
                inputs.pop();
                output_path = Some(last);
            }
        }
    }
    if inputs.is_empty() {
        return Err(anyhow!("No input files provided"));
    }

    // Language must be provided via CLI when transcribing audio (no detection from JSON/env)
    let lang_hint: Option<String> = if let Some(ref l) = args.language {
        normalize_lang_code(l).or_else(|| Some(l.trim().to_lowercase()))
    } else {
        None
    };
    let any_audio = inputs.iter().any(|p| is_audio_file(Path::new(p)));
    if any_audio && lang_hint.is_none() {
        return Err(anyhow!(
            "Please specify --language (e.g., --language en). Language detection was removed."
        ));
    }

    // Initialize progress manager BEFORE any interactive prompts so we can route
    // prompt lines via the synchronized ProgressManager APIs
    let pf = ProgressFactory::new(args.no_progress || args.quiet);
    let mode = pf.decide_mode(inputs.len());
    let progress = pf.make_manager(mode);
    progress.set_total(inputs.len());
    polyscribe::dlog!(1, "Progress mode: {:?}", mode);

    // Trigger model selection once upfront so any interactive messages appear cleanly
    if any_audio {
        progress.pause_for_prompt();
        if let Err(e) = polyscribe::find_model_file_with_printer(|s: &str| {
            progress.println_above_bars(s);
        }) {
            progress.resume_after_prompt();
            return Err(e);
        }
        // Blank line after model selection prompts
        progress.println_above_bars("");
        progress.resume_after_prompt();
    }

    // 1) Prompt all speaker names upfront (before creating per-file bars), respecting non-interactive stdin
    let mut speakers: Vec<String> = Vec::new();
    for s in &inputs {
        let path = Path::new(s);
        let default_speaker = sanitize_speaker_name(
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("speaker"),
        );
        let name = prompt_speaker_name_for_path(path, &default_speaker, args.set_speaker_names, &progress);
        speakers.push(name);
    }

    // 2) After collecting names, optionally print a compact mapping once
    // Only when interactive and not quiet
    if !args.quiet && !polyscribe::is_no_interaction() {
        progress.println_above_bars("Files to process:");
        for e in inputs.iter().zip(speakers.iter()) {
            let (input, speaker) = e;
            let p = Path::new(input);
            let display = p
                .file_name()
                .and_then(|os| os.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| p.to_string_lossy().to_string());
            progress.println_above_bars(&format!(" - {} -> {}", display, speaker));
        }
        // Blank line before progress display
        progress.println_above_bars("");
    }

    if args.merge_and_separate {
        polyscribe::dlog!(1, "Mode: merge-and-separate; output_dir={:?}", output_path);
        // Combined mode: write separate outputs per input and also a merged output set
        // Require an output directory
        let out_dir = match output_path.as_ref() {
            Some(p) => PathBuf::from(p),
            None => return Err(anyhow!("--merge-and-separate requires -o OUTPUT_DIR")),
        };
        if !out_dir.as_os_str().is_empty() {
            create_dir_all(&out_dir).with_context(|| {
                format!("Failed to create output directory: {}", out_dir.display())
            })?;
        }

        let mut merged_entries: Vec<OutputEntry> = Vec::new();

        let mut completed_count: usize = 0;
        let total_inputs = inputs.len();
        let mut summary: Vec<(String, String, bool, std::time::Duration)> = Vec::with_capacity(total_inputs);
        for (idx, input_path) in inputs.iter().enumerate() {
            let path = Path::new(input_path);
            let started_at = std::time::Instant::now();
            let display_name = path
                .file_name()
                .and_then(|os| os.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            // Single progress area: one item spinner/bar
            let item = progress.start_item(&format!("Processing: {}", path.display()));
            if matches!(mode, polyscribe::progress::ProgressMode::None) {
                polyscribe::ilog!("Processing: {} ... started", path.display());
            }
            let speaker = speakers[idx].clone();

            // Collect entries per file and extend merged
            let mut entries: Vec<OutputEntry> = Vec::new();
            if is_audio_file(path) {
                // Avoid println! while bars are active: only log when no bars, otherwise keep UI clean
                if matches!(mode, polyscribe::progress::ProgressMode::None) {
                    polyscribe::ilog!("Processing file: {} ...", path.display());
                }
                // Setup progress channel and receiver thread for this transcription
                let (tx, rx) = channel::<ProgressMessage>();
                let item_clone = item.clone();
                let recv_handle = std::thread::spawn(move || {
                    let mut last = -1.0f32;
                    while let Ok(msg) = rx.recv() {
                        if let Some(stage) = &msg.stage {
                            item_clone.set_message(stage);
                        }
                        let f = msg.fraction;
                        if (f - last).abs() >= 0.01 || f >= 0.999 {
                            item_clone.set_progress(f);
                            last = f;
                        }
                        if f >= 1.0 {
                            break;
                        }
                    }
                });
                let res = with_quiet_stdio_if_needed(args.quiet, || {
                    sel.backend.transcribe(
                        path,
                        &speaker,
                        lang_hint.as_deref(),
                        Some(tx),
                        args.gpu_layers,
                    )
                });
                let _ = recv_handle.join();
                match res {
                    Ok(items) => {
                        if matches!(mode, polyscribe::progress::ProgressMode::None) {
                            polyscribe::ilog!("done");
                        }
                        // Mark progress for this input after outputs are written (below)
                        entries.extend(items.into_iter());
                    }
                    Err(e) => {
                        if !polyscribe::is_no_interaction() && polyscribe::stdin_is_tty() {
                            polyscribe::elog!("{:#}", e);
                        }
                        return Err(e);
                    }
                }
            } else if is_json_file(path) {
                let mut buf = String::new();
                File::open(path)
                    .with_context(|| format!("Failed to open: {input_path}"))?
                    .read_to_string(&mut buf)
                    .with_context(|| format!("Failed to read: {input_path}"))?;
                let root: InputRoot = serde_json::from_str(&buf)
                    .with_context(|| format!("Invalid JSON transcript parsed from {input_path}"))?;
                for seg in root.segments {
                    entries.push(OutputEntry {
                        id: 0,
                        speaker: speaker.clone(),
                        start: seg.start,
                        end: seg.end,
                        text: seg.text,
                    });
                }
            } else {
                return Err(anyhow!(format!(
                    "Unsupported input type (expected .json or audio media): {}",
                    input_path
                )));
            }

            // Sort and reassign ids per file
            entries.sort_by(|a, b| {
                match a.start.partial_cmp(&b.start) {
                    Some(std::cmp::Ordering::Equal) | None => {}
                    Some(o) => return o,
                }
                a.end
                    .partial_cmp(&b.end)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, e) in entries.iter_mut().enumerate() {
                e.id = i as u64;
            }

            // Write separate outputs to out_dir
            let out = OutputRoot {
                items: entries.clone(),
            };
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let date = date_prefix();
            let base_name = format!("{date}_{stem}");
            let base_path = out_dir.join(&base_name);
            write_outputs(&base_path, &out, &OutputFormats::all())?;

            // Extend merged with per-file entries
            merged_entries.extend(out.items.into_iter());
            // progress: mark file complete (once per input)
            item.finish_with("done");
            progress.inc_completed();
            completed_count += 1;
            if matches!(mode, polyscribe::progress::ProgressMode::None) {
                polyscribe::ilog!("Total: {}/{} processed", completed_count, total_inputs);
            }
            // record summary row
            summary.push((display_name, speaker.clone(), true, started_at.elapsed()));
        }

        // Now write merged output set into out_dir
        merged_entries.sort_by(|a, b| {
            match a.start.partial_cmp(&b.start) {
                Some(std::cmp::Ordering::Equal) | None => {}
                Some(o) => return o,
            }
            a.end
                .partial_cmp(&b.end)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, e) in merged_entries.iter_mut().enumerate() {
            e.id = i as u64;
        }
        let merged_out = OutputRoot {
            items: merged_entries,
        };

        let date = date_prefix();
        let merged_base = format!("{date}_merged");
        let base_path = out_dir.join(&merged_base);
        write_outputs(&base_path, &merged_out, &OutputFormats::all())?;

        // Final concise summary table to stderr (below progress bars)
        if !args.quiet && !summary.is_empty() {
            progress.println_above_bars("Summary:");
            progress.println_above_bars(&format!("{:<22} {:<18} {:<8} {:<8}", "File", "Speaker", "Status", "Time"));
            for (file, speaker, ok, dur) in summary {
                let status = if ok { "OK" } else { "ERR" };
                progress.println_above_bars(&format!(
                    "{:<22} {:<18} {:<8} {:<8}",
                    file,
                    speaker,
                    status,
                    format!("{:.2?}", dur)
                ));
            }
            // One blank line before finishing bars
            progress.println_above_bars("");
        }
    } else if args.merge {
        polyscribe::dlog!(1, "Mode: merge; output_base={:?}", output_path);
        // MERGED MODE (previous default)
        let mut entries: Vec<OutputEntry> = Vec::new();
        let mut completed_count: usize = 0;
        let total_inputs = inputs.len();
        let mut summary: Vec<(String, String, bool, std::time::Duration)> = Vec::with_capacity(total_inputs);
        for (idx, input_path) in inputs.iter().enumerate() {
            let path = Path::new(input_path);
            let started_at = std::time::Instant::now();
            let display_name = path
                .file_name()
                .and_then(|os| os.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            let item = if progress.has_file_bars() { progress.item_handle_at(idx) } else { progress.start_item(&format!("Processing: {}", path.display())) };
            let speaker = speakers[idx].clone();

            let mut buf = String::new();
            if is_audio_file(path) {
                // Avoid println! while bars are active
                if matches!(mode, polyscribe::progress::ProgressMode::None) {
                    polyscribe::ilog!("Processing file: {} ...", path.display());
                }
                let (tx, rx) = channel::<ProgressMessage>();
                let item_clone = item.clone();
                let allow_stage_msgs = !progress.has_file_bars();
                let recv_handle = std::thread::spawn(move || {
                    let mut last = -1.0f32;
                    while let Ok(msg) = rx.recv() {
                        if allow_stage_msgs {
                            if let Some(stage) = &msg.stage {
                                item_clone.set_message(stage);
                            }
                        }
                        let f = msg.fraction;
                        if (f - last).abs() >= 0.01 || f >= 0.999 {
                            item_clone.set_progress(f);
                            last = f;
                        }
                        if f >= 1.0 {
                            break;
                        }
                    }
                });
                let res = with_quiet_stdio_if_needed(args.quiet, || {
                    sel.backend.transcribe(
                        path,
                        &speaker,
                        lang_hint.as_deref(),
                        Some(tx),
                        args.gpu_layers,
                    )
                });
                let _ = recv_handle.join();
                match res {
                    Ok(items) => {
                        if matches!(mode, polyscribe::progress::ProgressMode::None) {
                            polyscribe::ilog!("done");
                        }
                        item.finish_with("done");
                        progress.inc_completed();
                        completed_count += 1;
                        if matches!(mode, polyscribe::progress::ProgressMode::None) {
                            polyscribe::ilog!("Total: {}/{} processed", completed_count, total_inputs);
                        }
                        for e in items {
                            entries.push(e);
                        }
                        // record summary row
                        summary.push((display_name, speaker.clone(), true, started_at.elapsed()));
                        continue;
                    }
                    Err(e) => {
                        if !polyscribe::is_no_interaction() && polyscribe::stdin_is_tty() {
                            polyscribe::elog!("{:#}", e);
                        }
                        return Err(e);
                    }
                }
            } else if is_json_file(path) {
                File::open(path)
                    .with_context(|| format!("Failed to open: {input_path}"))?
                    .read_to_string(&mut buf)
                    .with_context(|| format!("Failed to read: {input_path}"))?;
                // progress: mark file complete (JSON parsed)
                item.finish_with("done");
                progress.inc_completed();
                completed_count += 1;
                if matches!(mode, polyscribe::progress::ProgressMode::None) {
                    polyscribe::ilog!("Total: {}/{} processed", completed_count, total_inputs);
                }
                // record summary row
                summary.push((display_name, speaker.clone(), true, started_at.elapsed()));
            } else {
                return Err(anyhow!(format!(
                    "Unsupported input type (expected .json or audio media): {}",
                    input_path
                )));
            }

            let root: InputRoot = serde_json::from_str(&buf)
                .with_context(|| format!("Invalid JSON transcript parsed from {input_path}"))?;

            for seg in root.segments {
                entries.push(OutputEntry {
                    id: 0,
                    speaker: speaker.clone(),
                    start: seg.start,
                    end: seg.end,
                    text: seg.text,
                });
            }
        }

        // Sort globally by (start, end)
        entries.sort_by(|a, b| {
            match a.start.partial_cmp(&b.start) {
                Some(std::cmp::Ordering::Equal) | None => {}
                Some(o) => return o,
            }
            a.end
                .partial_cmp(&b.end)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, e) in entries.iter_mut().enumerate() {
            e.id = i as u64;
        }
        let out = OutputRoot { items: entries };

        if let Some(path) = output_path {
            let base_path = Path::new(&path);
            let parent_opt = base_path.parent();
            if let Some(parent) = parent_opt {
                if !parent.as_os_str().is_empty() {
                    create_dir_all(parent).with_context(|| {
                        format!(
                            "Failed to create parent directory for output: {}",
                            parent.display()
                        )
                    })?;
                }
            }
            let stem = base_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let date = date_prefix();
            let base_name = format!("{date}_{stem}");
            let dir = parent_opt.unwrap_or(Path::new(""));
            let base_path = dir.join(&base_name);
            write_outputs(&base_path, &out, &OutputFormats::all())?;
        } else {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            serde_json::to_writer_pretty(&mut handle, &out)?;
            writeln!(&mut handle)?;
        }

        // Final concise summary table to stderr (below progress bars)
        if !args.quiet && !summary.is_empty() {
            progress.println_above_bars("Summary:");
            progress.println_above_bars(&format!("{:<22} {:<18} {:<8} {:<8}", "File", "Speaker", "Status", "Time"));
            for (file, speaker, ok, dur) in summary {
                let status = if ok { "OK" } else { "ERR" };
                progress.println_above_bars(&format!(
                    "{:<22} {:<18} {:<8} {:<8}",
                    file,
                    speaker,
                    status,
                    format!("{:.2?}", dur)
                ));
            }
            // One blank line before finishing bars
            progress.println_above_bars("");
        }
    } else {
        polyscribe::dlog!(1, "Mode: separate; output_dir={:?}", output_path);
        // SEPARATE MODE (default now)
        // If writing to stdout with multiple inputs, not supported
        if output_path.is_none() && inputs.len() > 1 {
            return Err(anyhow!(
                "Multiple inputs without --merge require -o OUTPUT_DIR to write separate files"
            ));
        }

        // If output_path is provided, treat it as a directory. Create it.
        let out_dir: Option<PathBuf> = output_path.as_ref().map(PathBuf::from);
        if let Some(dir) = &out_dir {
            if !dir.as_os_str().is_empty() {
                create_dir_all(dir).with_context(|| {
                    format!("Failed to create output directory: {}", dir.display())
                })?;
            }
        }

        let mut completed_count: usize = 0;
        let total_inputs = inputs.len();
        let mut summary: Vec<(String, String, bool, std::time::Duration)> = Vec::with_capacity(total_inputs);
        for (idx, input_path) in inputs.iter().enumerate() {
            let path = Path::new(input_path);
            let started_at = std::time::Instant::now();
            let display_name = path
                .file_name()
                .and_then(|os| os.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            let item = progress.start_item(&format!("Processing: {}", path.display()));
            let speaker = speakers[idx].clone();

            // Collect entries per file
            let mut entries: Vec<OutputEntry> = Vec::new();
            if is_audio_file(path) {
                // Avoid println! while bars are active
                if matches!(mode, polyscribe::progress::ProgressMode::None) {
                    polyscribe::ilog!("Processing file: {} ...", path.display());
                }
                let (tx, rx) = channel::<ProgressMessage>();
                let item_clone = item.clone();
                let allow_stage_msgs = !progress.has_file_bars();
                let recv_handle = std::thread::spawn(move || {
                    let mut last = -1.0f32;
                    while let Ok(msg) = rx.recv() {
                        if allow_stage_msgs {
                            if let Some(stage) = &msg.stage {
                                item_clone.set_message(stage);
                            }
                        }
                        let f = msg.fraction;
                        if (f - last).abs() >= 0.01 || f >= 0.999 {
                            item_clone.set_progress(f);
                            last = f;
                        }
                        if f >= 1.0 {
                            break;
                        }
                    }
                });
                let res = with_quiet_stdio_if_needed(args.quiet, || {
                    sel.backend.transcribe(
                        path,
                        &speaker,
                        lang_hint.as_deref(),
                        Some(tx),
                        args.gpu_layers,
                    )
                });
                let _ = recv_handle.join();
                match res {
                    Ok(items) => {
                        if matches!(mode, polyscribe::progress::ProgressMode::None) {
                            polyscribe::ilog!("done");
                        }
                        entries.extend(items);
                    }
                    Err(e) => {
                        if !polyscribe::is_no_interaction() && polyscribe::stdin_is_tty() {
                            polyscribe::elog!("{:#}", e);
                        }
                        return Err(e);
                    }
                }
            } else if is_json_file(path) {
                let mut buf = String::new();
                File::open(path)
                    .with_context(|| format!("Failed to open: {input_path}"))?
                    .read_to_string(&mut buf)
                    .with_context(|| format!("Failed to read: {input_path}"))?;
                let root: InputRoot = serde_json::from_str(&buf)
                    .with_context(|| format!("Invalid JSON transcript parsed from {input_path}"))?;
                for seg in root.segments {
                    entries.push(OutputEntry {
                        id: 0,
                        speaker: speaker.clone(),
                        start: seg.start,
                        end: seg.end,
                        text: seg.text,
                    });
                }
            } else {
                return Err(anyhow!(format!(
                    "Unsupported input type (expected .json or audio media): {}",
                    input_path
                )));
            }

            // Sort and reassign ids per file
            entries.sort_by(|a, b| {
                match a.start.partial_cmp(&b.start) {
                    Some(std::cmp::Ordering::Equal) | None => {}
                    Some(o) => return o,
                }
                a.end
                    .partial_cmp(&b.end)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, e) in entries.iter_mut().enumerate() {
                e.id = i as u64;
            }
            let out = OutputRoot { items: entries };

            if let Some(dir) = &out_dir {
                // Build file names using input stem
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");
                let date = date_prefix();
                let base_name = format!("{date}_{stem}");
                let base_path = dir.join(&base_name);
                write_outputs(&base_path, &out, &OutputFormats::all())?;
            } else {
                // stdout (only single input reaches here)
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                serde_json::to_writer_pretty(&mut handle, &out)?;
                writeln!(&mut handle)?;
            }
            // progress: mark file complete
            item.finish_with("done");
            progress.inc_completed();
            // record summary row
            summary.push((display_name, speaker.clone(), true, started_at.elapsed()));
        }

        // Final concise summary table to stderr (below progress bars)
        if !args.quiet && !summary.is_empty() {
            progress.println_above_bars("Summary:");
            progress.println_above_bars(&format!("{:<22} {:<18} {:<8} {:<8}", "File", "Speaker", "Status", "Time"));
            for (file, speaker, ok, dur) in summary {
                let status = if ok { "OK" } else { "ERR" };
                progress.println_above_bars(&format!(
                    "{:<22} {:<18} {:<8} {:<8}",
                    file,
                    speaker,
                    status,
                    format!("{:.2?}", dur)
                ));
            }
            // One blank line before finishing bars
            progress.println_above_bars("");
        }
    }

    // Finalize progress bars: keep total visible with final message
    progress.finish_all();
    // Final best-effort cleanup of .last_model on normal exit
    let _ = std::fs::remove_file(&last_model_path);
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
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cpu);
        // Vulkan only
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_VULKAN", "1");
        }
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Vulkan);
        // HIP preferred over Vulkan
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_HIP", "1");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_VULKAN");
        }
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Hip);
        // CUDA preferred over HIP
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_CUDA", "1");
        }
        let sel = select_backend(BackendKind::Auto, false).unwrap();
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
        assert!(select_backend(BackendKind::Cuda, false).is_err());
        assert!(select_backend(BackendKind::Hip, false).is_err());
        assert!(select_backend(BackendKind::Vulkan, false).is_err());
        // Turn on CUDA only
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_CUDA", "1");
        }
        assert!(select_backend(BackendKind::Cuda, false).is_ok());
        // Turn on HIP only
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std_env::set_var("POLYSCRIBE_TEST_FORCE_HIP", "1");
        }
        assert!(select_backend(BackendKind::Hip, false).is_ok());
        // Turn on Vulkan only
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
            std_env::set_var("POLYSCRIBE_TEST_FORCE_VULKAN", "1");
        }
        assert!(select_backend(BackendKind::Vulkan, false).is_ok());
        // Cleanup
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_VULKAN");
        }
    }
}
