// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

use std::fs::{File, create_dir_all};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum, CommandFactory};
use clap_complete::Shell;
use serde::{Deserialize, Serialize};

use polyscribe::{OutputEntry, date_prefix, normalize_lang_code, render_srt};

#[derive(Subcommand, Debug, Clone)]
enum AuxCommands {
    Completions {
        #[arg(value_enum)]
        shell: Shell,
    },
    Man,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
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
    /// Increase verbosity (-v, -vv). Repeat to increase.
    /// Debug logs appear with -v; very verbose with -vv. Logs go to stderr.
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Quiet mode: suppress non-error logging on stderr (overrides -v)
    /// Does not suppress interactive prompts or stdout output.
    #[arg(short = 'q', long = "quiet", global = true)]
    quiet: bool,

    /// Non-interactive mode: never prompt; use defaults instead.
    #[arg(long = "no-interaction", global = true)]
    no_interaction: bool,

    /// Disable interactive progress indicators (bars/spinners)
    #[arg(long = "no-progress", global = true)]
    no_progress: bool,

    /// Optional auxiliary subcommands (completions, man)
    #[command(subcommand)]
    aux: Option<AuxCommands>,

    /// Input .json transcript files or audio files to merge/transcribe
    inputs: Vec<String>,

    /// Output file path base or directory (date prefix added).
    /// In merge mode: base path.
    /// In separate mode: directory.
    /// If omitted: prints JSON to stdout for merge mode; separate mode requires directory for multiple inputs.
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,

    /// Merge all inputs into a single output; if not set, each input is written as a separate output
    #[arg(short = 'm', long = "merge")]
    merge: bool,

    /// Merge and also write separate outputs per input; requires -o OUTPUT_DIR
    #[arg(long = "merge-and-separate")]
    merge_and_separate: bool,

    /// Prompt for speaker names per input file
    #[arg(long = "set-speaker-names")]
    set_speaker_names: bool,

    /// Language code to use for transcription (e.g., en, de). No auto-detection.
    #[arg(short, long, value_name = "LANG")]
    language: Option<String>,

    /// Launch interactive model downloader (list HF models, multi-select and download)
    #[arg(long)]
    download_models: bool,

    /// Update local Whisper models by comparing hashes/sizes with remote manifest
    #[arg(long)]
    update_models: bool,
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
}

#[derive(Debug, Serialize)]
struct OutputRoot {
    items: Vec<OutputEntry>,
}

fn is_json_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()), Some(ext) if ext == "json")
}

fn is_audio_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()) {
        let exts = [
            "mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mkv", "mov", "avi",
            "m4b", "3gp", "opus", "aiff", "alac",
        ];
        return exts.contains(&ext.as_str());
    }
    false
}

fn validate_input_path(path: &Path) -> anyhow::Result<()> {
    let display = path.display();
    if !path.exists() {
        return Err(anyhow!("Input not found: {}", display));
    }
    let metadata = std::fs::metadata(path).with_context(|| format!("Failed to stat input: {}", display))?;
    if metadata.is_dir() {
        return Err(anyhow!("Input is a directory (expected a file): {}", display));
    }
    std::fs::File::open(path)
        .with_context(|| format!("Failed to open input file: {}", display))
        .map(|_| ())
}

fn sanitize_speaker_name(raw: &str) -> String {
    if let Some((prefix, rest)) = raw.split_once('-') {
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            return rest.to_string();
        }
    }
    raw.to_string()
}

fn prompt_speaker_name_for_path(
    _path: &Path,
    default_name: &str,
    enabled: bool,
) -> String {
    if !enabled || polyscribe::is_no_interaction() {
        return sanitize_speaker_name(default_name);
    }
    // TODO implement cliclack for this
    let mut input_line = String::new();
    match std::io::stdin().read_line(&mut input_line) {
        Ok(_) => {
            let trimmed = input_line.trim();
            if trimmed.is_empty() {
                sanitize_speaker_name(default_name)
            } else {
                sanitize_speaker_name(trimmed)
            }
        }
        Err(_) => sanitize_speaker_name(default_name),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize runtime flags for the library
    polyscribe::set_verbose(args.verbose);
    polyscribe::set_quiet(args.quiet);
    polyscribe::set_no_interaction(args.no_interaction);
    polyscribe::set_no_progress(args.no_progress);

    // Handle aux subcommands
    if let Some(aux) = &args.aux {
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
                let mut man_bytes = Vec::new();
                man.render(&mut man_bytes)?;
                io::stdout().write_all(&man_bytes)?;
                return Ok(());
            }
        }
    }

    // Optional model management actions
    if args.download_models {
        if let Err(err) = polyscribe::models::run_interactive_model_downloader() {
            polyscribe::elog!("Model downloader failed: {:#}", err);
        }
        if args.inputs.is_empty() {
            return Ok(())
        }
    }
    if args.update_models {
        if let Err(err) = polyscribe::models::update_local_models() {
            polyscribe::elog!("Model update failed: {:#}", err);
            return Err(err);
        }
        if args.inputs.is_empty() {
            return Ok(())
        }
    }

    // Process inputs
    let mut inputs = args.inputs;
    if inputs.is_empty() {
        return Err(anyhow!("No input files provided"));
    }

    // If last arg looks like an output path and not existing file, accept it as -o when multiple inputs
    let mut output_path = args.output;
    if output_path.is_none() && inputs.len() >= 2 {
        if let Some(candidate_output) = inputs.last().cloned() {
            if !Path::new(&candidate_output).exists() {
                inputs.pop();
                output_path = Some(candidate_output);
            }
        }
    }

    // Validate inputs; allow JSON and audio. For audio, require --language.
    for input_arg in &inputs {
        let path_ref = Path::new(input_arg);
        validate_input_path(path_ref)?;
        if !(is_json_file(path_ref) || is_audio_file(path_ref)) {
            return Err(anyhow!(
                "Unsupported input type (expected .json transcript or audio media): {}",
                path_ref.display()
            ));
        }
        if is_audio_file(path_ref) && args.language.is_none() {
            return Err(anyhow!("Please specify --language (e.g., --language en). Language detection was removed."));
        }
    }

    // Derive speakers (prompt if requested)
    let speakers: Vec<String> = inputs
        .iter()
        .map(|input_path| {
            let path = Path::new(input_path);
            let default_speaker = sanitize_speaker_name(
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("speaker"),
            );
            prompt_speaker_name_for_path(path, &default_speaker, args.set_speaker_names)
        })
        .collect();

    // MERGE-AND-SEPARATE mode
    if args.merge_and_separate {
        polyscribe::dlog!(1, "Mode: merge-and-separate; output_dir={:?}", output_path);
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
        for (idx, input_path) in inputs.iter().enumerate() {
            let path = Path::new(input_path);
            let speaker = speakers[idx].clone();
            // Decide based on input type (JSON transcript vs audio to transcribe)
            // TODO remove duplicate
            let mut entries: Vec<OutputEntry> = if is_json_file(path) {
                let mut buf = String::new();
                File::open(path)
                    .with_context(|| format!("Failed to open: {input_path}"))?
                    .read_to_string(&mut buf)
                    .with_context(|| format!("Failed to read: {input_path}"))?;
                let root: InputRoot = serde_json::from_str(&buf)
                    .with_context(|| format!("Invalid JSON transcript parsed from {input_path}"))?;
                root
                    .segments
                    .into_iter()
                    .map(|seg| OutputEntry { id: 0, speaker: speaker.clone(), start: seg.start, end: seg.end, text: seg.text })
                    .collect()
            } else {
                let lang_norm: Option<String> = args.language.as_deref().and_then(|s| normalize_lang_code(s));
                let selected_backend = polyscribe::backend::select_backend(polyscribe::backend::BackendKind::Auto, args.verbose > 0)?;
                selected_backend.backend.transcribe(path, &speaker, lang_norm.as_deref(), None, None)?
            };
            // Sort and id per-file
            // TODO remove duplicate
            entries.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal)));
            for (i, entry) in entries.iter_mut().enumerate() { entry.id = i as u64; }
            // Write per-file outputs
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            let date = date_prefix();
            let base_name = format!("{date}_{stem}");
            let json_path = out_dir.join(format!("{}.json", &base_name));
            let toml_path = out_dir.join(format!("{}.toml", &base_name));
            let srt_path = out_dir.join(format!("{}.srt", &base_name));

            let output_bundle = OutputRoot { items: entries.clone() };
            let mut json_file = File::create(&json_path).with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
            serde_json::to_writer_pretty(&mut json_file, &output_bundle)?; writeln!(&mut json_file)?;
            let toml_str = toml::to_string_pretty(&output_bundle)?;
            let mut toml_file = File::create(&toml_path).with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
            toml_file.write_all(toml_str.as_bytes())?; if !toml_str.ends_with('\n') { writeln!(&mut toml_file)?; }
            let srt_str = render_srt(&output_bundle.items);
            let mut srt_file = File::create(&srt_path).with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
            srt_file.write_all(srt_str.as_bytes())?;

            merged_entries.extend(output_bundle.items.into_iter());
        }
        // Write merged outputs into out_dir
        // TODO remove duplicate
        merged_entries.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal)));
        for (index, entry) in merged_entries.iter_mut().enumerate() { entry.id = index as u64; }
        let merged_output = OutputRoot { items: merged_entries };
        let date = date_prefix();
        let merged_base = format!("{date}_merged");
        let merged_json_path = out_dir.join(format!("{}.json", &merged_base));
        let merged_toml_path = out_dir.join(format!("{}.toml", &merged_base));
        let merged_srt_path = out_dir.join(format!("{}.srt", &merged_base));
        let mut merged_json_file = File::create(&merged_json_path).with_context(|| format!("Failed to create output file: {}", merged_json_path.display()))?;
        serde_json::to_writer_pretty(&mut merged_json_file, &merged_output)?; writeln!(&mut merged_json_file)?;
        let merged_toml_str = toml::to_string_pretty(&merged_output)?;
        let mut merged_toml_file = File::create(&merged_toml_path).with_context(|| format!("Failed to create output file: {}", merged_toml_path.display()))?;
        merged_toml_file.write_all(merged_toml_str.as_bytes())?; if !merged_toml_str.ends_with('\n') { writeln!(&mut merged_toml_file)?; }
        let merged_srt_str = render_srt(&merged_output.items);
        let mut merged_srt_file = File::create(&merged_srt_path).with_context(|| format!("Failed to create output file: {}", merged_srt_path.display()))?;
        merged_srt_file.write_all(merged_srt_str.as_bytes())?;
        return Ok(());
    }

    // MERGE mode
    if args.merge {
        polyscribe::dlog!(1, "Mode: merge; output_base={:?}", output_path);
        let mut entries: Vec<OutputEntry> = Vec::new();
        for (index, input_path) in inputs.iter().enumerate() {
            let path = Path::new(input_path);
            let speaker = speakers[index].clone();
            if is_json_file(path) {
                let mut buf = String::new();
                File::open(path)
                    .with_context(|| format!("Failed to open: {}", input_path))?
                    .read_to_string(&mut buf)
                    .with_context(|| format!("Failed to read: {}", input_path))?;
                let root: InputRoot = serde_json::from_str(&buf)
                    .with_context(|| format!("Invalid JSON transcript parsed from {}", input_path))?;
                for seg in root.segments {
                    entries.push(OutputEntry { id: 0, speaker: speaker.clone(), start: seg.start, end: seg.end, text: seg.text });
                }
            } else {
                let lang_norm: Option<String> = args.language.as_deref().and_then(|s| normalize_lang_code(s));
                let selected_backend = polyscribe::backend::select_backend(polyscribe::backend::BackendKind::Auto, args.verbose > 0)?;
                let mut new_entries = selected_backend.backend.transcribe(path, &speaker, lang_norm.as_deref(), None, None)?;
                entries.append(&mut new_entries);
            }
        }
        // TODO remove duplicate
        entries.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal)));
        for (i, entry) in entries.iter_mut().enumerate() { entry.id = i as u64; }
        let output_bundle = OutputRoot { items: entries };

        if let Some(path) = output_path {
            let base_path = Path::new(&path);
            let parent_opt = base_path.parent();
            if let Some(parent) = parent_opt {
                if !parent.as_os_str().is_empty() {
                    create_dir_all(parent).with_context(|| {
                        format!("Failed to create parent directory for output: {}", parent.display())
                    })?;
                }
            }
            let stem = base_path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            let date = date_prefix();
            let base_name = format!("{}_{}", date, stem);
            let dir = parent_opt.unwrap_or(Path::new(""));
            let json_path = dir.join(format!("{}.json", &base_name));
            let toml_path = dir.join(format!("{}.toml", &base_name));
            let srt_path = dir.join(format!("{}.srt", &base_name));

            let mut json_file = File::create(&json_path).with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
            serde_json::to_writer_pretty(&mut json_file, &output_bundle)?; writeln!(&mut json_file)?;
            let toml_str = toml::to_string_pretty(&output_bundle)?;
            let mut toml_file = File::create(&toml_path).with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
            toml_file.write_all(toml_str.as_bytes())?; if !toml_str.ends_with('\n') { writeln!(&mut toml_file)?; }
            let srt_str = render_srt(&output_bundle.items);
            let mut srt_file = File::create(&srt_path).with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
            srt_file.write_all(srt_str.as_bytes())?;
        } else {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            serde_json::to_writer_pretty(&mut handle, &output_bundle)?; writeln!(&mut handle)?;
        }
        return Ok(());
    }

    // SEPARATE (default)
    polyscribe::dlog!(1, "Mode: separate; output_dir={:?}", output_path);
    if output_path.is_none() && inputs.len() > 1 {
        return Err(anyhow!("Multiple inputs without --merge require -o OUTPUT_DIR to write separate files"));
    }
    let out_dir: Option<PathBuf> = output_path.as_ref().map(PathBuf::from);
    if let Some(dir) = &out_dir {
        if !dir.as_os_str().is_empty() {
            create_dir_all(dir).with_context(|| format!("Failed to create output directory: {}", dir.display()))?;
        }
    }

    for (index, input_path) in inputs.iter().enumerate() {
        let path = Path::new(input_path);
        let speaker = speakers[index].clone();
        // TODO remove duplicate
        let mut entries: Vec<OutputEntry> = if is_json_file(path) {
            let mut buf = String::new();
            File::open(path)
                .with_context(|| format!("Failed to open: {input_path}"))?
                .read_to_string(&mut buf)
                .with_context(|| format!("Failed to read: {input_path}"))?;
            let root: InputRoot = serde_json::from_str(&buf).with_context(|| format!("Invalid JSON transcript parsed from {input_path}"))?;
            root
                .segments
                .into_iter()
                .map(|seg| OutputEntry { id: 0, speaker: speaker.clone(), start: seg.start, end: seg.end, text: seg.text })
                .collect()
        } else {
            // Audio file: transcribe to entries
            let lang_norm: Option<String> = args.language.as_deref().and_then(|s| normalize_lang_code(s));
            let selected_backend = polyscribe::backend::select_backend(polyscribe::backend::BackendKind::Auto, args.verbose > 0)?;
            selected_backend.backend.transcribe(path, &speaker, lang_norm.as_deref(), None, None)?
        };
        // TODO remove duplicate
        entries.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal)));
        for (i, entry) in entries.iter_mut().enumerate() { entry.id = i as u64; }
        let output_bundle = OutputRoot { items: entries };

        if let Some(dir) = &out_dir {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            let date = date_prefix();
            let base_name = format!("{date}_{stem}");
            let json_path = dir.join(format!("{}.json", &base_name));
            let toml_path = dir.join(format!("{}.toml", &base_name));
            let srt_path = dir.join(format!("{}.srt", &base_name));

            let mut json_file = File::create(&json_path).with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
            serde_json::to_writer_pretty(&mut json_file, &output_bundle)?; writeln!(&mut json_file)?;
            let toml_str = toml::to_string_pretty(&output_bundle)?;
            let mut toml_file = File::create(&toml_path).with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
            toml_file.write_all(toml_str.as_bytes())?; if !toml_str.ends_with('\n') { writeln!(&mut toml_file)?; }
            let srt_str = render_srt(&output_bundle.items);
            let mut srt_file = File::create(&srt_path).with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
            srt_file.write_all(srt_str.as_bytes())?;
        } else {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            serde_json::to_writer_pretty(&mut handle, &output_bundle)?; writeln!(&mut handle)?;
        }
    }

    Ok(())
}
