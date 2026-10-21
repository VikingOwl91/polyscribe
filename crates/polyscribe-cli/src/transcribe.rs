// SPDX-License-Identifier: MIT

//! The `transcribe` subcommand: transcribe audio/video files or merge JSON
//! transcripts into JSON/TOML/SRT outputs (or JSON on stdout).

use std::fs::create_dir_all;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

use polyscribe_core::backend::{BackendKind, collect_entries_for_input, select_backend};
use polyscribe_core::output::{
    OutputRoot, is_audio_file, sort_and_reindex, write_json_stdout, write_output_files,
};
use polyscribe_core::ui::progress::ProgressReporter;
use polyscribe_core::{date_prefix, normalize_lang_code};

use crate::cli::GpuBackend;

/// Arguments for the `transcribe` subcommand.
pub struct TranscribeArgs {
    pub output: Option<PathBuf>,
    pub merge: bool,
    pub merge_and_separate: bool,
    pub language: Option<String>,
    pub set_speaker_names: bool,
    pub gpu_backend: GpuBackend,
    pub gpu_layers: usize,
    pub inputs: Vec<PathBuf>,
}

pub fn run(args: TranscribeArgs) -> Result<()> {
    let mut progress = ProgressReporter::new(polyscribe_core::is_no_interaction());
    progress.step("Validating inputs");

    // Determine inputs and optional output path. With at least two inputs and
    // no -o, a trailing argument that does not exist on disk is treated as the
    // output path (historical behavior).
    polyscribe_core::dlog!(1, "Parsed {} input(s)", args.inputs.len());
    let mut inputs = args.inputs;
    let mut output_path = args.output;
    if output_path.is_none()
        && inputs.len() >= 2
        && let Some(last) = inputs.last().cloned()
        && !last.exists()
    {
        inputs.pop();
        output_path = Some(last);
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
    let any_audio = inputs.iter().any(|p| is_audio_file(p));
    if any_audio && lang_hint.is_none() {
        return Err(anyhow!(
            "Please specify --language (e.g., --language en). Language detection was removed."
        ));
    }

    progress.step("Selecting backend");
    let requested = match args.gpu_backend {
        GpuBackend::Auto => BackendKind::Auto,
        GpuBackend::Cpu => BackendKind::Cpu,
        GpuBackend::Cuda => BackendKind::Cuda,
        GpuBackend::Hip => BackendKind::Hip,
    };
    let mut sel = select_backend(requested, polyscribe_core::verbose_level() > 0)?;
    polyscribe_core::dlog!(1, "Using backend: {:?}", sel.chosen);
    let gpu_layers = if args.gpu_layers > 0 {
        Some(args.gpu_layers as u32)
    } else {
        None
    };
    let progress_cb = |p: i32| {
        polyscribe_core::dlog!(1, "transcription progress: {p}%");
    };

    if args.merge_and_separate {
        polyscribe_core::dlog!(1, "Mode: merge-and-separate; output_dir={:?}", output_path);
        // Combined mode: write separate outputs per input and also a merged output set.
        // Require an output directory.
        let out_dir = match output_path.as_ref() {
            Some(p) => p.clone(),
            None => return Err(anyhow!("--merge-and-separate requires -o OUTPUT_DIR")),
        };
        if !out_dir.as_os_str().is_empty() {
            create_dir_all(&out_dir).with_context(|| {
                format!("Failed to create output directory: {}", out_dir.display())
            })?;
        }

        let mut merged_entries = Vec::new();
        for input_path in &inputs {
            let speaker = resolve_speaker_name(input_path, args.set_speaker_names);
            let mut entries = collect_entries_for_input(
                sel.backend.as_mut(),
                input_path,
                &speaker,
                lang_hint.as_deref(),
                gpu_layers,
                Some(&progress_cb),
            )?;
            sort_and_reindex(&mut entries);

            // Write separate outputs to out_dir
            let base_name = format!("{}_{}", date_prefix(), input_stem(input_path));
            write_output_files(
                &out_dir,
                &base_name,
                &OutputRoot {
                    items: entries.clone(),
                },
            )?;

            // Extend merged with per-file entries
            merged_entries.extend(entries);
        }

        // Now write merged output set into out_dir
        sort_and_reindex(&mut merged_entries);
        let merged_base = format!("{}_merged", date_prefix());
        write_output_files(
            &out_dir,
            &merged_base,
            &OutputRoot {
                items: merged_entries,
            },
        )?;
    } else if args.merge {
        polyscribe_core::dlog!(1, "Mode: merge; output_base={:?}", output_path);
        // MERGED MODE: concatenate all inputs into one sorted transcript.
        let mut entries = Vec::new();
        for input_path in &inputs {
            let speaker = resolve_speaker_name(input_path, args.set_speaker_names);
            entries.extend(collect_entries_for_input(
                sel.backend.as_mut(),
                input_path,
                &speaker,
                lang_hint.as_deref(),
                gpu_layers,
                Some(&progress_cb),
            )?);
        }
        sort_and_reindex(&mut entries);
        let out = OutputRoot { items: entries };

        if let Some(path) = output_path {
            let base_path = &path;
            let parent_opt = base_path.parent();
            if let Some(parent) = parent_opt
                && !parent.as_os_str().is_empty()
            {
                create_dir_all(parent).with_context(|| {
                    format!(
                        "Failed to create parent directory for output: {}",
                        parent.display()
                    )
                })?;
            }
            let stem = base_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let dir = parent_opt.unwrap_or(Path::new(""));
            let base_name = format!("{}_{}", date_prefix(), stem);
            write_output_files(dir, &base_name, &out)?;
        } else {
            write_json_stdout(&out)?;
        }
    } else {
        polyscribe_core::dlog!(1, "Mode: separate; output_dir={:?}", output_path);
        // SEPARATE MODE (default): one output set per input.
        // Writing to stdout with multiple inputs is not supported.
        if output_path.is_none() && inputs.len() > 1 {
            return Err(anyhow!(
                "Multiple inputs without --merge require -o OUTPUT_DIR to write separate files"
            ));
        }

        // If output_path is provided, treat it as a directory. Create it.
        let out_dir: Option<PathBuf> = output_path.clone();
        if let Some(dir) = &out_dir
            && !dir.as_os_str().is_empty()
        {
            create_dir_all(dir)
                .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;
        }

        for input_path in &inputs {
            let speaker = resolve_speaker_name(input_path, args.set_speaker_names);
            let mut entries = collect_entries_for_input(
                sel.backend.as_mut(),
                input_path,
                &speaker,
                lang_hint.as_deref(),
                gpu_layers,
                Some(&progress_cb),
            )?;
            sort_and_reindex(&mut entries);
            let out = OutputRoot { items: entries };

            if let Some(dir) = &out_dir {
                // Build file names using input stem
                let base_name = format!("{}_{}", date_prefix(), input_stem(input_path));
                write_output_files(dir, &base_name, &out)?;
            } else {
                // stdout (only a single input reaches here)
                write_json_stdout(&out)?;
            }
        }
    }

    progress.finish_with_message("Transcription completed");
    Ok(())
}

fn sanitize_speaker_name(raw: &str) -> String {
    if let Some((prefix, rest)) = raw.split_once('-')
        && !prefix.is_empty()
        && prefix.chars().all(|c| c.is_ascii_digit())
    {
        return rest.to_string();
    }
    raw.to_string()
}

fn prompt_speaker_name_for_path(path: &Path, default_name: &str, enabled: bool) -> String {
    if !enabled {
        return default_name.to_string();
    }
    if polyscribe_core::is_no_interaction() {
        // Explicitly non-interactive: never prompt
        return default_name.to_string();
    }
    let display_owned: String = path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    eprint!("Enter speaker name for {display_owned} [default: {default_name}]: ");
    io::stderr().flush().ok();
    let mut buf = String::new();
    match io::stdin().read_line(&mut buf) {
        Ok(_) => {
            let raw = buf.trim();
            if raw.is_empty() {
                return default_name.to_string();
            }
            let sanitized = sanitize_speaker_name(raw);
            if sanitized.is_empty() {
                default_name.to_string()
            } else {
                sanitized
            }
        }
        Err(_) => default_name.to_string(),
    }
}

/// Return the file stem of an input path, or "output" if it has none.
fn input_stem(path: &Path) -> &str {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output")
}

/// Resolve the speaker label for an input: the sanitized file stem by default,
/// or a prompted name when speaker prompting is enabled.
fn resolve_speaker_name(path: &Path, prompt: bool) -> String {
    let default = sanitize_speaker_name(
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("speaker"),
    );
    prompt_speaker_name_for_path(path, &default, prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_input_stem() {
        assert_eq!(input_stem(Path::new("a/b/c.json")), "c");
        assert_eq!(input_stem(Path::new("noext")), "noext");
    }

    #[test]
    fn test_resolve_speaker_name_defaults_without_prompt() {
        // Without prompting, the default is the sanitized file stem.
        assert_eq!(
            resolve_speaker_name(Path::new("input/1-s0wlz.json"), false),
            "s0wlz"
        );
        assert_eq!(
            resolve_speaker_name(Path::new("input/2-vikingowl.json"), false),
            "vikingowl"
        );
    }
}
