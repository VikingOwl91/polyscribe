// SPDX-License-Identifier: MIT

//! The `transcribe` subcommand: transcribe audio/video files or merge JSON
//! transcripts into JSON/TOML/SRT outputs (or JSON on stdout).

use std::fs::create_dir_all;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

use polyscribe_core::backend::{
    BackendKind, TranscribeBackend, collect_entries_for_input, select_backend,
};
use polyscribe_core::fallback::FallbackBackend;
use polyscribe_core::output::{
    OutputRoot, is_audio_file, sort_and_reindex, write_json_stdout, write_output_files,
};
use polyscribe_core::ui::progress::{ProgressReporter, TranscriptProgress};
use polyscribe_core::voxtral_cloud::{CloudVoxtralBackend, DEFAULT_VOXTRAL_CLOUD_MODEL};
use polyscribe_core::{date_prefix, normalize_lang_code};

use crate::cli::{Engine, GpuBackend};

/// Arguments for the `transcribe` subcommand.
pub struct TranscribeArgs {
    pub output: Option<PathBuf>,
    pub merge: bool,
    pub merge_and_separate: bool,
    pub language: Option<String>,
    pub set_speaker_names: bool,
    pub engine: Engine,
    pub voxtral_model: Option<String>,
    pub yes: bool,
    pub gpu_backend: GpuBackend,
    pub gpu_layers: usize,
    pub inputs: Vec<PathBuf>,
}

/// The engine a run actually uses after `auto` is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedEngine {
    /// Local whisper.
    Whisper,
    /// Explicit `--engine voxtral-cloud`: no fallback, a cloud error aborts
    /// the run as before.
    VoxtralCloud,
    /// `--engine auto` resolved to the cloud: cloud first, and after the
    /// first cloud error the rest of the run falls back to local whisper.
    AutoCloud,
}

/// Resolve the requested engine. `auto` selects the Mistral cloud only when
/// the token FILE is configured (`voxtral_cloud::token_file_configured`);
/// the `MISTRAL_API_KEY` env var alone does not activate it, because other
/// tools set it too. Without the token file, `auto` is the local whisper
/// engine.
pub fn resolve_engine(engine: Engine) -> ResolvedEngine {
    match engine {
        Engine::Whisper => ResolvedEngine::Whisper,
        Engine::VoxtralCloud => ResolvedEngine::VoxtralCloud,
        Engine::Auto => {
            if polyscribe_core::voxtral_cloud::token_file_configured() {
                ResolvedEngine::AutoCloud
            } else {
                ResolvedEngine::Whisper
            }
        }
    }
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

    if args.yes {
        polyscribe_core::voxtral_cloud::set_cloud_consent(true);
    }

    progress.step("Selecting engine");
    let resolved = resolve_engine(args.engine);
    polyscribe_core::dlog!(1, "Resolved engine: {resolved:?}");
    let requested = match args.gpu_backend {
        GpuBackend::Auto => BackendKind::Auto,
        GpuBackend::Cpu => BackendKind::Cpu,
        GpuBackend::Cuda => BackendKind::Cuda,
        GpuBackend::Hip => BackendKind::Hip,
    };
    let verbose = polyscribe_core::verbose_level() > 0;
    let mut backend: Box<dyn TranscribeBackend + Send + Sync> = match resolved {
        ResolvedEngine::Whisper => {
            let sel = select_backend(requested, verbose)?;
            polyscribe_core::dlog!(1, "Using backend: {:?}", sel.chosen);
            if matches!(args.gpu_backend, GpuBackend::Auto)
                && sel.chosen == BackendKind::Cpu
                && sel.compiled.is_empty()
            {
                polyscribe_core::ilog!(
                    "No GPU backend is compiled into this binary; using the CPU backend. Build with --features gpu-hip for AMD ROCm or --features gpu-cuda for NVIDIA."
                );
            }
            sel.backend
        }
        ResolvedEngine::VoxtralCloud => {
            let model = args
                .voxtral_model
                .clone()
                .unwrap_or_else(|| DEFAULT_VOXTRAL_CLOUD_MODEL.to_string());
            polyscribe_core::dlog!(1, "Using engine voxtral-cloud with model {model}");
            Box::new(CloudVoxtralBackend::new(model)?)
        }
        ResolvedEngine::AutoCloud => {
            polyscribe_core::voxtral_cloud::warn_auto_cloud_once();
            let model = args
                .voxtral_model
                .clone()
                .unwrap_or_else(|| DEFAULT_VOXTRAL_CLOUD_MODEL.to_string());
            polyscribe_core::dlog!(
                1,
                "Engine auto: token file configured, using voxtral-cloud (model {model}) with a local whisper fallback"
            );
            Box::new(FallbackBackend::with_local_whisper_fallback(
                Box::new(CloudVoxtralBackend::new(model)?),
                requested,
                verbose,
            ))
        }
    };
    let gpu_layers = if args.gpu_layers > 0 {
        Some(args.gpu_layers as u32)
    } else {
        None
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
                format!(
                    "Failed to create output directory: {}",
                    polyscribe_core::ui::terminal_safe_path(&out_dir)
                )
            })?;
        }

        let mut merged_entries = Vec::new();
        for input_path in &inputs {
            let speaker = resolve_speaker_name(input_path, args.set_speaker_names);
            let mut entries = collect_entries_with_progress(
                backend.as_mut(),
                input_path,
                &speaker,
                lang_hint.as_deref(),
                gpu_layers,
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
            entries.extend(collect_entries_with_progress(
                backend.as_mut(),
                input_path,
                &speaker,
                lang_hint.as_deref(),
                gpu_layers,
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
                        polyscribe_core::ui::terminal_safe_path(parent)
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
            create_dir_all(dir).with_context(|| {
                format!(
                    "Failed to create output directory: {}",
                    polyscribe_core::ui::terminal_safe_path(dir)
                )
            })?;
        }

        for input_path in &inputs {
            let speaker = resolve_speaker_name(input_path, args.set_speaker_names);
            let mut entries = collect_entries_with_progress(
                backend.as_mut(),
                input_path,
                &speaker,
                lang_hint.as_deref(),
                gpu_layers,
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

fn collect_entries_with_progress(
    backend: &mut (dyn TranscribeBackend + Send + Sync),
    input_path: &Path,
    speaker: &str,
    lang_hint: Option<&str>,
    gpu_layers: Option<u32>,
) -> Result<Vec<polyscribe_core::OutputEntry>> {
    if !is_audio_file(input_path) {
        return collect_entries_for_input(
            backend, input_path, speaker, lang_hint, gpu_layers, None,
        )
        .map_err(anyhow::Error::from);
    }
    let name = input_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("input");
    let progress = TranscriptProgress::start(format!("Transcribing {name}"));
    let progress_cb = |percent: i32| {
        progress.update(percent);
        if !progress.is_enabled() {
            polyscribe_core::dlog!(1, "transcription progress: {percent}%");
        }
    };
    let result = collect_entries_for_input(
        backend,
        input_path,
        speaker,
        lang_hint,
        gpu_layers,
        Some(&progress_cb),
    )
    .map_err(anyhow::Error::from);
    progress.finish(if result.is_ok() {
        "Transcript ready"
    } else {
        "Transcription failed"
    });
    result
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
        .map(polyscribe_core::ui::escape_terminal_controls)
        .unwrap_or_else(|| polyscribe_core::ui::terminal_safe_path(path));
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

    /// Serializes tests that mutate process environment variables.
    static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    #[test]
    fn test_resolve_engine_explicit_values_pass_through_without_fallback() {
        // Explicit whisper stays whisper; explicit voxtral-cloud stays
        // voxtral-cloud and is NOT wrapped in a fallback (only `auto` gets
        // one).
        assert_eq!(resolve_engine(Engine::Whisper), ResolvedEngine::Whisper);
        assert_eq!(
            resolve_engine(Engine::VoxtralCloud),
            ResolvedEngine::VoxtralCloud
        );
    }

    #[test]
    fn test_resolve_engine_auto() {
        use polyscribe_core::voxtral_cloud::{ENV_API_KEY, ENV_TOKEN_FILE};
        let _guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let old_key = std::env::var(ENV_API_KEY).ok();
        let old_file = std::env::var_os(ENV_TOKEN_FILE);
        // Always point the override into the temp dir, so the real token
        // file of this machine is never read.
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("mistral_token");

        // No token file -> local whisper.
        unsafe { std::env::set_var(ENV_TOKEN_FILE, dir.path().join("absent")) };
        assert_eq!(resolve_engine(Engine::Auto), ResolvedEngine::Whisper);

        // MISTRAL_API_KEY set but no token file -> still local whisper.
        unsafe { std::env::set_var(ENV_API_KEY, "env-only-key") };
        assert_eq!(resolve_engine(Engine::Auto), ResolvedEngine::Whisper);

        // Empty token file -> local whisper.
        std::fs::write(&token, "  \n").unwrap();
        unsafe { std::env::set_var(ENV_TOKEN_FILE, &token) };
        assert_eq!(resolve_engine(Engine::Auto), ResolvedEngine::Whisper);

        // Token file with a key -> cloud, with the local fallback.
        std::fs::write(&token, "some-key\n").unwrap();
        assert_eq!(resolve_engine(Engine::Auto), ResolvedEngine::AutoCloud);

        match old_key {
            Some(v) => unsafe { std::env::set_var(ENV_API_KEY, v) },
            None => unsafe { std::env::remove_var(ENV_API_KEY) },
        }
        match old_file {
            Some(v) => unsafe { std::env::set_var(ENV_TOKEN_FILE, v) },
            None => unsafe { std::env::remove_var(ENV_TOKEN_FILE) },
        }
    }
}
