// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

use std::fs::create_dir_all;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use clap_complete::Shell;

use polyscribe::backend::{BackendKind, collect_entries_for_input, select_backend};
use polyscribe::{
    OutputRoot, date_prefix, is_audio_file, models_dir_path, normalize_lang_code, sort_and_reindex,
    write_json_stdout, write_output_files,
};

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

    /// Choose transcription backend (auto|cpu|cuda|hip). GPU backends require a
    /// build with the matching cargo feature (gpu-cuda, gpu-hip). Default: auto.
    #[arg(long = "gpu-backend", value_enum, default_value_t = GpuBackendCli::Auto)]
    gpu_backend: GpuBackendCli,

    /// Reserved for future GPU layer offload control; currently unused.
    #[arg(long = "gpu-layers", value_name = "N", hide = true)]
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
    if polyscribe::is_no_interaction() {
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

fn run() -> Result<()> {
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

    // Select backend
    let requested = match args.gpu_backend {
        GpuBackendCli::Auto => BackendKind::Auto,
        GpuBackendCli::Cpu => BackendKind::Cpu,
        GpuBackendCli::Cuda => BackendKind::Cuda,
        GpuBackendCli::Hip => BackendKind::Hip,
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
    let mut inputs = args.inputs;
    let mut output_path = args.output;
    if output_path.is_none()
        && inputs.len() >= 2
        && let Some(last) = inputs.last().cloned()
        && !Path::new(&last).exists()
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
    let any_audio = inputs.iter().any(|p| is_audio_file(Path::new(p)));
    if any_audio && lang_hint.is_none() {
        return Err(anyhow!(
            "Please specify --language (e.g., --language en). Language detection was removed."
        ));
    }

    if args.merge_and_separate {
        polyscribe::dlog!(1, "Mode: merge-and-separate; output_dir={:?}", output_path);
        // Combined mode: write separate outputs per input and also a merged output set.
        // Require an output directory.
        let out_dir = match output_path.as_ref() {
            Some(p) => PathBuf::from(p),
            None => return Err(anyhow!("--merge-and-separate requires -o OUTPUT_DIR")),
        };
        if !out_dir.as_os_str().is_empty() {
            create_dir_all(&out_dir).with_context(|| {
                format!("Failed to create output directory: {}", out_dir.display())
            })?;
        }

        let mut merged_entries = Vec::new();
        for input_path in &inputs {
            let path = Path::new(input_path);
            let speaker = resolve_speaker_name(path, args.set_speaker_names);
            let mut entries = collect_entries_for_input(
                sel.backend.as_ref(),
                path,
                &speaker,
                lang_hint.as_deref(),
                args.gpu_layers,
            )?;
            sort_and_reindex(&mut entries);

            // Write separate outputs to out_dir
            let base_name = format!("{}_{}", date_prefix(), input_stem(path));
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
        polyscribe::dlog!(1, "Mode: merge; output_base={:?}", output_path);
        // MERGED MODE: concatenate all inputs into one sorted transcript.
        let mut entries = Vec::new();
        for input_path in &inputs {
            let path = Path::new(input_path);
            let speaker = resolve_speaker_name(path, args.set_speaker_names);
            entries.extend(collect_entries_for_input(
                sel.backend.as_ref(),
                path,
                &speaker,
                lang_hint.as_deref(),
                args.gpu_layers,
            )?);
        }
        sort_and_reindex(&mut entries);
        let out = OutputRoot { items: entries };

        if let Some(path) = output_path {
            let base_path = Path::new(&path);
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
        polyscribe::dlog!(1, "Mode: separate; output_dir={:?}", output_path);
        // SEPARATE MODE (default): one output set per input.
        // Writing to stdout with multiple inputs is not supported.
        if output_path.is_none() && inputs.len() > 1 {
            return Err(anyhow!(
                "Multiple inputs without --merge require -o OUTPUT_DIR to write separate files"
            ));
        }

        // If output_path is provided, treat it as a directory. Create it.
        let out_dir: Option<PathBuf> = output_path.as_ref().map(PathBuf::from);
        if let Some(dir) = &out_dir
            && !dir.as_os_str().is_empty()
        {
            create_dir_all(dir)
                .with_context(|| format!("Failed to create output directory: {}", dir.display()))?;
        }

        for input_path in &inputs {
            let path = Path::new(input_path);
            let speaker = resolve_speaker_name(path, args.set_speaker_names);
            let mut entries = collect_entries_for_input(
                sel.backend.as_ref(),
                path,
                &speaker,
                lang_hint.as_deref(),
                args.gpu_layers,
            )?;
            sort_and_reindex(&mut entries);
            let out = OutputRoot { items: entries };

            if let Some(dir) = &out_dir {
                // Build file names using input stem
                let base_name = format!("{}_{}", date_prefix(), input_stem(path));
                write_output_files(dir, &base_name, &out)?;
            } else {
                // stdout (only a single input reaches here)
                write_json_stdout(&out)?;
            }
        }
    }

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
    use polyscribe::{OutputEntry, format_srt_time, is_json_file, render_srt};
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
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
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
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
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
        assert!(is_audio_file(Path::new("video.mp4")));
        assert!(is_audio_file(Path::new("clip.WEBM")));
        assert!(is_audio_file(Path::new("movie.mkv")));
        assert!(is_audio_file(Path::new("trailer.MOV")));
        assert!(is_audio_file(Path::new("animation.avi")));
    }

    /// Set the test-only backend detection overrides explicitly. Setting a
    /// variable (even to "0") forces the value, which keeps these tests
    /// independent of the cargo features the suite is compiled with.
    fn force_backend_env(cuda: bool, hip: bool) {
        unsafe {
            std_env::set_var("POLYSCRIBE_TEST_FORCE_CUDA", if cuda { "1" } else { "0" });
            std_env::set_var("POLYSCRIBE_TEST_FORCE_HIP", if hip { "1" } else { "0" });
        }
    }

    #[test]
    fn test_backend_auto_order_prefers_cuda_then_hip_then_cpu() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // No GPU -> CPU
        force_backend_env(false, false);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cpu);
        // HIP only
        force_backend_env(false, true);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Hip);
        // CUDA preferred over HIP
        force_backend_env(true, true);
        let sel = select_backend(BackendKind::Auto, false).unwrap();
        assert_eq!(sel.chosen, BackendKind::Cuda);
        // Cleanup
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
        }
    }

    #[test]
    fn test_backend_explicit_missing_errors() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        // Ensure all off
        force_backend_env(false, false);
        assert!(select_backend(BackendKind::Cuda, false).is_err());
        assert!(select_backend(BackendKind::Hip, false).is_err());
        // Turn on CUDA only
        force_backend_env(true, false);
        assert!(select_backend(BackendKind::Cuda, false).is_ok());
        // Turn on HIP only
        force_backend_env(false, true);
        assert!(select_backend(BackendKind::Hip, false).is_ok());
        // Cleanup
        unsafe {
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_CUDA");
            std_env::remove_var("POLYSCRIBE_TEST_FORCE_HIP");
        }
    }

    #[test]
    fn test_cli_gpu_backend_values() {
        use clap::Parser;
        // Default is auto
        let args = Args::try_parse_from(["polyscribe", "a.json"]).unwrap();
        assert!(matches!(args.gpu_backend, GpuBackendCli::Auto));
        // Supported values parse
        for value in ["cpu", "cuda", "hip", "auto"] {
            let args =
                Args::try_parse_from(["polyscribe", "--gpu-backend", value, "a.json"]).unwrap();
            assert!(matches!(
                args.gpu_backend,
                GpuBackendCli::Cpu | GpuBackendCli::Cuda | GpuBackendCli::Hip | GpuBackendCli::Auto
            ));
        }
        // Vulkan is not a supported backend and must be rejected by the CLI
        let err = Args::try_parse_from(["polyscribe", "--gpu-backend", "vulkan", "a.json"])
            .expect_err("vulkan must not be an accepted backend value");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid value") && msg.contains("cuda"),
            "error should explain valid values, got: {msg}"
        );
    }
}
