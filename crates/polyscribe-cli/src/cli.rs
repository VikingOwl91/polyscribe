use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, ValueEnum)]
pub enum GpuBackend {
    Auto,
    Cpu,
    Cuda,
    Hip,
}

#[derive(Debug, Clone, Args)]
pub struct OutputOpts {
    /// Emit machine-readable JSON to stdout; suppress decorative logs
    #[arg(long, global = true, action = clap::ArgAction::SetTrue)]
    pub json: bool,
    /// Reduce log chatter (errors only unless --json)
    #[arg(long, global = true, action = clap::ArgAction::SetTrue)]
    pub quiet: bool,
}

#[derive(Debug, Parser)]
#[command(
    name = "polyscribe",
    version,
    about = "PolyScribe – local-first transcription and plugins",
    propagate_version = true,
    arg_required_else_help = true
)]
pub struct Cli {
    /// Global output options
    #[command(flatten)]
    pub output: OutputOpts,

    /// Increase verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Never prompt for user input (non-interactive mode)
    #[arg(long, default_value_t = false)]
    pub no_interaction: bool,

    /// Disable progress bars/spinners
    #[arg(long, default_value_t = false)]
    pub no_progress: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Transcribe audio/video files or merge existing transcripts
    Transcribe {
        /// Output file or directory (date prefix is added when directory)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Merge multiple inputs into one output
        #[arg(short = 'm', long, default_value_t = false)]
        merge: bool,

        /// Write both merged and per-input outputs (requires -o dir)
        #[arg(long, default_value_t = false)]
        merge_and_separate: bool,

        /// Language code hint, e.g. en, de
        #[arg(long)]
        language: Option<String>,

        /// Prompt for a speaker label per input file
        #[arg(long, default_value_t = false)]
        set_speaker_names: bool,

        /// Transcription backend among those compiled in (auto|cpu|cuda|hip).
        /// GPU backends require a build with the matching cargo feature
        /// (gpu-cuda, gpu-hip). Default: auto.
        #[arg(long, value_enum, default_value_t = GpuBackend::Auto)]
        gpu_backend: GpuBackend,

        /// Reserved for future GPU layer offload control; currently unused.
        #[arg(long, default_value_t = 0, hide = true)]
        gpu_layers: usize,

        /// Input paths: audio/video files or JSON transcripts
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },

    /// Manage Whisper GGUF models (Hugging Face)
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },

    /// Discover and run plugins
    Plugins {
        #[command(subcommand)]
        cmd: PluginsCmd,
    },

    /// Generate shell completions to stdout
    Completions {
        /// Shell to generate completions for
        #[arg(value_parser = ["bash", "zsh", "fish", "powershell", "elvish"])]
        shell: String,
    },

    /// Generate a man page to stdout
    Man,
}

#[derive(Debug, Clone, Parser)]
pub struct ModelCommon {
    /// Concurrency for ranged downloads
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Limit download rate in bytes/sec (approximate)
    #[arg(long)]
    pub limit_rate: Option<u64>,
}

#[derive(Debug, Subcommand)]
pub enum ModelsCmd {
    /// List installed models (from manifest)
    Ls {
        #[command(flatten)]
        common: ModelCommon,
    },
    /// Add or update a model
    Add {
        /// Hugging Face repo, e.g. ggml-org/models
        repo: String,
        /// File name in repo (e.g., gguf-tiny-q4_0.bin)
        file: String,
        #[command(flatten)]
        common: ModelCommon,
    },
    /// Remove a model by alias
    Rm {
        alias: String,
        #[command(flatten)]
        common: ModelCommon,
    },
    /// Verify model file integrity by alias
    Verify {
        alias: String,
        #[command(flatten)]
        common: ModelCommon,
    },
    /// Update all models (HEAD + ETag; skip if unchanged)
    Update {
        #[command(flatten)]
        common: ModelCommon,
    },
    /// Launch the interactive model downloader (lists Hugging Face models)
    Download,
    /// Clear the cached Hugging Face model manifest (force fresh fetch)
    ClearCache,
    /// Garbage-collect unreferenced files and stale manifest entries
    Gc {
        #[command(flatten)]
        common: ModelCommon,
    },
    /// Search a repo for GGUF files
    Search {
        /// Hugging Face repo, e.g. ggml-org/models
        repo: String,
        /// Optional substring to filter filenames
        #[arg(long)]
        query: Option<String>,
        #[command(flatten)]
        common: ModelCommon,
    },
}

#[derive(Debug, Subcommand)]
pub enum PluginsCmd {
    /// List installed plugins
    List,
    /// Show a plugin's capabilities (as JSON)
    Info {
        /// Plugin short name, e.g., "tubescribe"
        name: String,
    },
    /// Run a plugin command (JSON-RPC over NDJSON via stdio)
    Run {
        /// Plugin short name
        name: String,
        /// Command name in plugin's API
        command: String,
        /// JSON payload string (must not be named --json: that clashes with
        /// the global --json output flag)
        #[arg(long)]
        payload: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_gpu_backend_values() {
        // Default is auto
        let cli = Cli::try_parse_from(["polyscribe", "transcribe", "a.json"]).unwrap();
        let Commands::Transcribe { gpu_backend, .. } = cli.command else {
            panic!("expected transcribe command");
        };
        assert!(matches!(gpu_backend, GpuBackend::Auto));
        // Supported values parse
        for value in ["auto", "cpu", "cuda", "hip"] {
            Cli::try_parse_from(["polyscribe", "transcribe", "--gpu-backend", value, "a.json"])
                .unwrap();
        }
        // Vulkan is not a supported backend and must be rejected by the CLI
        let err = Cli::try_parse_from([
            "polyscribe",
            "transcribe",
            "--gpu-backend",
            "vulkan",
            "a.json",
        ])
        .expect_err("vulkan must not be an accepted backend value");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid value") && msg.contains("cuda"),
            "error should explain valid values, got: {msg}"
        );
    }

    #[test]
    fn test_models_subcommands_parse() {
        for args in [
            &["models", "ls"][..],
            &["models", "add", "repo", "file.bin"][..],
            &["models", "rm", "alias"][..],
            &["models", "verify", "alias"][..],
            &["models", "update"][..],
            &["models", "download"][..],
            &["models", "clear-cache"][..],
            &["models", "gc"][..],
            &["models", "search", "repo"][..],
        ] {
            let mut full = vec!["polyscribe"];
            full.extend_from_slice(args);
            Cli::try_parse_from(full).unwrap();
        }
    }

    #[test]
    fn test_plugins_subcommands_parse() {
        for args in [
            &["plugins", "list"][..],
            &["plugins", "info", "tubescribe"][..],
            &["plugins", "run", "tubescribe", "generate_metadata"][..],
        ] {
            let mut full = vec!["polyscribe"];
            full.extend_from_slice(args);
            Cli::try_parse_from(full).unwrap();
        }
    }

    #[test]
    fn test_transcribe_requires_inputs() {
        assert!(Cli::try_parse_from(["polyscribe", "transcribe"]).is_err());
    }

    #[test]
    fn test_completions_and_man_parse() {
        Cli::try_parse_from(["polyscribe", "completions", "bash"]).unwrap();
        Cli::try_parse_from(["polyscribe", "man"]).unwrap();
        assert!(Cli::try_parse_from(["polyscribe", "completions", "nope"]).is_err());
    }
}
