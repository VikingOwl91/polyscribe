use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, ValueEnum)]
pub enum GpuBackend {
    Auto,
    Cpu,
    Cuda,
    Hip,
    Vulkan,
}

#[derive(Debug, Parser)]
#[command(name = "polyscribe", version, about = "PolyScribe – local-first transcription and plugins")]
pub struct Cli {
    /// Increase verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode (suppresses non-error logs)
    #[arg(short, long, default_value_t = false)]
    pub quiet: bool,

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

        /// GPU backend selection
        #[arg(long, value_enum, default_value_t = GpuBackend::Auto)]
        gpu_backend: GpuBackend,

        /// Offload N layers to GPU (when supported)
        #[arg(long, default_value_t = 0)]
        gpu_layers: usize,

        /// Input paths: audio/video files or JSON transcripts
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },

    /// Manage Whisper models
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

#[derive(Debug, Subcommand)]
pub enum ModelsCmd {
    /// Verify or update local models non-interactively
    Update,
    /// Interactive multi-select downloader
    Download,
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
        /// JSON payload string
        #[arg(long)]
        json: Option<String>,
    },
}