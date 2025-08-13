mod cli;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, CommandFactory};
use cli::{Cli, Commands, GpuBackend, ModelsCmd, PluginsCmd};
use polyscribe_core::{config::ConfigService, ui::progress::ProgressReporter};
use polyscribe_core::models; // Added: call into core models
use polyscribe_host::PluginManager;
use tokio::io::AsyncWriteExt;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn init_tracing(quiet: bool, verbose: u8) {
    let level = if quiet {
        "error"
    } else {
        match verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        }
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(true)
        .compact()
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    init_tracing(args.quiet, args.verbose);

    // Optionally propagate quiet/no-interaction/verbosity to core if your lib exposes setters.
    // polyscribe_core::set_quiet(args.quiet);
    // polyscribe_core::set_no_interaction(args.no_interaction);
    // polyscribe_core::set_verbose(args.verbose);

    let _cfg = ConfigService::load_or_default().context("loading configuration")?;

    match args.command {
        Commands::Transcribe {
            output: _output,
            merge: _merge,
            merge_and_separate: _merge_and_separate,
            language: _language,
            set_speaker_names: _set_speaker_names,
            gpu_backend,
            gpu_layers,
            inputs,
        } => {
            info!("starting transcription workflow");
            let mut progress = ProgressReporter::new(args.no_interaction);

            progress.step("Validating inputs");
            if inputs.is_empty() {
                return Err(anyhow!("no inputs provided"));
            }

            progress.step("Selecting backend and preparing model");
            match gpu_backend {
                GpuBackend::Auto => {}
                GpuBackend::Cpu => {}
                GpuBackend::Cuda => {
                    let _ = gpu_layers;
                }
                GpuBackend::Hip => {}
                GpuBackend::Vulkan => {}
            }

            progress.finish_with_message("Transcription completed (stub)");
            Ok(())
        }

        Commands::Models { cmd } => {
            match cmd {
                ModelsCmd::Update => {
                    info!("verifying/updating local models");
                    tokio::task::spawn_blocking(|| models::update_local_models())
                        .await
                        .map_err(|e| anyhow!("blocking task join error: {e}"))?
                        .context("updating models")?;
                    println!("Models updated.");
                }
                ModelsCmd::Download => {
                    info!("interactive model selection and download");
                    tokio::task::spawn_blocking(|| models::run_interactive_model_downloader())
                        .await
                        .map_err(|e| anyhow!("blocking task join error: {e}"))?
                        .context("running downloader")?;
                    println!("Model download complete.");
                }

            }
            Ok(())
        }

        Commands::Plugins { cmd } => {
            let pm = PluginManager::default();

            match cmd {
                PluginsCmd::List => {
                    let list = pm.list().context("discovering plugins")?;
                    for item in list {
                        println!("{}", item.name);
                    }
                    Ok(())
                }
                PluginsCmd::Info { name } => {
                    let info = pm.info(&name).with_context(|| format!("getting info for {}", name))?;
                    println!("{}", serde_json::to_string_pretty(&info)?);
                    Ok(())
                }
                PluginsCmd::Run { name, command, json } => {
                    let payload = json.unwrap_or_else(|| "{}".to_string());
                    let mut child = pm
                        .spawn(&name, &command)
                        .with_context(|| format!("spawning plugin {name} {command}"))?;

                    if let Some(mut stdin) = child.stdin.take() {
                        stdin
                            .write_all(payload.as_bytes())
                            .await
                            .context("writing JSON payload to plugin stdin")?;
                    }

                    let status = pm.forward_stdio(&mut child).await?;
                    if !status.success() {
                        error!("plugin returned non-zero exit code: {}", status);
                        return Err(anyhow!("plugin failed"));
                    }
                    Ok(())
                }
            }
        }

        Commands::Completions { shell } => {
            use clap_complete::{generate, shells};
            use std::io;

            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();

            match shell.as_str() {
                "bash" => generate(shells::Bash, &mut cmd, name, &mut io::stdout()),
                "zsh" => generate(shells::Zsh, &mut cmd, name, &mut io::stdout()),
                "fish" => generate(shells::Fish, &mut cmd, name, &mut io::stdout()),
                "powershell" => generate(shells::PowerShell, &mut cmd, name, &mut io::stdout()),
                "elvish" => generate(shells::Elvish, &mut cmd, name, &mut io::stdout()),
                _ => return Err(anyhow!("unsupported shell: {shell}")),
            }
            Ok(())
        }

        Commands::Man => {
            use clap_mangen::Man;
            let cmd = Cli::command();
            let man = Man::new(cmd);
            man.render(&mut std::io::stdout())?;
            Ok(())
        }
    }
}
