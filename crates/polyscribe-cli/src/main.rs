mod cli;

use anyhow::{Context, Result, anyhow};
use clap::{CommandFactory, Parser};
use cli::{Cli, Commands, GpuBackend, ModelsCmd, PluginsCmd};
use polyscribe_core::models;
use polyscribe_core::ui::progress::ProgressReporter;
use polyscribe_host::PluginManager;
use tokio::io::AsyncWriteExt;
use tracing_subscriber::EnvFilter;

fn init_tracing(quiet: bool, verbose: u8) {
    let log_level = if quiet {
        "error"
    } else {
        match verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        }
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
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

    polyscribe_core::set_quiet(args.quiet);
    polyscribe_core::set_no_interaction(args.no_interaction);
    polyscribe_core::set_verbose(args.verbose);
    polyscribe_core::set_no_progress(args.no_progress);

    match args.command {
        Commands::Transcribe {
            gpu_backend,
            gpu_layers,
            inputs,
            ..
        } => {
            polyscribe_core::ui::info("starting transcription workflow");
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
                    polyscribe_core::ui::info("verifying/updating local models");
                    tokio::task::spawn_blocking(models::update_local_models)
                        .await
                        .map_err(|e| anyhow!("blocking task join error: {e}"))?
                        .context("updating models")?;
                }
                ModelsCmd::Download => {
                    polyscribe_core::ui::info("interactive model selection and download");
                    tokio::task::spawn_blocking(models::run_interactive_model_downloader)
                        .await
                        .map_err(|e| anyhow!("blocking task join error: {e}"))?
                        .context("running downloader")?;
                    polyscribe_core::ui::success("Model download complete.");
                }
                ModelsCmd::ClearCache => {
                    polyscribe_core::ui::info("clearing manifest cache");
                    tokio::task::spawn_blocking(models::clear_manifest_cache)
                        .await
                        .map_err(|e| anyhow!("blocking task join error: {e}"))?
                        .context("clearing cache")?;
                    polyscribe_core::ui::success("Manifest cache cleared.");
                }
            }
            Ok(())
        }

        Commands::Plugins { cmd } => {
            let plugin_manager = PluginManager;

            match cmd {
                PluginsCmd::List => {
                    let list = plugin_manager.list().context("discovering plugins")?;
                    for item in list {
                        polyscribe_core::ui::info(item.name);
                    }
                    Ok(())
                }
                PluginsCmd::Info { name } => {
                    let info = plugin_manager
                        .info(&name)
                        .with_context(|| format!("getting info for {}", name))?;
                    let info_json = serde_json::to_string_pretty(&info)?;
                    polyscribe_core::ui::info(info_json);
                    Ok(())
                }
                PluginsCmd::Run {
                    name,
                    command,
                    json,
                } => {
                    let payload = json.unwrap_or_else(|| "{}".to_string());
                    let mut child = plugin_manager
                        .spawn(&name, &command)
                        .with_context(|| format!("spawning plugin {name} {command}"))?;

                    if let Some(mut stdin) = child.stdin.take() {
                        stdin
                            .write_all(payload.as_bytes())
                            .await
                            .context("writing JSON payload to plugin stdin")?;
                    }

                    let status = plugin_manager.forward_stdio(&mut child).await?;
                    if !status.success() {
                        polyscribe_core::ui::error(format!(
                            "plugin returned non-zero exit code: {}",
                            status
                        ));
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
