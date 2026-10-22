mod cli;
mod output;
mod record;
mod transcribe;
mod youtube;

use anyhow::{Context, Result, anyhow};
use clap::{CommandFactory, Parser};
use cli::{Cli, Commands, ModelCommon, ModelsCmd, PluginsCmd};
use output::OutputMode;
use polyscribe_core::model_manager::{ModelManager, ReqwestClient, Settings};
use polyscribe_core::ui;
fn normalized_similarity(a: &str, b: &str) -> f64 {
    // simple Levenshtein distance; normalized to [0,1]
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let n = a_bytes.len();
    let m = b_bytes.len();
    if n == 0 && m == 0 {
        return 1.0;
    }
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    let dist = prev[m] as f64;
    let max_len = n.max(m) as f64;
    1.0 - (dist / max_len)
}

fn human_size(bytes: Option<u64>) -> String {
    match bytes {
        Some(n) => {
            let x = n as f64;
            const KB: f64 = 1024.0;
            const MB: f64 = 1024.0 * KB;
            const GB: f64 = 1024.0 * MB;
            if x >= GB {
                format!("{:.2} GiB", x / GB)
            } else if x >= MB {
                format!("{:.2} MiB", x / MB)
            } else if x >= KB {
                format!("{:.2} KiB", x / KB)
            } else {
                format!("{} B", n)
            }
        }
        None => "?".to_string(),
    }
}
use polyscribe_host::PluginManager;
use tokio::io::AsyncWriteExt;
use tracing_subscriber::EnvFilter;

fn init_tracing(json_mode: bool, quiet: bool, verbose: u8) {
    // In JSON mode, suppress human logs; route errors to stderr only.
    let level = if json_mode || quiet {
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
        .with_writer(std::io::stderr)
        .compact()
        .init();
}

fn main() -> Result<()> {
    let args = Cli::parse();

    // Determine output mode early for logging and UI configuration
    let output_mode = if args.output.json {
        OutputMode::Json
    } else {
        OutputMode::Human {
            quiet: args.output.quiet,
        }
    };

    init_tracing(
        matches!(output_mode, OutputMode::Json),
        args.output.quiet,
        args.verbose,
    );

    // Suppress decorative UI output in JSON mode as well
    polyscribe_core::set_quiet(args.output.quiet || matches!(output_mode, OutputMode::Json));
    polyscribe_core::set_no_interaction(args.no_interaction);
    polyscribe_core::set_verbose(args.verbose);
    polyscribe_core::set_no_progress(args.no_progress);

    match args.command {
        Commands::Transcribe {
            output,
            merge,
            merge_and_separate,
            language,
            set_speaker_names,
            gpu_backend,
            gpu_layers,
            inputs,
        } => transcribe::run(transcribe::TranscribeArgs {
            output,
            merge,
            merge_and_separate,
            language,
            set_speaker_names,
            gpu_backend,
            gpu_layers,
            inputs,
        }),

        Commands::Record {
            model,
            language,
            device,
            window,
            output,
            gpu_backend,
        } => record::run(record::RecordArgs {
            model,
            language,
            device,
            window,
            output,
            gpu_backend,
            json: matches!(output_mode, OutputMode::Json),
        }),

        Commands::Models { cmd } => {
            // predictable exit codes
            const EXIT_OK: i32 = 0;
            const EXIT_NOT_FOUND: i32 = 2;
            const EXIT_NETWORK: i32 = 3;
            const EXIT_VERIFY_FAILED: i32 = 4;
            // const EXIT_NO_CHANGE: i32 = 5; // reserved

            let handle_common = |c: &ModelCommon| Settings {
                concurrency: c.concurrency.max(1),
                limit_rate: c.limit_rate,
                ..Default::default()
            };

            let exit = match cmd {
                ModelsCmd::Ls { common } => {
                    let mm: ModelManager<ReqwestClient> =
                        ModelManager::new(handle_common(&common))?;
                    let list = mm.ls()?;
                    match output_mode {
                        OutputMode::Json => {
                            // Always emit JSON array (possibly empty)
                            output_mode.print_json(&list);
                        }
                        OutputMode::Human { quiet } => {
                            if list.is_empty() {
                                if !quiet {
                                    println!("No models installed.");
                                }
                            } else {
                                if !quiet {
                                    println!("Model (Repo)");
                                }
                                for r in list {
                                    if !quiet {
                                        println!("{} ({})", r.file, r.repo);
                                    }
                                }
                            }
                        }
                    }
                    EXIT_OK
                }
                ModelsCmd::Add { repo, file, common } => {
                    let settings = handle_common(&common);
                    let mm: ModelManager<ReqwestClient> = ModelManager::new(settings.clone())?;
                    // Derive an alias automatically from repo and file
                    fn derive_alias(repo: &str, file: &str) -> String {
                        use std::path::Path;
                        let repo_tail = repo.rsplit('/').next().unwrap_or(repo);
                        let stem = Path::new(file)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(file);
                        format!("{}-{}", repo_tail, stem)
                    }
                    let alias = derive_alias(&repo, &file);
                    match mm.add_or_update(&alias, &repo, &file) {
                        Ok(rec) => {
                            match output_mode {
                                OutputMode::Json => output_mode.print_json(&rec),
                                OutputMode::Human { quiet } => {
                                    if !quiet {
                                        println!("installed: {} -> {}/{}", alias, repo, rec.file);
                                    }
                                }
                            }
                            EXIT_OK
                        }
                        Err(e) => {
                            // On not found or similar errors, try suggesting close matches interactively
                            if matches!(output_mode, OutputMode::Json)
                                || polyscribe_core::is_no_interaction()
                            {
                                match output_mode {
                                    OutputMode::Json => {
                                        // Emit error JSON object
                                        #[derive(serde::Serialize)]
                                        struct ErrObj<'a> {
                                            error: &'a str,
                                        }
                                        let eo = ErrObj {
                                            error: &e.to_string(),
                                        };
                                        output_mode.print_json(&eo);
                                    }
                                    _ => {
                                        eprintln!("error: {e}");
                                    }
                                }
                                EXIT_NOT_FOUND
                            } else {
                                ui::warn(format!("{}", e));
                                ui::info("Searching for similar model filenames…");
                                match polyscribe_core::model_manager::search_repo(&repo, None) {
                                    Ok(mut files) => {
                                        if files.is_empty() {
                                            ui::warn("No files found in repository.");
                                            EXIT_NOT_FOUND
                                        } else {
                                            // rank by similarity
                                            files.sort_by(|a, b| {
                                                normalized_similarity(&file, b)
                                                    .partial_cmp(&normalized_similarity(&file, a))
                                                    .unwrap_or(std::cmp::Ordering::Equal)
                                            });
                                            let top: Vec<String> =
                                                files.into_iter().take(5).collect();
                                            if top.is_empty() {
                                                EXIT_NOT_FOUND
                                            } else if top.len() == 1 {
                                                let cand = &top[0];
                                                // Fetch repo size list once
                                                let size_map: std::collections::HashMap<String, Option<u64>> =
                                                    polyscribe_core::model_manager::list_repo_files_with_meta(&repo)
                                                    .unwrap_or_default()
                                                    .into_iter().collect();
                                                let mut size =
                                                    size_map.get(cand).cloned().unwrap_or(None);
                                                if size.is_none() {
                                                    size = polyscribe_core::model_manager::head_len_for_file(&repo, cand);
                                                }
                                                let local_files: std::collections::HashSet<String> =
                                                    mm.ls()?.into_iter().map(|r| r.file).collect();
                                                let is_local = local_files.contains(cand);
                                                let label = format!(
                                                    "{} [{}]{}",
                                                    cand,
                                                    human_size(size),
                                                    if is_local { " (local)" } else { "" }
                                                );
                                                let ok = ui::prompt_confirm(
                                                    &format!("Did you mean {}?", label),
                                                    true,
                                                )
                                                .unwrap_or(false);
                                                if !ok {
                                                    EXIT_NOT_FOUND
                                                } else {
                                                    let mm2: ModelManager<ReqwestClient> =
                                                        ModelManager::new(settings)?;
                                                    let alias2 = derive_alias(&repo, cand);
                                                    match mm2.add_or_update(&alias2, &repo, cand) {
                                                        Ok(rec) => {
                                                            match output_mode {
                                                                OutputMode::Json => {
                                                                    output_mode.print_json(&rec)
                                                                }
                                                                OutputMode::Human { quiet } => {
                                                                    if !quiet {
                                                                        println!(
                                                                            "installed: {} -> {}/{}",
                                                                            alias2, repo, rec.file
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                            EXIT_OK
                                                        }
                                                        Err(e2) => {
                                                            eprintln!("error: {e2}");
                                                            EXIT_NETWORK
                                                        }
                                                    }
                                                }
                                            } else {
                                                let opts: Vec<String> = top;
                                                let local_files: std::collections::HashSet<String> =
                                                    mm.ls()?.into_iter().map(|r| r.file).collect();
                                                // Enrich labels with size and local tag using a single API call
                                                let size_map: std::collections::HashMap<String, Option<u64>> =
                                                    polyscribe_core::model_manager::list_repo_files_with_meta(&repo)
                                                    .unwrap_or_default()
                                                    .into_iter().collect();
                                                let mut labels_owned: Vec<String> = Vec::new();
                                                for f in &opts {
                                                    let mut size =
                                                        size_map.get(f).cloned().unwrap_or(None);
                                                    if size.is_none() {
                                                        size = polyscribe_core::model_manager::head_len_for_file(&repo, f);
                                                    }
                                                    let is_local = local_files.contains(f);
                                                    let suffix =
                                                        if is_local { " (local)" } else { "" };
                                                    labels_owned.push(format!(
                                                        "{} [{}]{}",
                                                        f,
                                                        human_size(size),
                                                        suffix
                                                    ));
                                                }
                                                let labels: Vec<&str> = labels_owned
                                                    .iter()
                                                    .map(|s| s.as_str())
                                                    .collect();
                                                match ui::prompt_select("Pick a model", &labels) {
                                                    Ok(idx) => {
                                                        let chosen = &opts[idx];
                                                        let mm2: ModelManager<ReqwestClient> =
                                                            ModelManager::new(settings)?;
                                                        let alias2 = derive_alias(&repo, chosen);
                                                        match mm2
                                                            .add_or_update(&alias2, &repo, chosen)
                                                        {
                                                            Ok(rec) => {
                                                                match output_mode {
                                                                    OutputMode::Json => {
                                                                        output_mode.print_json(&rec)
                                                                    }
                                                                    OutputMode::Human { quiet } => {
                                                                        if !quiet {
                                                                            println!(
                                                                                "installed: {} -> {}/{}",
                                                                                alias2,
                                                                                repo,
                                                                                rec.file
                                                                            );
                                                                        }
                                                                    }
                                                                }
                                                                EXIT_OK
                                                            }
                                                            Err(e2) => {
                                                                eprintln!("error: {e2}");
                                                                EXIT_NETWORK
                                                            }
                                                        }
                                                    }
                                                    Err(_) => EXIT_NOT_FOUND,
                                                }
                                            }
                                        }
                                    }
                                    Err(e2) => {
                                        eprintln!("error: {}", e2);
                                        EXIT_NETWORK
                                    }
                                }
                            }
                        }
                    }
                }
                ModelsCmd::Rm { alias, common } => {
                    let mm: ModelManager<ReqwestClient> =
                        ModelManager::new(handle_common(&common))?;
                    let ok = mm.rm(&alias)?;
                    match output_mode {
                        OutputMode::Json => {
                            #[derive(serde::Serialize)]
                            struct R {
                                removed: bool,
                            }
                            output_mode.print_json(&R { removed: ok });
                        }
                        OutputMode::Human { quiet } => {
                            if !quiet {
                                println!("{}", if ok { "removed" } else { "not found" });
                            }
                        }
                    }
                    if ok { EXIT_OK } else { EXIT_NOT_FOUND }
                }
                ModelsCmd::Verify { alias, common } => {
                    let mm: ModelManager<ReqwestClient> =
                        ModelManager::new(handle_common(&common))?;
                    let found = mm.ls()?.into_iter().any(|r| r.alias == alias);
                    if !found {
                        match output_mode {
                            OutputMode::Json => {
                                #[derive(serde::Serialize)]
                                struct R<'a> {
                                    ok: bool,
                                    error: &'a str,
                                }
                                output_mode.print_json(&R {
                                    ok: false,
                                    error: "not found",
                                });
                            }
                            OutputMode::Human { quiet } => {
                                if !quiet {
                                    println!("not found");
                                }
                            }
                        }
                        EXIT_NOT_FOUND
                    } else {
                        let ok = mm.verify(&alias)?;
                        match output_mode {
                            OutputMode::Json => {
                                #[derive(serde::Serialize)]
                                struct R {
                                    ok: bool,
                                }
                                output_mode.print_json(&R { ok });
                            }
                            OutputMode::Human { quiet } => {
                                if !quiet {
                                    println!("{}", if ok { "ok" } else { "corrupt" });
                                }
                            }
                        }
                        if ok { EXIT_OK } else { EXIT_VERIFY_FAILED }
                    }
                }
                ModelsCmd::Update { common } => {
                    let mm: ModelManager<ReqwestClient> =
                        ModelManager::new(handle_common(&common))?;
                    let mut rc = EXIT_OK;
                    for rec in mm.ls()? {
                        match mm.add_or_update(&rec.alias, &rec.repo, &rec.file) {
                            Ok(_) => {}
                            Err(e) => {
                                rc = EXIT_NETWORK;
                                match output_mode {
                                    OutputMode::Json => {
                                        #[derive(serde::Serialize)]
                                        struct R<'a> {
                                            alias: &'a str,
                                            error: String,
                                        }
                                        output_mode.print_json(&R {
                                            alias: &rec.alias,
                                            error: e.to_string(),
                                        });
                                    }
                                    _ => {
                                        eprintln!("update {}: {e}", rec.alias);
                                    }
                                }
                            }
                        }
                    }
                    rc
                }
                ModelsCmd::Download => {
                    // Interactive downloader (skips itself in non-interactive mode)
                    polyscribe_core::models::run_interactive_model_downloader()?;
                    EXIT_OK
                }
                ModelsCmd::ClearCache => {
                    polyscribe_core::models::clear_manifest_cache()?;
                    match output_mode {
                        OutputMode::Json => {
                            #[derive(serde::Serialize)]
                            struct R {
                                cleared: bool,
                            }
                            output_mode.print_json(&R { cleared: true });
                        }
                        OutputMode::Human { quiet } => {
                            if !quiet {
                                println!("cleared manifest cache");
                            }
                        }
                    }
                    EXIT_OK
                }
                ModelsCmd::Gc { common } => {
                    let mm: ModelManager<ReqwestClient> =
                        ModelManager::new(handle_common(&common))?;
                    let (files_removed, entries_removed) = mm.gc()?;
                    match output_mode {
                        OutputMode::Json => {
                            #[derive(serde::Serialize)]
                            struct R {
                                files_removed: usize,
                                entries_removed: usize,
                            }
                            output_mode.print_json(&R {
                                files_removed,
                                entries_removed,
                            });
                        }
                        OutputMode::Human { quiet } => {
                            if !quiet {
                                println!(
                                    "files_removed={} entries_removed={}",
                                    files_removed, entries_removed
                                );
                            }
                        }
                    }
                    EXIT_OK
                }
                ModelsCmd::Search { repo, query, .. } => {
                    let res = polyscribe_core::model_manager::search_repo(&repo, query.as_deref());
                    match res {
                        Ok(files) => {
                            match output_mode {
                                OutputMode::Json => output_mode.print_json(&files),
                                OutputMode::Human { quiet } => {
                                    for f in files {
                                        if !quiet {
                                            println!("{}", f);
                                        }
                                    }
                                }
                            }
                            EXIT_OK
                        }
                        Err(e) => {
                            match output_mode {
                                OutputMode::Json => {
                                    #[derive(serde::Serialize)]
                                    struct R {
                                        error: String,
                                    }
                                    output_mode.print_json(&R {
                                        error: e.to_string(),
                                    });
                                }
                                _ => {
                                    eprintln!("error: {e}");
                                }
                            }
                            EXIT_NETWORK
                        }
                    }
                }
            };
            std::process::exit(exit);
        }

        Commands::Plugins { cmd } => {
            let plugin_manager = PluginManager;

            match cmd {
                PluginsCmd::List => {
                    let list = plugin_manager.list().context("discovering plugins")?;
                    // Results go to stdout so the output is scriptable.
                    for item in list {
                        println!("{}", item.name);
                    }
                    Ok(())
                }
                PluginsCmd::Info { name } => {
                    let info = plugin_manager
                        .info(&name)
                        .with_context(|| format!("getting info for {}", name))?;
                    let info_json = serde_json::to_string_pretty(&info)?;
                    println!("{info_json}");
                    Ok(())
                }
                PluginsCmd::Run {
                    name,
                    command,
                    payload: payload_arg,
                } => {
                    // Use a local Tokio runtime only for this async path
                    let rt = tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .context("building tokio runtime")?;

                    rt.block_on(async {
                        let payload = payload_arg.unwrap_or_else(|| "{}".to_string());
                        // Build a JSON-RPC request line for the plugin.
                        let params = serde_json::from_str::<serde_json::Value>(&payload)
                            .unwrap_or(serde_json::Value::String(payload));
                        let request = polyscribe_protocol::Request {
                            id: "1".to_string(),
                            method: command.clone(),
                            params: Some(params),
                        };
                        let line = serde_json::to_string(&request)?;
                        let mut child = plugin_manager
                            .spawn(&name)
                            .with_context(|| format!("spawning plugin {name} {command}"))?;

                        if let Some(mut stdin) = child.stdin.take() {
                            stdin
                                .write_all(line.as_bytes())
                                .await
                                .context("writing JSON-RPC request to plugin stdin")?;
                            stdin
                                .write_all(b"\n")
                                .await
                                .context("writing JSON-RPC request newline to plugin stdin")?;
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
                    })
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

        Commands::YouTube {
            url,
            language,
            format,
            output,
            prefer_manual,
        } => youtube::run(url, language, format, output, prefer_manual),
    }
}
