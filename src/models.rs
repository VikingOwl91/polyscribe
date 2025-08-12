// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! Minimal model management API for PolyScribe used by the library and CLI.
//! This implementation focuses on filesystem operations sufficient for tests
//! and basic non-interactive workflows. It can be extended later to support
//! remote discovery and verification.

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Pick the best local Whisper model in the given directory.
///
/// Heuristic: choose the largest .bin file by size. Returns None if none found.
pub fn pick_best_local_model(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    let rd = fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let p = e.path();
        if p.is_file() {
            if p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("bin")).unwrap_or(false) {
                if let Ok(md) = fs::metadata(&p) {
                    let sz = md.len();
                    match &best {
                        Some((b_sz, _)) if *b_sz >= sz => {}
                        _ => best = Some((sz, p.clone())),
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Ensure a model file with the given short name exists locally (non-interactive).
///
/// This stub creates an empty file named `<name>.bin` inside the models dir if it
/// does not yet exist, and returns its path. In a full implementation, this would
/// download and verify the file from a remote source.
pub fn ensure_model_available_noninteractive(name: &str) -> Result<PathBuf> {
    let models_dir = crate::models_dir_path();
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).with_context(|| {
            format!("Failed to create models dir: {}", models_dir.display())
        })?;
    }
    let filename = if name.ends_with(".bin") { name.to_string() } else { format!("{}.bin", name) };
    let path = models_dir.join(filename);
    if !path.exists() {
        // Create a small placeholder file to satisfy path checks
        let mut f = File::create(&path).with_context(|| format!("Failed to create model file: {}", path.display()))?;
        // Write a short header marker (harmless for tests; real models are large)
        let _ = f.write_all(b"POLYSCRIBE_PLACEHOLDER_MODEL\n");
    }
    Ok(path)
}

/// Run an interactive model downloader UI.
///
/// Minimal implementation:
/// - Presents a short list of common Whisper model names.
/// - Prompts the user to select models by comma-separated indices.
/// - Ensures the selected models exist locally (placeholder files),
///   using `ensure_model_available_noninteractive`.
/// - Respects --no-interaction by returning early with an info message.
pub fn run_interactive_model_downloader() -> Result<()> {
    use crate::ui;

    // Respect non-interactive mode
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        ui::info("Non-interactive mode: skipping interactive model downloader.");
        return Ok(());
    }

    // Available models (ordered from small to large). In a full implementation,
    // this would come from a remote manifest.
    let available = vec![
        ("tiny.en", "English-only tiny model (~75 MB)"),
        ("tiny", "Multilingual tiny model (~75 MB)"),
        ("base.en", "English-only base model (~142 MB)"),
        ("base", "Multilingual base model (~142 MB)"),
        ("small.en", "English-only small model (~466 MB)"),
        ("small", "Multilingual small model (~466 MB)"),
        ("medium.en", "English-only medium model (~1.5 GB)"),
        ("medium", "Multilingual medium model (~1.5 GB)"),
        ("large-v2", "Multilingual large v2 (~3.1 GB)"),
        ("large-v3", "Multilingual large v3 (~3.1 GB)"),
        ("large-v3-turbo", "Multilingual large v3 turbo (~1.5 GB)"),
    ];

    ui::intro("PolyScribe model downloader");
    ui::info("Select one or more models to download. Enter comma-separated numbers (e.g., 1,3,4). Press Enter to accept default [1].");
    ui::println_above_bars("Available models:");
    for (i, (name, desc)) in available.iter().enumerate() {
        ui::println_above_bars(format!("  {}. {:<16} – {}", i + 1, name, desc));
    }

    let answer = ui::prompt_input("Your selection", Some("1"))?;
    let selection_raw = match answer {
        Some(s) => s.trim().to_string(),
        None => "1".to_string(),
    };
    let selection = if selection_raw.is_empty() { "1" } else { &selection_raw };

    // Parse indices
    let mut picked_indices: Vec<usize> = Vec::new();
    for part in selection.split(|c| c == ',' || c == ' ' || c == ';') {
        let t = part.trim();
        if t.is_empty() { continue; }
        match t.parse::<usize>() {
            Ok(n) if n >= 1 && n <= available.len() => {
                let idx = n - 1;
                if !picked_indices.contains(&idx) {
                    picked_indices.push(idx);
                }
            }
            _ => {
                ui::warn(format!("Ignoring invalid selection: '{}'", t));
            }
        }
    }
    if picked_indices.is_empty() {
        // Fallback to default first item
        picked_indices.push(0);
    }

    // Prepare progress (TTY-aware)
    let labels: Vec<String> = picked_indices
        .iter()
        .map(|&i| available[i].0.to_string())
        .collect();
    let mut pm = ui::progress::ProgressManager::default_for_files(labels.len());
    pm.init_files(&labels);

    // Ensure models exist
    for (i, idx) in picked_indices.iter().enumerate() {
        let (name, _desc) = available[*idx];
        if let Some(pb) = pm.per_bar(i) {
            pb.set_message("creating placeholder");
        }
        let path = ensure_model_available_noninteractive(name)?;
        ui::println_above_bars(format!("Ready: {}", path.display()));
        pm.mark_file_done(i);
    }

    if let Some(total) = pm.total_bar() { total.finish_with_message("all done"); }
    ui::outro("Model selection complete.");
    Ok(())
}

/// Verify/update local models by comparing with a remote manifest.
///
/// Stub that currently succeeds and logs a short message.
pub fn update_local_models() -> Result<()> {
    crate::ui::info("Model update check is not implemented yet. Nothing to do.");
    Ok(())
}
