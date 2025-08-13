// SPDX-License-Identifier: MIT
//! Model management for PolyScribe: discovery, download, and verification.

use anyhow::{anyhow, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Represents a downloadable Whisper model artifact.
#[derive(Debug, Clone)]
struct ModelEntry {
    /// Display name and local short name (without extension if using default naming)
    name: &'static str,
    /// Remote file name (with extension)
    file: &'static str,
    /// Remote URL
    url: &'static str,
    /// Expected file size (optional)
    size: Option<u64>,
    /// Expected SHA-256 in hex (optional)
    sha256: Option<&'static str>,
}

/// Minimal built-in manifest.
/// You can extend this list or replace URLs to match your preferred source.
/// Large sizes/hashes are optional; leave None to skip checks.
fn builtin_manifest() -> Vec<ModelEntry> {
    // Example URLs (Hugging Face). Replace or extend as needed.
    // The filenames are typical GGUF/GGML whisper distributions.
    vec![
        ModelEntry {
            name: "tiny.en",
            file: "ggml-tiny.en.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "tiny",
            file: "ggml-tiny.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "base.en",
            file: "ggml-base.en.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "base",
            file: "ggml-base.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "small.en",
            file: "ggml-small.en.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "small",
            file: "ggml-small.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "medium.en",
            file: "ggml-medium.en.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "medium",
            file: "ggml-medium.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "large-v2",
            file: "ggml-large-v2.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "large-v3",
            file: "ggml-large-v3.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true",
            size: None,
            sha256: None,
        },
        ModelEntry {
            name: "large-v3-turbo",
            file: "ggml-large-v3-turbo.bin",
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true",
            size: None,
            sha256: None,
        },
    ]
}

/// Pick the best local Whisper model in the given directory.
/// Heuristic: choose the largest .bin file by size. Returns None if none found.
pub fn pick_best_local_model(dir: &Path) -> Option<PathBuf> {
    let rd = fs::read_dir(dir).ok()?;
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.eq_ignore_ascii_case("bin"))
        })
        .filter_map(|p| fs::metadata(&p).ok().map(|md| (md.len(), p)))
        .max_by_key(|(sz, _)| *sz)
        .map(|(_, p)| p)
}

/// Ensure a model file with the given short name exists locally (non-interactive).
/// It uses the built-in manifest to find URL and optionally verify size/hash.
pub fn ensure_model_available_noninteractive(name: &str) -> Result<PathBuf> {
    let Some(entry) = find_manifest_entry(name) else {
        return Err(anyhow!("unknown model name: {name}"));
    };

    let models_dir = crate::models_dir_path();
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).with_context(|| {
            format!("Failed to create models dir: {}", models_dir.display())
        })?;
    }
    let path = models_dir.join(entry.file);

    // If exists and passes checks, return early
    if path.exists() {
        if file_matches(&path, entry.size, entry.sha256)? {
            return Ok(path);
        }
        // Otherwise redownload
        crate::ilog!(
            "Existing model '{}' did not match expected checks; re-downloading.",
            entry.name
        );
        fs::remove_file(&path).ok();
    }

    // Download with progress to a temp file then atomically move.
    download_with_progress(&path, &entry)
        .with_context(|| format!("downloading {} from {}", entry.file, entry.url))?;

    // Final verification
    if !file_matches(&path, entry.size, entry.sha256)? {
        return Err(anyhow!(
            "downloaded file failed verification: {}",
            path.display()
        ));
    }

    Ok(path)
}

fn find_manifest_entry(name: &str) -> Option<ModelEntry> {
    // Accept either the short names in `name` field or a direct file name
    // For unknown suffixes, attempt stripping ".bin"
    let name_no_ext = name.strip_suffix(".bin").unwrap_or(name);
    for e in builtin_manifest() {
        if e.name.eq_ignore_ascii_case(name_no_ext) || e.file.eq_ignore_ascii_case(name) {
            return Some(e);
        }
    }
    None
}

fn file_matches(path: &Path, size: Option<u64>, sha256_hex: Option<&str>) -> Result<bool> {
    let md = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if let Some(sz) = size {
        if md.len() != sz {
            return Ok(false);
        }
    }
    if let Some(expected_hex) = sha256_hex {
        let mut f = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 128 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let got = hasher.finalize();
        let got_hex = hex::encode(got);
        if !got_hex.eq_ignore_ascii_case(expected_hex) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn download_with_progress(dest_path: &Path, entry: &ModelEntry) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("polyscribe/0.1")
        .build()?;

    crate::ilog!("Downloading {} …", entry.file);
    let mut resp = client.get(entry.url).send()?;
    if !resp.status().is_success() {
        return Err(anyhow!("HTTP {} for {}", resp.status(), entry.url));
    }

    let total_len = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .or(entry.size);

    // TTY-aware progress
    let pb = if !crate::is_quiet() && !crate::is_no_progress() && crate::stdin_is_tty() {
        let bar = ProgressBar::new(total_len.unwrap_or(0));
        bar.set_style(
            ProgressStyle::with_template("{bar:40.cyan/blue} {bytes}/{total_bytes} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        if let Some(t) = total_len {
            bar.set_length(t);
        }
        Some(bar)
    } else {
        None
    };

    let mut out_tmp = NamedTempFile::new_in(
        dest_path
            .parent()
            .ok_or_else(|| anyhow!("invalid destination path"))?,
    )?;
    let mut hasher = Sha256::new();
    let mut written: u64 = 0;

    // Read response body in chunks using a buffer
    let mut buffer = [0u8; 8192]; // 8KB buffer for reading chunks
    loop {
        let bytes_read = resp.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        let chunk = &buffer[..bytes_read];
        out_tmp.write_all(chunk)?;
        if entry.sha256.is_some() {
            hasher.update(chunk);
        }
        written += bytes_read as u64;
        if let Some(ref bar) = pb {
            if let Some(total) = total_len {
                bar.set_position(written.min(total));
            } else {
                bar.set_message(format!("{:.1} MB", (written as f64) / 1_000_000.0));
            }
        }
    }

    if let Some(sz) = entry.size {
        if written != sz {
            return Err(anyhow!(
                "incomplete download: expected {} bytes, got {}",
                sz,
                written
            ));
        }
    }

    if let Some(expected_hex) = entry.sha256 {
        let got_hex = hex::encode(hasher.finalize());
        if !got_hex.eq_ignore_ascii_case(expected_hex) {
            return Err(anyhow!("SHA-256 mismatch for {}", entry.file));
        }
    }

    out_tmp
        .persist(dest_path)
        .with_context(|| format!("persist {}", dest_path.display()))?;

    if let Some(bar) = pb {
        bar.finish_with_message("done");
    }
    Ok(())
}

/// Run an interactive model downloader UI.
/// - Lists models from the built-in manifest
/// - Prompts for selection
/// - Downloads selected models with verification
pub fn run_interactive_model_downloader() -> Result<()> {
    use crate::ui;

    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        ui::info("Non-interactive mode: skipping interactive model downloader.");
        return Ok(());
    }

    let available = builtin_manifest();

    ui::intro("PolyScribe model downloader");
    ui::info("Select one or more models to download. Enter comma-separated numbers (e.g., 1,3,4). Press Enter to accept default [1].");
    ui::println_above_bars("Available models:");
    for (i, m) in available.iter().enumerate() {
        ui::println_above_bars(format!("  {}. {:<18} – {}", i + 1, m.name, m.file));
    }

    let answer = ui::prompt_input("Your selection", Some("1"))?;
    let selection_raw = if answer.trim().is_empty() {
        "1".to_string()
    } else {
        answer.trim().to_string()
    };
    let selection = if selection_raw.is_empty() { "1" } else { &selection_raw };

    use std::collections::BTreeSet;
    let mut picked_set: BTreeSet<usize> = BTreeSet::new();
    for part in selection.split([',', ' ', ';']) {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        match t.parse::<usize>() {
            Ok(n) if (1..=available.len()).contains(&n) => {
                picked_set.insert(n - 1);
            }
            _ => ui::warn(format!("Ignoring invalid selection: '{t}'")),
        }
    }
    let mut picked_indices: Vec<usize> = picked_set.into_iter().collect();
    if picked_indices.is_empty() {
        picked_indices.push(0);
    }

    // Progress display (per-file style from UI)
    let labels: Vec<String> = picked_indices
        .iter()
        .map(|&i| available[i].name.to_string())
        .collect();
    let mut pm = ui::progress::ProgressManager::default_for_files(labels.len());
    pm.init_files(&labels);

    for (i, idx) in picked_indices.iter().enumerate() {
        let model = &available[*idx];
        if let Some(pb) = pm.per_bar(i) {
            pb.set_message("downloading");
        }
        let path = ensure_model_available_noninteractive(model.name)?;
        ui::println_above_bars(format!("Ready: {}", path.display()));
        pm.mark_file_done(i);
    }

    if let Some(total) = pm.total_bar() {
        total.finish_with_message("all done");
    }
    ui::outro("Model selection complete.");
    Ok(())
}

/// Verify/update local models by comparing with the built-in manifest.
/// - If a model file exists and matches expected size/hash (when provided), it is kept.
/// - If missing or mismatched, it will be downloaded.
pub fn update_local_models() -> Result<()> {
    use crate::ui;

    let manifest = builtin_manifest();
    let dir = crate::models_dir_path();
    fs::create_dir_all(&dir).ok();

    ui::info("Checking local models against manifest…");
    let mut fixed = 0usize;

    for m in manifest {
        let path = dir.join(m.file);
        let ok = path.exists() && file_matches(&path, m.size, m.sha256)?;
        if ok {
            crate::dlog!(1, "OK: {}", path.display());
            continue;
        }
        crate::ilog!("Updating {}", m.name);
        download_with_progress(&path, &m.clone())?;
        fixed += 1;
    }

    if fixed == 0 {
        ui::info("All models are up to date.");
    } else {
        ui::info(format!("Updated {fixed} model(s)."));
    }
    Ok(())
}
