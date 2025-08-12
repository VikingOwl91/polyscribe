// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! Model discovery, selection, and downloading logic for PolyScribe.

use std::any::Any;
use std::collections::BTreeMap;
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use atty::Stream;
use clap::builder::Str;
// --- Model downloader: list & download ggml models from Hugging Face ---

#[derive(Debug, Deserialize)]
struct HFLfsMeta {
    oid: Option<String>,
    size: Option<u64>,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HFSibling {
    rfilename: String,
    size: Option<u64>,
    sha256: Option<String>,
    lfs: Option<HFLfsMeta>,
}

#[derive(Debug, Deserialize)]
struct HFRepoInfo {
    // When using ?expand=files the field is named 'siblings'
    siblings: Option<Vec<HFSibling>>,
}

#[derive(Debug, Deserialize)]
struct HFTreeItem {
    path: String,
    size: Option<u64>,
    sha256: Option<String>,
    lfs: Option<HFLfsMeta>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelEntry {
    // e.g. "tiny.en-q5_1"
    name: String,
    base: String,
    subtype: String,
    size: u64,
    sha256: Option<String>,
    repo: String, // e.g. "ggerganov/whisper.cpp"
}

fn split_model_name(model: &str) -> (String, String) {
    let mut idx = None;
    for (i, ch) in model.char_indices() {
        if ch == '.' || ch == '-' {
            idx = Some(i);
            break;
        }
    }
    if let Some(i) = idx {
        (model[..i].to_string(), model[i + 1..].to_string())
    } else {
        (model.to_string(), String::new())
    }
}

fn human_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

fn to_hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn expected_sha_from_sibling(s: &HFSibling) -> Option<String> {
    if let Some(h) = &s.sha256 {
        return Some(h.to_lowercase());
    }
    if let Some(lfs) = &s.lfs {
        if let Some(h) = &lfs.sha256 {
            return Some(h.to_lowercase());
        }
        if let Some(oid) = &lfs.oid {
            // e.g. "sha256:abcdef..."
            if let Some(rest) = oid.strip_prefix("sha256:") {
                return Some(rest.to_lowercase().to_string());
            }
        }
    }
    None
}

fn size_from_sibling(s: &HFSibling) -> Option<u64> {
    if let Some(sz) = s.size {
        return Some(sz);
    }
    if let Some(lfs) = &s.lfs {
        return lfs.size;
    }
    None
}

fn expected_sha_from_tree(s: &HFTreeItem) -> Option<String> {
    if let Some(h) = &s.sha256 {
        return Some(h.to_lowercase());
    }
    if let Some(lfs) = &s.lfs {
        if let Some(h) = &lfs.sha256 {
            return Some(h.to_lowercase());
        }
        if let Some(oid) = &lfs.oid {
            if let Some(rest) = oid.strip_prefix("sha256:") {
                return Some(rest.to_lowercase().to_string());
            }
        }
    }
    None
}

fn size_from_tree(s: &HFTreeItem) -> Option<u64> {
    if let Some(sz) = s.size {
        return Some(sz);
    }
    if let Some(lfs) = &s.lfs {
        return lfs.size;
    }
    None
}

fn fill_meta_via_head(repo: &str, name: &str) -> (Option<u64>, Option<String>) {
    let head_client = match Client::builder()
        .user_agent("PolyScribe/0.1 (+https://github.com/)")
        .redirect(Policy::none())
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(_) => return (None, None),
    };
    let url = format!("https://huggingface.co/{repo}/resolve/main/ggml-{name}.bin");
    let resp = match head_client
        .head(url)
        .send()
        .and_then(|r| r.error_for_status())
    {
        Ok(r) => r,
        Err(_) => return (None, None),
    };
    let headers = resp.headers();
    let size = headers
        .get("x-linked-size")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let mut sha = headers
        .get("x-linked-etag")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().trim_matches('"').to_string());
    if let Some(h) = &mut sha {
        h.make_ascii_lowercase();
        if h.len() != 64 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
            sha = None;
        }
    }
    // Fallback: try x-xet-hash header if x-linked-etag is missing/invalid
    if sha.is_none() {
        sha = headers
            .get("x-xet-hash")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().trim_matches('"').to_string());
        if let Some(h) = &mut sha {
            h.make_ascii_lowercase();
            if h.len() != 64 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
                sha = None;
            }
        }
    }
    (size, sha)
}

fn hf_fetch_repo_models(client: &Client, repo: &'static str) -> Result<Vec<ModelEntry>> {
    if !(crate::is_no_interaction() && crate::verbose_level() < 2) {
        ilog!("Fetching online data: listing models from {}...", repo);
    }
    // Prefer the tree endpoint for reliable size/hash metadata, then fall back to model metadata
    let tree_url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=1");
    let mut out: Vec<ModelEntry> = Vec::new();

    match client
        .get(tree_url)
        .send()
        .and_then(|r| r.error_for_status())
    {
        Ok(resp) => {
            match resp.json::<Vec<HFTreeItem>>() {
                Ok(items) => {
                    for it in items {
                        let path = it.path.clone();
                        if !(path.starts_with("ggml-") && path.ends_with(".bin")) {
                            continue;
                        }
                        let model_name = path
                            .trim_start_matches("ggml-")
                            .trim_end_matches(".bin")
                            .to_string();
                        let (base, subtype) = split_model_name(&model_name);
                        let size = size_from_tree(&it).unwrap_or(0);
                        let sha256 = expected_sha_from_tree(&it);
                        out.push(ModelEntry {
                            name: model_name,
                            base,
                            subtype,
                            size,
                            sha256,
                            repo: repo.to_string(),
                        });
                    }
                }
                Err(_) => { /* fall back below */ }
            }
        }
        Err(_) => { /* fall back below */ }
    }

    if out.is_empty() {
        let url = format!("https://huggingface.co/api/models/{repo}");
        let resp = client
            .get(url)
            .send()
            .and_then(|r| r.error_for_status())
            .context("Failed to query Hugging Face API")?;

        let info: HFRepoInfo = resp
            .json()
            .context("Failed to parse Hugging Face API response")?;

        if let Some(files) = info.siblings {
            for s in files {
                let rf = s.rfilename.clone();
                if !(rf.starts_with("ggml-") && rf.ends_with(".bin")) {
                    continue;
                }
                let model_name = rf
                    .trim_start_matches("ggml-")
                    .trim_end_matches(".bin")
                    .to_string();
                let (base, subtype) = split_model_name(&model_name);
                let size = size_from_sibling(&s).unwrap_or(0);
                let sha256 = expected_sha_from_sibling(&s);
                out.push(ModelEntry {
                    name: model_name,
                    base,
                    subtype,
                    size,
                    sha256,
                    repo: repo.to_string(),
                });
            }
        }
    }

    // Fill missing metadata (size/hash) via HEAD request if necessary
    if out.iter().any(|m| m.size == 0 || m.sha256.is_none())
        && !(crate::is_no_interaction() && crate::verbose_level() < 2)
    {
        ilog!(
            "Fetching online data: completing metadata checks for models in {}...",
            repo
        );
    }
    for m in out.iter_mut() {
        if m.size == 0 || m.sha256.is_none() {
            let (sz, sha) = fill_meta_via_head(&m.repo, &m.name);
            if m.size == 0 {
                if let Some(s) = sz {
                    m.size = s;
                }
            }
            if m.sha256.is_none() {
                m.sha256 = sha;
            }
        }
    }

    // Sort by base then subtype then name for stable listing
    out.sort_by(|a, b| {
        a.base
            .cmp(&b.base)
            .then(a.subtype.cmp(&b.subtype))
            .then(a.name.cmp(&b.name))
    });
    Ok(out)
}

fn fetch_all_models(client: &Client) -> Result<Vec<ModelEntry>> {
    if !(crate::is_no_interaction() && crate::verbose_level() < 2) {
        ilog!("Fetching online data: aggregating available models from Hugging Face...");
    }
    let mut v1 = hf_fetch_repo_models(client, "ggerganov/whisper.cpp")?; // main repo must succeed

    // Optional tinydiarize repo; ignore errors but log to stderr
    let mut v2: Vec<ModelEntry> =
        match hf_fetch_repo_models(client, "akashmjn/tinydiarize-whisper.cpp") {
            Ok(v) => v,
            Err(e) => {
                wlog!(
                    "Failed to fetch optional repo akashmjn/tinydiarize-whisper.cpp: {:#}",
                    e
                );
                Vec::new()
            }
        };

    v1.append(&mut v2);

    // Deduplicate by name preferring ggerganov repo if duplicates
    let mut map: BTreeMap<String, ModelEntry> = BTreeMap::new();
    for m in v1 {
        map.entry(m.name.clone())
            .and_modify(|existing| {
                if existing.repo != "ggerganov/whisper.cpp" && m.repo == "ggerganov/whisper.cpp" {
                    *existing = m.clone();
                }
            })
            .or_insert(m);
    }

    let mut list: Vec<ModelEntry> = map.into_values().collect();
    list.sort_by(|a, b| {
        a.base
            .cmp(&b.base)
            .then(a.subtype.cmp(&b.subtype))
            .then(a.name.cmp(&b.name))
    });
    Ok(list)
}

fn format_model_list(models: &[ModelEntry]) -> String {
    let mut out = String::new();
    out.push_str("Available ggml Whisper models:\n");

    // Compute alignment widths
    let idx_width = std::cmp::max(2, models.len().to_string().len());
    let name_width = models.iter().map(|m| m.name.len()).max().unwrap_or(0);

    let mut idx = 1usize;
    let mut current = String::new();
    for m in models.iter() {
        if m.base != current {
            current = m.base.clone();
            out.push('\n');
            out.push_str(&format!("{current}:\n"));
        }
        // Format without hash and with aligned columns
        out.push_str(&format!(
            "  {i:>iw$}) {name:<nw$} [{repo} | {size}]\n",
            i = idx,
            iw = idx_width,
            name = m.name,
            nw = name_width,
            repo = m.repo,
            size = human_size(m.size),
        ));
        idx += 1;
    }
    out.push_str(
        "\nEnter selection by indices (e.g., 1 3 5-7), or 'all', '*' for all, 'q' to cancel.\n",
    );
    out
}

fn prompt_select_models_two_stage(models: &[ModelEntry]) -> Result<Vec<ModelEntry>> {
    // Non-interactive: pick a sensible default or exit cleanly
    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        // Prefer the default English base model (e.g., "base.en")
        if let Some(default) = models.iter().find(|m| m.base == "base" && m.subtype == "en") {
            ilog!("Non-Interactive: selecting default model {}", default.name);
            return Ok(vec![default.clone()]);
        }
        // Fallback: any 'base' family model
        if let Some(fallback) = models.iter().find(|m| m.base == "base") {
            ilog!("Non-Interactive: selecting default model {}", fallback.name);
            return Ok(vec![fallback.clone()]);
        }
        // Nothing sensible to pick
        wlog!("No interactive selection possible and no default model found; skipping model selection.");
        return Ok(Vec::new());
    }

    // Know Whisper base families in preferred ordering
    let mut known_order: Vec<&str> = vec!["tiny", "small", "base", "medium", "large"];
    // Collect available bases from the incoming list
    use std::collections::{BTreeMap, BTreeSet};

    let mut bases_available: BTreeSet<String> = BTreeSet::new();
    let mut by_base: BTreeMap<String, Vec<(usize, &ModelEntry)>> = BTreeMap::new();
    for (i, m) in models.iter().enumerate() {
        bases_available.insert(m.base.clone());
        by_base.entry(m.base.clone()).or_default().push((i, m));
    }

    // Filter known_order by what is available; append any unknown bases at the end (sorted)
    let mut base_choices: Vec<String> = Vec::new();
    for base in &known_order {
        if bases_available.contains(*base) {
            base_choices.push((*base).to_string());
        }
    }
    for b in &bases_available {
        if !known_order.iter().any(|k| k == b) {
            base_choices.push(b.clone());
        }
    }
    if base_choices.is_empty() {
        wlog!("No models available to select from.");
        return Ok(Vec::new());
    }

    // Build select items for bases
    let base_prompt = "Choose a base model family";
    let base_items: Vec<(String, String, String)> = base_choices
        .iter()
        .map(|b| {
            let count = by_base.get(b).map(|v| v.len()).unwrap_or(0);
            let label = format!("{b} ({count} variants)");
            (b.clone(), label, String::new())
        })
        .collect();

    let selected_base = match cliclack::select::<String>(base_prompt).items(&base_items).interact() {
        Ok(val) => val,
        Err(e) => {
            wlog!("Selection canceled or failed: {}", e);
            return Ok(Vec::new());
        }
    };

    // Second stage: multiselect among the chosen base's variants
    let Some(variants) = by_base.get(&selected_base) else {
        wlog!("No variants found for base '{}'.", selected_base);
        return Ok(Vec::new());
    };

    // Sort variants by subtype then name for stable presentation
    let mut variants_sorted = variants.clone();
    variants_sorted.sort_by(|a, b| {
        let (_, ma) = a;
        let (_, mb) = b;
        ma.subtype.cmp(&mb.subtype)
            .then(ma.name.cmp(&mb.name))
            .then(ma.repo.cmp(&mb.repo))
    });

    // Build multiselect items where value is the original index into 'models'
    let name_width = variants_sorted.iter()
        .map(|(_, m)| m.name.len())
        .max()
        .unwrap_or(0);

    let prompt = format!(
        "select {base} variant(s) (↑/↓ move, space toggle, enter confirm)", base = selected_base
    );

    let mut items: Vec<(usize, String, String)> = Vec::with_capacity(variants_sorted.len());
    for (idx, m) in variants_sorted.iter() {
        let label = format!(
            "{name:<nw$} [{repo} | {size}]",
            name = m.name,
            nw = name_width,
            repo = m.repo,
            size = human_size(m.size)
        );
        items.push((*idx, label, String::new()));
    }

    match cliclack::multiselect::<usize>(&prompt).items(&items).interact() {
        Ok(selected_indices) => {
            if selected_indices.is_empty() {
                ilog!("No variants selected; nothing to download.");
                return Ok(Vec::new());
            }
            let mut chosen: Vec<ModelEntry> = Vec::with_capacity(selected_indices.len());
            for mi in selected_indices {
                if let Some(m) = models.get(mi) {
                    chosen.push(m.clone());
                }
            }
            Ok(chosen)
        }
        Err(e) => {
            wlog!("Selection canceled or failed: {}", e);
            Ok(Vec::new())
        }
    }
}

fn compute_file_sha256_hex(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open for hashing: {}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 128];
    loop {
        let n = reader.read(&mut buf).context("Read error during hashing")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex_lower(&hasher.finalize()))
}

/// Interactively list and download Whisper models from Hugging Face into the models directory.
pub fn run_interactive_model_downloader() -> Result<()> {
    let models_dir_buf = crate::models_dir_path();
    let models_dir = models_dir_buf.as_path();
    if !models_dir.exists() {
        create_dir_all(models_dir).context("Failed to create models directory")?;
    }
    let client = Client::builder()
        .user_agent("PolyScribe/0.1 (+https://github.com/)")
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("Failed to build HTTP client")?;

    ilog!(
        "Fetching online data: contacting Hugging Face to retrieve available models (this may take a moment)..."
    );
    let models = fetch_all_models(&client)?;
    if models.is_empty() {
        qlog!("No models found on Hugging Face listing. Please try again later.");
        return Ok(());
    }
    let selected = prompt_select_models_two_stage(&models)?;
    if selected.is_empty() {
        qlog!("No selection. Aborting download.");
        return Ok(());
    }

    // Parallel downloads with bounded concurrency. Default 3; override via POLYSCRIBE_MAX_PARALLEL_DOWNLOADS (1..=6).
    let max_jobs = std::env::var("POLYSCRIBE_MAX_PARALLEL_DOWNLOADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.clamp(1, 6))
        .unwrap_or(3);

    // Use a MultiProgress to render per-model bars concurrently when interactive.
    let mp_opt = if !crate::is_quiet() && !crate::is_no_progress() && atty::is(Stream::Stderr) {
        Some(MultiProgress::new())
    } else {
        None
    };

    let mut i = 0;
    while i < selected.len() {
        let end = std::cmp::min(i + max_jobs, selected.len());
        let mut handles = Vec::new();
        for m in selected[i..end].iter().cloned() {
            let client2 = client.clone();
            let models_dir2 = models_dir.to_path_buf();
            let pb_opt = if let Some(mp) = &mp_opt {
                let pb = mp.add(ProgressBar::new(m.size));
                let style = ProgressStyle::with_template("Downloading {prefix} ({total_bytes}) [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%)")
                    .unwrap()
                    .progress_chars("=>-");
                pb.set_style(style);
                pb.set_prefix(format!("{}", m.name));
                Some(pb)
            } else { None };
            handles.push(std::thread::spawn(move || {
                if let Err(e) = download_one_model(&client2, &models_dir2, &m, pb_opt) {
                    crate::elog!("Error: {:#}", e);
                }
            }));
        }
        for h in handles { let _ = h.join(); }
        i = end;
    }

    // Drop MultiProgress after threads are joined; bars finish naturally.
    drop(mp_opt);

    Ok(())
}

/// Download a single model entry into the given models directory, verifying SHA-256 when available.
fn download_one_model(client: &Client, models_dir: &Path, entry: &ModelEntry, pb: Option<indicatif::ProgressBar>) -> Result<()> {
    let final_path = models_dir.join(format!("ggml-{}.bin", entry.name));

    // If the model already exists, verify against online metadata
    if final_path.exists() {
        if let Some(expected) = &entry.sha256 {
            match compute_file_sha256_hex(&final_path) {
                Ok(local_hash) => {
                    if local_hash.eq_ignore_ascii_case(expected) {
                        qlog!("Model {} is up-to-date (hash match).", final_path.display());
                        return Ok(());
                    } else {
                        qlog!(
                            "Local model {} hash differs from online (local {}.., online {}..). Updating...",
                            final_path.display(),
                            &local_hash[..std::cmp::min(8, local_hash.len())],
                            &expected[..std::cmp::min(8, expected.len())]
                        );
                    }
                }
                Err(e) => {
                    wlog!(
                        "Failed to hash existing {}: {}. Will re-download to ensure correctness.",
                        final_path.display(),
                        e
                    );
                }
            }
        } else if entry.size > 0 {
            match std::fs::metadata(&final_path) {
                Ok(md) => {
                    if md.len() == entry.size {
                        qlog!(
                            "Model {} appears up-to-date by size ({}).",
                            final_path.display(),
                            entry.size
                        );
                        return Ok(());
                    } else {
                        qlog!(
                            "Local model {} size ({}) differs from online ({}). Updating...",
                            final_path.display(),
                            md.len(),
                            entry.size
                        );
                    }
                }
                Err(e) => {
                    wlog!(
                        "Failed to stat existing {}: {}. Will re-download to ensure correctness.",
                        final_path.display(),
                        e
                    );
                }
            }
        } else {
            qlog!(
                "Model {} exists but remote hash/size not available; will download to verify contents.",
                final_path.display()
            );
            // Fall through to download/copy for content comparison
        }
    }

    // Offline/local copy mode for tests: if set, copy from a given base directory instead of HTTP
    if let Ok(base_dir) = env::var("POLYSCRIBE_MODELS_BASE_COPY_DIR") {
        let src_path = std::path::Path::new(&base_dir).join(format!("ggml-{}.bin", entry.name));
        if src_path.exists() {
            qlog!("Copying {} from {}...", entry.name, src_path.display());
            let tmp_path = models_dir.join(format!("ggml-{}.bin.part", entry.name));
            if tmp_path.exists() {
                let _ = std::fs::remove_file(&tmp_path);
            }
            std::fs::copy(&src_path, &tmp_path).with_context(|| {
                format!(
                    "Failed to copy from {} to {}",
                    src_path.display(),
                    tmp_path.display()
                )
            })?;
            // Verify hash if available
            if let Some(expected) = &entry.sha256 {
                let got = compute_file_sha256_hex(&tmp_path)?;
                if !got.eq_ignore_ascii_case(expected) {
                    let _ = std::fs::remove_file(&tmp_path);
                    return Err(anyhow!(
                        "SHA-256 mismatch for {} (copied): expected {}, got {}",
                        entry.name,
                        expected,
                        got
                    ));
                }
            }
            // Replace existing file safely
            if final_path.exists() {
                let _ = std::fs::remove_file(&final_path);
            }
            std::fs::rename(&tmp_path, &final_path)
                .with_context(|| format!("Failed to move into place: {}", final_path.display()))?;
            qlog!("Saved: {}", final_path.display());
            return Ok(());
        }
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/ggml-{}.bin",
        entry.repo, entry.name
    );
    qlog!(
        "Downloading {} ({} | {})...",
        entry.name,
        human_size(entry.size),
        url
    );
    let mut resp = client
        .get(url.clone())
        .send()
        .and_then(|r| r.error_for_status())
        .with_context(|| format!(
            "Failed to download model {} from {}. If your terminal has display/TTY issues, try running with --no-progress.",
            entry.name, url
        ))?;

    let tmp_path = models_dir.join(format!("ggml-{}.bin.part", entry.name));
    if tmp_path.exists() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    let mut file = std::io::BufWriter::new(
        File::create(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?,
    );

    // Set up progress bar: use provided one if present; otherwise create if interactive and we know size
    let show_progress = !crate::is_quiet() && !crate::is_no_progress() && atty::is(Stream::Stderr) && entry.size > 0;
    let pb_opt = if let Some(p) = pb {
        Some(p)
    } else if show_progress {
        let pb = ProgressBar::new(entry.size);
        let style = ProgressStyle::with_template("Downloading {prefix} ({total_bytes}) [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({percent}%)")
            .unwrap()
            .progress_chars("=>-");
        pb.set_style(style);
        pb.set_prefix(format!("{}", entry.name));
        Some(pb)
    } else { None };

    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut buf = [0u8; 1024 * 128];
    let mut read_err: Option<anyhow::Error> = None;
    loop {
        let nres = resp.read(&mut buf);
        match nres {
            Ok(n) => {
                if n == 0 { break; }
                hasher.update(&buf[..n]);
                if let Err(e) = file.write_all(&buf[..n]) { read_err = Some(anyhow!(e)); break; }
                downloaded += n as u64;
                if let Some(pb) = &pb_opt { pb.set_position(downloaded.min(entry.size)); }
            }
            Err(e) => { read_err = Some(anyhow!("Network read error: {}", e)); break; }
        }
    }
    file.flush().ok();

    if let Some(err) = read_err {
        if let Some(pb) = &pb_opt { pb.abandon_with_message("download failed"); }
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }

    let got = to_hex_lower(&hasher.finalize());
    if let Some(expected) = &entry.sha256 {
        if got != expected.to_lowercase() {
            if let Some(pb) = &pb_opt { pb.abandon_with_message("hash mismatch"); }
            let _ = std::fs::remove_file(&tmp_path);
            return Err(anyhow!(
                "SHA-256 mismatch for {}: expected {}, got {}",
                entry.name,
                expected,
                got
            ));
        }
    } else {
        wlog!(
            "No SHA-256 available for {}. Skipping verification.",
            entry.name
        );
    }
    // Replace existing file safely
    if final_path.exists() {
        let _ = std::fs::remove_file(&final_path);
    }
    std::fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("Failed to move into place: {}", final_path.display()))?;
    if let Some(pb) = &pb_opt { pb.finish_with_message("saved"); }
    qlog!("Saved: {}", final_path.display());
    Ok(())
}

// Update locally stored models by re-downloading when size or hash does not match online metadata.
fn qlog_size_comparison(fname: &str, local: u64, remote: u64) -> bool {
    if local == remote {
        qlog!("{} appears up-to-date by size ({}).", fname, remote);
        true
    } else {
        qlog!(
            "{} size {} differs from remote {}. Updating...",
            fname, local, remote
        );
        false
    }
}

/// Update locally stored models by re-downloading when size or hash does not match online metadata.
pub fn update_local_models() -> Result<()> {
    let models_dir_buf = crate::models_dir_path();
    let models_dir = models_dir_buf.as_path();
    if !models_dir.exists() {
        create_dir_all(models_dir).context("Failed to create models directory")?;
    }

    // Build HTTP client (may be unused in offline copy mode)
    let client = Client::builder()
        .user_agent("PolyScribe/0.1 (+https://github.com/)")
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("Failed to build HTTP client")?;

    // Obtain manifest: env override or online fetch
    let models: Vec<ModelEntry> = if let Ok(manifest_path) = env::var("POLYSCRIBE_MODELS_MANIFEST")
    {
        let data = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("Failed to read manifest at {manifest_path}"))?;
        let mut list: Vec<ModelEntry> = serde_json::from_str(&data)
            .with_context(|| format!("Invalid JSON manifest: {manifest_path}"))?;
        // sort for stability
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    } else {
        fetch_all_models(&client)?
    };

    // Map name -> entry for fast lookup
    let mut map: BTreeMap<String, ModelEntry> = BTreeMap::new();
    for m in models {
        map.insert(m.name.clone(), m);
    }

    // Scan local ggml-*.bin models
    let rd = std::fs::read_dir(models_dir)
        .with_context(|| format!("Failed to read models dir: {}", models_dir.display()))?;
    for entry in rd {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if !fname.starts_with("ggml-") || !fname.ends_with(".bin") {
            continue;
        }
        let model_name = fname
            .trim_start_matches("ggml-")
            .trim_end_matches(".bin")
            .to_string();

        if let Some(remote) = map.get(&model_name) {
            // If SHA256 available, verify and update if mismatch
            if let Some(expected) = &remote.sha256 {
                // Show a small spinner while verifying hash (TTY, not quiet, not no-progress)
                let show_spin = !crate::is_quiet() && !crate::is_no_progress() && atty::is(Stream::Stderr);
                let spinner = if show_spin {
                    let pb = ProgressBar::new_spinner();
                    pb.enable_steady_tick(std::time::Duration::from_millis(100));
                    pb.set_message(format!("Verifying {}", fname));
                    Some(pb)
                } else { None };
                let verify_res = compute_file_sha256_hex(&path);
                if let Some(pb) = &spinner { pb.finish_and_clear(); }
                match verify_res {
                    Ok(local_hash) => {
                        if local_hash.eq_ignore_ascii_case(expected) {
                            qlog!("{} is up-to-date.", fname);
                            continue;
                        } else {
                            qlog!(
                                "{} hash differs (local {}.. != remote {}..). Updating...",
                                fname,
                                &local_hash[..std::cmp::min(8, local_hash.len())],
                                &expected[..std::cmp::min(8, expected.len())]
                            );
                        }
                    }
                    Err(e) => {
                        wlog!("Failed hashing {}: {}. Re-downloading.", fname, e);
                    }
                }
                download_one_model(&client, models_dir, remote, None)?;
            } else if remote.size > 0 {
                match std::fs::metadata(&path) {
                    Ok(md) => {
                        if qlog_size_comparison(&fname, md.len(), remote.size) {
                            continue;
                        }
                        download_one_model(&client, models_dir, remote, None)?;
                    }
                    Err(e) => {
                        wlog!("Stat failed for {}: {}. Updating...", fname, e);
                        download_one_model(&client, models_dir, remote, None)?;
                    }
                }
            } else {
                qlog!("No remote hash/size for {}. Skipping.", fname);
            }
        } else {
            qlog!("No remote metadata for {}. Skipping.", fname);
        }
    }

    Ok(())
}

/// Pick the best local ggml-*.bin model: largest by file size; tie-break by lexicographic filename.
pub fn pick_best_local_model(models_dir: &Path) -> Option<std::path::PathBuf> {
    let mut best: Option<(u64, String, std::path::PathBuf)> = None;
    let rd = std::fs::read_dir(models_dir).ok()?;
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if !fname.starts_with("ggml-") || !fname.ends_with(".bin") {
            continue;
        }
        let size = std::fs::metadata(&path).ok()?.len();
        match &mut best {
            None => best = Some((size, fname, path.clone())),
            Some((bsize, bname, bpath)) => {
                if size > *bsize || (size == *bsize && fname < *bname) {
                    *bsize = size;
                    *bname = fname;
                    *bpath = path.clone();
                }
            }
        }
    }
    best.map(|(_, _, p)| p)
}

/// Ensure a specific model is available locally without any interactive prompts.
/// If found locally, returns its path. Otherwise downloads it and returns the path.
pub fn ensure_model_available_noninteractive(model_name: &str) -> Result<std::path::PathBuf> {
    let models_dir_buf = crate::models_dir_path();
    let models_dir = models_dir_buf.as_path();
    if !models_dir.exists() {
        create_dir_all(models_dir).context("Failed to create models directory")?;
    }
    let final_path = models_dir.join(format!("ggml-{model_name}.bin"));
    if final_path.exists() {
        return Ok(final_path);
    }

    let client = Client::builder()
        .user_agent("PolyScribe/0.1 (+https://github.com/)")
        .timeout(Duration::from_secs(600))
        .redirect(Policy::limited(10))
        .build()
        .context("Failed to build HTTP client")?;

    // Prefer fetching metadata to construct a proper ModelEntry
    let models = fetch_all_models(&client)?;
    if let Some(entry) = models.into_iter().find(|m| m.name == model_name) {
        download_one_model(&client, models_dir, &entry, None)?;
        return Ok(models_dir.join(format!("ggml-{}.bin", entry.name)));
    }
    Err(anyhow!(
        "Model '{}' not found in remote listings; cannot download non-interactively.",
        model_name
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_format_model_list_spacing_and_structure() {
        let models = vec![
            ModelEntry {
                name: "tiny.en-q5_1".to_string(),
                base: "tiny".to_string(),
                subtype: "en-q5_1".to_string(),
                size: 1024 * 1024,
                sha256: Some(
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
                ),
                repo: "ggerganov/whisper.cpp".to_string(),
            },
            ModelEntry {
                name: "tiny-q5_1".to_string(),
                base: "tiny".to_string(),
                subtype: "q5_1".to_string(),
                size: 2048,
                sha256: None,
                repo: "ggerganov/whisper.cpp".to_string(),
            },
            ModelEntry {
                name: "base.en-q5_1".to_string(),
                base: "base".to_string(),
                subtype: "en-q5_1".to_string(),
                size: 10,
                sha256: Some(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
                ),
                repo: "akashmjn/tinydiarize-whisper.cpp".to_string(),
            },
        ];
        let s = format_model_list(&models);
        // Header present
        assert!(s.starts_with("Available ggml Whisper models:\n"));
        // Group headers and blank line before header
        assert!(s.contains("\ntiny:\n"));
        assert!(s.contains("\nbase:\n"));
        // No immediate double space before a bracket after parenthesis
        assert!(
            !s.contains(")  ["),
            "should not have double space immediately before bracket"
        );
        // Lines contain normalized spacing around pipes and no hash
        assert!(s.contains("[ggerganov/whisper.cpp | 1.00 MiB]"));
        assert!(s.contains("[ggerganov/whisper.cpp | 2.00 KiB]"));
        // Verify alignment: the '[' position should match across multiple lines
        let bracket_positions: Vec<usize> = s
            .lines()
            .filter(|l| l.contains("ggerganov/whisper.cpp"))
            .map(|l| l.find('[').unwrap())
            .collect();
        assert!(bracket_positions.len() >= 2);
        for w in bracket_positions.windows(2) {
            assert_eq!(w[0], w[1], "bracket columns should align");
        }
        // Footer instruction present
        assert!(s.contains("Enter selection by indices"));
    }

    #[test]
    fn test_format_model_list_unaffected_by_quiet_flag() {
        let models = vec![
            ModelEntry {
                name: "tiny.en-q5_1".to_string(),
                base: "tiny".to_string(),
                subtype: "en-q5_1".to_string(),
                size: 1024,
                sha256: None,
                repo: "ggerganov/whisper.cpp".to_string(),
            },
            ModelEntry {
                name: "base.en-q5_1".to_string(),
                base: "base".to_string(),
                subtype: "en-q5_1".to_string(),
                size: 2048,
                sha256: None,
                repo: "ggerganov/whisper.cpp".to_string(),
            },
        ];
        // Compute with quiet off and on; the pure formatter should not depend on quiet.
        crate::set_quiet(false);
        let a = format_model_list(&models);
        crate::set_quiet(true);
        let b = format_model_list(&models);
        assert_eq!(a, b);
        // reset quiet for other tests
        crate::set_quiet(false);
    }

    fn sha256_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let out = hasher.finalize();
        let mut s = String::new();
        for b in out {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    #[test]
    fn test_update_local_models_offline_copy_and_manifest() {
        use std::sync::{Mutex, OnceLock};
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();

        let tmp_models = tempdir().unwrap();
        let tmp_base = tempdir().unwrap();
        let tmp_manifest = tempdir().unwrap();

        // Prepare source model file content and hash
        let model_name = "tiny.en-q5_1";
        let src_path = tmp_base.path().join(format!("ggml-{}.bin", model_name));
        let new_content = b"new model content";
        fs::write(&src_path, new_content).unwrap();
        let expected_sha = sha256_hex(new_content);
        let expected_size = new_content.len() as u64;

        // Write a wrong existing local file to trigger update
        let local_path = tmp_models.path().join(format!("ggml-{}.bin", model_name));
        fs::write(&local_path, b"old content").unwrap();

        // Write manifest JSON
        let manifest_path = tmp_manifest.path().join("manifest.json");
        let manifest = serde_json::json!([
            {
                "name": model_name,
                "base": "tiny",
                "subtype": "en-q5_1",
                "size": expected_size,
                "sha256": expected_sha,
                "repo": "ggerganov/whisper.cpp"
            }
        ]);
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        // Set env vars to force offline behavior and directories
        unsafe {
            std::env::set_var("POLYSCRIBE_MODELS_MANIFEST", &manifest_path);
            std::env::set_var("POLYSCRIBE_MODELS_BASE_COPY_DIR", tmp_base.path());
            std::env::set_var("POLYSCRIBE_MODELS_DIR", tmp_models.path());
        }

        // Run update
        update_local_models().unwrap();

        // Verify local file equals source content
        let got = fs::read(&local_path).unwrap();
        assert_eq!(got, new_content);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_models_dir_path_default_debug_and_env_override_models_mod() {
        // clear override
        unsafe {
            std::env::remove_var("POLYSCRIBE_MODELS_DIR");
        }
        assert_eq!(crate::models_dir_path(), std::path::PathBuf::from("models"));
        // override
        let tmp = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("POLYSCRIBE_MODELS_DIR", tmp.path());
        }
        assert_eq!(crate::models_dir_path(), tmp.path().to_path_buf());
        // cleanup
        unsafe {
            std::env::remove_var("POLYSCRIBE_MODELS_DIR");
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_models_dir_path_default_release_models_mod() {
        unsafe {
            std::env::remove_var("POLYSCRIBE_MODELS_DIR");
        }
        // With XDG_DATA_HOME set
        let tmp_xdg = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("XDG_DATA_HOME", tmp_xdg.path());
            std::env::remove_var("HOME");
        }
        assert_eq!(
            crate::models_dir_path(),
            std::path::PathBuf::from(tmp_xdg.path())
                .join("polyscribe")
                .join("models")
        );
        // With HOME fallback
        let tmp_home = tempfile::tempdir().unwrap();
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
            std::env::set_var("HOME", tmp_home.path());
        }
        assert_eq!(
            super::models_dir_path(),
            std::path::PathBuf::from(tmp_home.path())
                .join(".local")
                .join("share")
                .join("polyscribe")
                .join("models")
        );
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
            std::env::remove_var("HOME");
        }
    }
}
