// SPDX-License-Identifier: MIT

use crate::config::ConfigService;
use crate::prelude::*;
use anyhow::{Context, anyhow};
use chrono::{DateTime, Utc};
use hex::ToHex;
use reqwest::blocking::Client;
use reqwest::header::{
    ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, ETAG, IF_RANGE, LAST_MODIFIED, RANGE,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn format_size_mb(size: Option<u64>) -> String {
    match size {
        Some(bytes) => {
            let mib = bytes as f64 / 1024.0 / 1024.0;
            format!("{mib:.2} MiB")
        }
        None => "? MiB".to_string(),
    }
}

fn format_size_gib(bytes: u64) -> String {
    let gib = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    format!("{gib:.2} GiB")
}

fn short_date(s: &str) -> String {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc).format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| s.to_string())
}

fn free_space_bytes_for_path(path: &Path) -> Result<u64> {
    use libc::statvfs;
    use std::ffi::CString;

    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };

    let cpath = CString::new(dir.as_os_str().to_string_lossy().as_bytes())
        .map_err(|_| anyhow!("invalid path for statvfs"))?;
    unsafe {
        let mut s: libc::statvfs = std::mem::zeroed();
        if statvfs(cpath.as_ptr(), &mut s) != 0 {
            return Err(anyhow!("statvfs failed for {}", dir.display()).into());
        }
        Ok((s.f_bsize as u64) * (s.f_bavail as u64))
    }
}

fn mirror_label(url: &str) -> &'static str {
    if url.contains("eu") {
        "EU mirror"
    } else if url.contains("us") {
        "US mirror"
    } else {
        "source"
    }
}

type HeadMeta = (Option<u64>, Option<String>, Option<String>, bool);

fn head_entry(client: &Client, url: &str) -> Result<HeadMeta> {
    let resp = client.head(url).send()?.error_for_status()?;
    let len = resp
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    let etag = resp
        .headers()
        .get(ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string());
    let last_mod = resp
        .headers()
        .get(LAST_MODIFIED)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let ranges_ok = resp
        .headers()
        .get(ACCEPT_RANGES)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_ascii_lowercase().contains("bytes"))
        .unwrap_or(false);
    Ok((len, etag, last_mod, ranges_ok))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelEntry {
    name: String,
    file: String,
    url: String,
    size: Option<u64>,
    sha256: Option<String>,
    last_modified: Option<String>,
    base: String,
    variant: String,
}


#[derive(Debug, Deserialize)]
struct HfModelInfo {
    siblings: Option<Vec<HfFile>>,
    files: Option<Vec<HfFile>>,
}

#[derive(Debug, Deserialize)]
struct HfLfsInfo {
    oid: Option<String>,
    size: Option<u64>,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfFile {
    rfilename: String,
    size: Option<u64>,
    sha256: Option<String>,
    lfs: Option<HfLfsInfo>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
}

fn parse_base_variant(display_name: &str) -> (String, String) {
    let mut variant = "default".to_string();
    let mut head = display_name;
    if let Some((h, rest)) = display_name.split_once('.') {
        head = h;
        variant = rest.to_string();
    }
    if let Some((b, v)) = head.split_once('-') {
        return (b.to_string(), v.to_string());
    }
    (head.to_string(), variant)
}

fn hf_repo_manifest_api(repo: &str) -> Result<Vec<ModelEntry>> {
    let client = Client::builder()
        .user_agent(ConfigService::user_agent())
        .build()?;

    let base = ConfigService::hf_api_base_for(repo);
    let resp = client.get(&base).send()?;
    let mut entries = if resp.status().is_success() {
        let info: HfModelInfo = resp.json()?;
        hf_info_to_entries(repo, info)?
    } else {
        Vec::new()
    };

    if entries.is_empty() {
        let url = format!("{base}?expand=files");
        let resp2 = client.get(&url).send()?;
        if !resp2.status().is_success() {
            return Err(anyhow!("HF API {} for {}", resp2.status(), url).into());
        }
        let info: HfModelInfo = resp2.json()?;
        entries = hf_info_to_entries(repo, info)?;
    }

    if entries.is_empty() {
        return Err(anyhow!("HF API returned no usable .bin files").into());
    }
    Ok(entries)
}

fn hf_info_to_entries(repo: &str, info: HfModelInfo) -> Result<Vec<ModelEntry>> {
    let files = info.files.or(info.siblings).unwrap_or_default();
    let mut out = Vec::new();
    for f in files {
        let fname = f.rfilename;
        if !fname.ends_with(".bin") {
            continue;
        }

        let stem = fname.strip_suffix(".bin").unwrap_or(&fname).to_string();
        let name_no_prefix = stem
            .strip_prefix("ggml-")
            .or_else(|| stem.strip_prefix("gguf-"))
            .unwrap_or(&stem)
            .to_string();

        let sha_from_lfs = f.lfs.as_ref().and_then(|l| {
            l.sha256.clone().or_else(|| {
                l.oid
                    .as_ref()
                    .and_then(|oid| oid.strip_prefix("sha256:").map(|s| s.to_string()))
            })
        });

        let size = f.size.or_else(|| f.lfs.as_ref().and_then(|l| l.size));

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}?download=true",
            repo, fname
        );

        let (base, variant) = parse_base_variant(&name_no_prefix);

        out.push(ModelEntry {
            name: name_no_prefix,
            file: fname,
            url,
            size,
            sha256: sha_from_lfs.or(f.sha256),
            last_modified: f.last_modified.clone(),
            base,
            variant,
        });
    }
    Ok(out)
}


fn scrape_tree_manifest(repo: &str) -> Result<Vec<ModelEntry>> {
    let client = Client::builder()
        .user_agent(ConfigService::user_agent())
        .build()?;

    let url = format!("https://huggingface.co/{}/tree/main?recursive=1", repo);
    let resp = client.get(&url).send()?;
    if !resp.status().is_success() {
        return Err(anyhow!("tree page HTTP {} for {}", resp.status(), url).into());
    }
    let html = resp.text()?;

    let mut files = BTreeSet::new();
    for mat in html.match_indices(".bin") {
        let end = mat.0 + 4;
        let start = html[..end]
            .rfind("/blob/main/")
            .or_else(|| html[..end].rfind("/resolve/main/"));
        if let Some(s) = start {
            let slice = &html[s..end];
            if let Some(pos) = slice.find("/blob/main/") {
                let path = slice[pos + "/blob/main/".len()..].to_string();
                files.insert(path);
            } else if let Some(pos) = slice.find("/resolve/main/") {
                let path = slice[pos + "/resolve/main/".len()..].to_string();
                files.insert(path);
            }
        }
    }

    let mut out = Vec::new();
    for fname in files.into_iter().filter(|f| f.ends_with(".bin")) {
        let stem = fname.strip_suffix(".bin").unwrap_or(&fname).to_string();
        let name = stem
            .rsplit('/')
            .next()
            .unwrap_or(&stem)
            .strip_prefix("ggml-")
            .or_else(|| {
                stem.rsplit('/')
                    .next()
                    .unwrap_or(&stem)
                    .strip_prefix("gguf-")
            })
            .unwrap_or_else(|| stem.rsplit('/').next().unwrap_or(&stem))
            .to_string();

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}?download=true",
            repo, fname
        );

        let (base, variant) = parse_base_variant(&name);

        out.push(ModelEntry {
            name,
            file: fname.rsplit('/').next().unwrap_or(&fname).to_string(),
            url,
            size: None,
            sha256: None,
            last_modified: None,
            base,
            variant,
        });
    }

    if out.is_empty() {
        return Err(anyhow!("tree scraper found no .bin files").into());
    }
    Ok(out)
}


fn parse_sha_from_header_value(s: &str) -> Option<String> {
    let lower = s.to_ascii_lowercase();
    if let Some(idx) = lower.find("sha256:") {
        let tail = &lower[idx + "sha256:".len()..];
        let hex: String = tail.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        if !hex.is_empty() {
            return Some(hex);
        }
    }
    None
}

fn enrich_entry_via_head(entry: &mut ModelEntry) -> Result<()> {
    if entry.size.is_some() && entry.sha256.is_some() && entry.last_modified.is_some() {
        return Ok(());
    }

    let client = Client::builder()
        .user_agent(ConfigService::user_agent())
        .timeout(Duration::from_secs(ConfigService::http_timeout_secs()))
        .build()?;

    let mut head_url = entry.url.clone();
    if let Some((base, _)) = entry.url.split_once('?') {
        head_url = base.to_string();
    }

    let started = Instant::now();
    crate::dlog!(1, "HEAD {}", head_url);

    let resp = client
        .head(&head_url)
        .send()
        .or_else(|_| client.head(&entry.url).send())?;

    if !resp.status().is_success() {
        crate::dlog!(1, "HEAD {} -> HTTP {}", head_url, resp.status());
        return Ok(());
    }

    let mut filled_size = false;
    let mut filled_sha = false;
    let mut filled_lm = false;

    if entry.size.is_none()
        && let Some(sz) = resp
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
    {
        entry.size = Some(sz);
        filled_size = true;
    }

    if entry.sha256.is_none() {
        let _ = resp
            .headers()
            .get("x-linked-etag")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_sha_from_header_value)
            .map(|hex| {
                entry.sha256 = Some(hex);
                filled_sha = true;
            });
        if !filled_sha {
            let _ = resp
                .headers()
                .get(ETAG)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_sha_from_header_value)
                .map(|hex| {
                    entry.sha256 = Some(hex);
                    filled_sha = true;
                });
        }
    }

    if entry.last_modified.is_none() {
        let _ = resp
            .headers()
            .get(LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .map(|v| {
                entry.last_modified = Some(v.to_string());
                filled_lm = true;
            });
    }

    let elapsed_ms = started.elapsed().as_millis();
    crate::dlog!(
        1,
        "HEAD ok in {} ms for {} (size: {}, sha256: {}, last-modified: {})",
        elapsed_ms,
        entry.file,
        if filled_size {
            "new"
        } else if entry.size.is_some() {
            "kept"
        } else {
            "missing"
        },
        if filled_sha {
            "new"
        } else if entry.sha256.is_some() {
            "kept"
        } else {
            "missing"
        },
        if filled_lm {
            "new"
        } else if entry.last_modified.is_some() {
            "kept"
        } else {
            "missing"
        },
    );

    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedManifest {
    fetched_at: u64,
    etag: Option<String>,
    last_modified: Option<String>,
    entries: Vec<ModelEntry>,
}

fn get_cache_dir() -> Result<PathBuf> {
    Ok(ConfigService::manifest_cache_dir()
        .ok_or_else(|| anyhow!("could not determine platform directories"))?)
}

fn get_cached_manifest_path() -> Result<PathBuf> {
    let cache_dir = get_cache_dir()?;
    Ok(cache_dir.join(ConfigService::manifest_cache_filename()))
}

fn should_bypass_cache() -> bool {
    ConfigService::bypass_manifest_cache()
}

fn get_cache_ttl() -> u64 {
    ConfigService::manifest_cache_ttl_seconds()
}

fn load_cached_manifest() -> Option<CachedManifest> {
    if should_bypass_cache() {
        return None;
    }

    let cache_path = get_cached_manifest_path().ok()?;
    if !cache_path.exists() {
        return None;
    }

    let cache_file = File::open(cache_path).ok()?;
    let cached: CachedManifest = serde_json::from_reader(cache_file).ok()?;

    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();

    let ttl = get_cache_ttl();
    if now.saturating_sub(cached.fetched_at) > ttl {
        crate::dlog!(
            1,
            "Cache expired (age: {}s, TTL: {}s)",
            now.saturating_sub(cached.fetched_at),
            ttl
        );
        return None;
    }

    crate::dlog!(
        1,
        "Using cached manifest (age: {}s)",
        now.saturating_sub(cached.fetched_at)
    );
    Some(cached)
}

fn save_manifest_to_cache(
    entries: &[ModelEntry],
    etag: Option<&str>,
    last_modified: Option<&str>,
) -> Result<()> {
    if should_bypass_cache() {
        return Ok(());
    }

    let cache_dir = get_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;

    let cache_path = get_cached_manifest_path()?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow!("system time error"))?
        .as_secs();

    let cached = CachedManifest {
        fetched_at: now,
        etag: etag.map(|s| s.to_string()),
        last_modified: last_modified.map(|s| s.to_string()),
        entries: entries.to_vec(),
    };

    let cache_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&cache_path)
        .with_context(|| format!("opening cache file {}", cache_path.display()))?;

    serde_json::to_writer_pretty(cache_file, &cached)
        .with_context(|| "serializing cached manifest")?;

    crate::dlog!(1, "Saved manifest to cache: {} entries", entries.len());
    Ok(())
}

fn fetch_manifest_with_cache() -> Result<Vec<ModelEntry>> {
    let cached = load_cached_manifest();

    let client = Client::builder()
        .user_agent(ConfigService::user_agent())
        .build()?;
    let repo = ConfigService::hf_repo();
    let base_url = ConfigService::hf_api_base_for(&repo);

    let mut req = client.get(&base_url);
    if let Some(ref cached) = cached {
        if let Some(ref etag) = cached.etag {
            req = req.header("If-None-Match", format!("\"{}\"", etag));
        } else if let Some(ref last_mod) = cached.last_modified {
            req = req.header("If-Modified-Since", last_mod);
        }
    }

    let resp = req.send()?;

    if resp.status().as_u16() == 304 {
        if let Some(cached) = cached {
            crate::dlog!(1, "Manifest not modified, using cache");
            return Ok(cached.entries);
        }
    }

    if !resp.status().is_success() {
        return Err(anyhow!("HF API {} for {}", resp.status(), base_url).into());
    }

    let etag = resp
        .headers()
        .get(ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').to_string());

    let last_modified = resp
        .headers()
        .get(LAST_MODIFIED)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let info: HfModelInfo = resp.json()?;
    let mut entries = hf_info_to_entries(&repo, info)?;

    if entries.is_empty() {
        let url = format!("{}?expand=files", base_url);
        let resp2 = client.get(&url).send()?;
        if !resp2.status().is_success() {
            return Err(anyhow!("HF API {} for {}", resp2.status(), url).into());
        }
        let info: HfModelInfo = resp2.json()?;
        entries = hf_info_to_entries(&repo, info)?;
    }

    if entries.is_empty() {
        return Err(anyhow!("HF API returned no usable .bin files").into());
    }

    let _ = save_manifest_to_cache(&entries, etag.as_deref(), last_modified.as_deref());

    Ok(entries)
}

fn current_manifest() -> Result<Vec<ModelEntry>> {
    let started = Instant::now();
    crate::dlog!(1, "Fetching HF manifest…");

    let mut list = match fetch_manifest_with_cache() {
        Ok(list) if !list.is_empty() => {
            crate::dlog!(
                1,
                "Manifest loaded from HF API with cache ({} entries)",
                list.len()
            );
            list
        }
        _ => {
            crate::ilog!("Cache failed, falling back to direct API");
            let repo = ConfigService::hf_repo();
            let list = match hf_repo_manifest_api(&repo) {
                Ok(list) if !list.is_empty() => {
                    crate::dlog!(1, "Manifest loaded from HF API ({} entries)", list.len());
                    list
                }
                _ => {
                    crate::ilog!("Falling back to scraping the repository tree page");
                    let scraped = scrape_tree_manifest(&repo)?;
                    crate::dlog!(1, "Manifest loaded via scrape ({} entries)", scraped.len());
                    scraped
                }
            };

            let _ = save_manifest_to_cache(&list, None, None);
            list
        }
    };

    let mut need_enrich = 0usize;
    for m in &list {
        if m.size.is_none() || m.sha256.is_none() || m.last_modified.is_none() {
            need_enrich += 1;
        }
    }
    crate::dlog!(1, "Enriching {} entries via HEAD…", need_enrich);
    for m in &mut list {
        if m.size.is_none() || m.sha256.is_none() || m.last_modified.is_none() {
            let _ = enrich_entry_via_head(m);
        }
    }

    let elapsed_ms = started.elapsed().as_millis();
    let sizes_known = list.iter().filter(|m| m.size.is_some()).count();
    let hashes_known = list.iter().filter(|m| m.sha256.is_some()).count();
    crate::dlog!(
        1,
        "Manifest ready in {} ms (entries: {}, sizes: {}/{}, hashes: {}/{})",
        elapsed_ms,
        list.len(),
        sizes_known,
        list.len(),
        hashes_known,
        list.len()
    );

    if list.is_empty() {
        return Err(anyhow!("no usable .bin files discovered").into());
    }
    Ok(list)
}

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

fn resolve_models_dir() -> Result<PathBuf> {
    Ok(ConfigService::models_dir(None)
        .ok_or_else(|| anyhow!("could not determine models directory"))?)
}

pub fn ensure_model_available_noninteractive(name: &str) -> Result<PathBuf> {
    let entry = find_manifest_entry(name)?.ok_or_else(|| anyhow!("unknown model: {name}"))?;

    let dir = resolve_models_dir()?;
    fs::create_dir_all(&dir).ok();
    let dest = dir.join(&entry.file);

    if file_matches(&dest, entry.size, entry.sha256.as_deref())? {
        crate::ui::info(format!("Already up to date: {}", dest.display()));
        return Ok(dest);
    }

    let base = &entry.base;
    let variant = &entry.variant;
    let size_str = format_size_mb(entry.size);
    crate::ui::println_above_bars(format!("Base: {base} • Type: {variant}"));
    crate::ui::println_above_bars(format!(
        "Source: {} • Size: {}",
        mirror_label(&entry.url),
        size_str
    ));

    download_with_progress(&dest, &entry)?;
    Ok(dest)
}

pub fn clear_manifest_cache() -> Result<()> {
    let cache_path = get_cached_manifest_path()?;
    if cache_path.exists() {
        fs::remove_file(&cache_path)?;
        crate::dlog!(1, "Cleared manifest cache");
    }
    Ok(())
}

fn find_manifest_entry(name: &str) -> Result<Option<ModelEntry>> {
    let wanted_name = name
        .strip_suffix(".bin")
        .unwrap_or(name)
        .to_ascii_lowercase();
    let wanted_file = name.to_ascii_lowercase();

    for e in current_manifest()? {
        let file_lc = e.file.to_ascii_lowercase();
        let stem_lc = e
            .file
            .strip_suffix(".bin")
            .unwrap_or(&e.file)
            .to_ascii_lowercase();
        if e.name.to_ascii_lowercase() == wanted_name
            || file_lc == wanted_file
            || stem_lc == wanted_name
        {
            return Ok(Some(e));
        }
    }
    Ok(None)
}

fn file_matches(path: &Path, size: Option<u64>, sha256_hex: Option<&str>) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    if let Some(exp_hash) = sha256_hex {
        let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let actual = hasher.finalize();
        let actual_hex = actual.encode_hex::<String>();
        return Ok(actual_hex.eq_ignore_ascii_case(exp_hash));
    }

    if let Some(expected) = size {
        let meta = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
        return Ok(meta.len() == expected);
    }

    Ok(false)
}

fn download_with_progress(dest_path: &Path, entry: &ModelEntry) -> Result<()> {
    let url = &entry.url;
    let client = Client::builder()
        .user_agent(ConfigService::downloader_user_agent())
        .build()?;

    crate::ui::info(format!("Resolving source: {} ({})", mirror_label(url), url));

    let (mut total_len, remote_etag, _remote_last_mod, ranges_ok) =
        head_entry(&client, url).context("probing remote file")?;

    if total_len.is_none() {
        total_len = entry.size;
    }

    if let Some(expected) = total_len {
        let free = free_space_bytes_for_path(dest_path)?;
        let need = expected + (expected / 10) + 16 * 1024 * 1024;
        if free < need {
            return Err(anyhow!(
                "insufficient disk space: need {}, have {}",
                format_size_mb(Some(need)),
                format_size_gib(free)
            )
            .into());
        }
    }

    if dest_path.exists() && file_matches(dest_path, total_len, entry.sha256.as_deref())? {
        crate::ui::info(format!("Already up to date: {}", dest_path.display()));
        return Ok(());
    }

    let part_path = dest_path.with_extension("part");
    // Guard to cleanup .part on errors
    struct TempGuard { path: std::path::PathBuf, armed: bool }
    impl TempGuard { fn disarm(&mut self) { self.armed = false; } }
    impl Drop for TempGuard {
        fn drop(&mut self) { if self.armed { let _ = fs::remove_file(&self.path); } }
    }
    let mut _tmp_guard = TempGuard { path: part_path.clone(), armed: true };

    let mut resume_from: u64 = 0;
    if part_path.exists() && ranges_ok {
        let meta = fs::metadata(&part_path)?;
        resume_from = meta.len();
    }

    let mut part_file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&part_path)
        .with_context(|| format!("opening {}", part_path.display()))?;

    let mut req = client.get(url);
    if ranges_ok && resume_from > 0 {
        req = req.header(RANGE, format!("bytes={resume_from}-"));
        if let Some(etag) = &remote_etag {
            req = req.header(IF_RANGE, format!("\"{etag}\""));
        }
    }

    crate::ui::info(format!("Download: {}", part_path.display()));

    let pb_total = total_len.unwrap_or(0);
    let mut bar = crate::ui::BytesProgress::start(pb_total, "Downloading", resume_from);

    let start = Instant::now();
    let mut resp = req.send()?.error_for_status()?;

    if resp.status().as_u16() == 304 && resume_from == 0 {
        let req2 = client.get(url);
        resp = req2.send()?.error_for_status()?;
    }

    let is_partial_response = resp.headers().get(CONTENT_RANGE).is_some();
    if resume_from > 0 && !is_partial_response {
        drop(part_file);
        fs::remove_file(&part_path).ok();

        let req2 = client.get(url);
        resp = req2.send()?.error_for_status()?;
        bar.stop("restarting");
        bar = crate::ui::BytesProgress::start(pb_total, "Downloading", 0);

        part_file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&part_path)
            .with_context(|| format!("opening {}", part_path.display()))?;
    }

    {
        let mut body = resp;
        let mut buf = vec![0u8; 1024 * 64];
        loop {
            let read = body.read(&mut buf)?;
            if read == 0 {
                break;
            }
            part_file.write_all(&buf[..read])?;
            bar.inc(read as u64);
        }
        part_file.flush()?;
        part_file.sync_all()?;
    }

    bar.stop("done");

    if let Some(expected_hex) = entry.sha256.as_deref() {
        crate::ui::info("Verify: SHA-256");
        let mut f = File::open(&part_path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let actual_hex = hasher.finalize().encode_hex::<String>();
        if !actual_hex.eq_ignore_ascii_case(expected_hex) {
            return Err(anyhow!(
                "checksum mismatch: expected {}, got {}",
                expected_hex,
                actual_hex
            )
            .into());
        }
    } else {
        crate::ui::info("Verify: checksum not provided by source (skipped)");
    }

    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    drop(part_file);
    fs::rename(&part_path, dest_path)
        .with_context(|| format!("renaming {} → {}", part_path.display(), dest_path.display()))?;
    _tmp_guard.disarm();

    let final_size = fs::metadata(dest_path).map(|m| m.len()).ok();
    let elapsed = start.elapsed().as_secs_f64();

    if let Some(sz) = final_size {
        if elapsed > 0.0 {
            let mib = sz as f64 / 1024.0 / 1024.0;
            let rate = mib / elapsed;
            crate::ui::success(format!(
                "Saved: {} ({}) in {:.1}s, {:.1} MiB/s",
                dest_path.display(),
                format_size_mb(Some(sz)),
                elapsed,
                rate
            ));
        } else {
            crate::ui::success(format!(
                "Saved: {} ({})",
                dest_path.display(),
                format_size_mb(Some(sz))
            ));
        }
    } else {
        crate::ui::success(format!(
            "Saved: {} ({})",
            dest_path.display(),
            format_size_mb(None)
        ));
    }

    Ok(())
}

pub fn run_interactive_model_downloader() -> Result<()> {
    use crate::ui;

    if crate::is_no_interaction() || !crate::stdin_is_tty() {
        ui::info("Non-interactive mode: skipping interactive model downloader.");
        return Ok(());
    }

    let available = current_manifest()?;

    use std::collections::BTreeMap;
    let mut by_base: BTreeMap<String, Vec<ModelEntry>> = BTreeMap::new();
    for m in available.into_iter() {
        by_base.entry(m.base.clone()).or_default().push(m);
    }

    let pref_order = ["tiny", "small", "base", "medium", "large"];
    let mut ordered_bases: Vec<String> = Vec::new();
    for b in pref_order {
        if by_base.contains_key(b) {
            ordered_bases.push(b.to_string());
        }
    }
    for b in by_base.keys() {
        if !ordered_bases.iter().any(|x| x == b) {
            ordered_bases.push(b.clone());
        }
    }

    ui::intro("PolyScribe model downloader");

    let mut base_labels: Vec<String> = Vec::new();
    for base in &ordered_bases {
        let variants = &by_base[base];
        let (min_sz, max_sz) = variants.iter().fold((None, None), |acc, m| {
            let (mut lo, mut hi) = acc;
            if let Some(sz) = m.size {
                lo = Some(lo.map(|v: u64| v.min(sz)).unwrap_or(sz));
                hi = Some(hi.map(|v: u64| v.max(sz)).unwrap_or(sz));
            }
            (lo, hi)
        });
        let size_info = match (min_sz, max_sz) {
            (Some(lo), Some(hi)) if lo != hi => format!(
                " ~{:.2}–{:.2} MB",
                lo as f64 / 1_000_000.0,
                hi as f64 / 1_000_000.0
            ),
            (Some(sz), _) => format!(" ~{:.2} MB", sz as f64 / 1_000_000.0),
            _ => String::new(),
        };
        base_labels.push(format!("{} ({} types){}", base, variants.len(), size_info));
    }
    let base_refs: Vec<&str> = base_labels.iter().map(|s| s.as_str()).collect();
    let base_idx = ui::prompt_select("Choose a model base", &base_refs)?;
    let chosen_base = ordered_bases[base_idx].clone();

    let mut variants = by_base.remove(&chosen_base).unwrap_or_default();
    variants.sort_by(|a, b| {
        let rank = |v: &str| match v {
            "default" => 0,
            "en" => 1,
            _ => 2,
        };
        rank(&a.variant)
            .cmp(&rank(&b.variant))
            .then_with(|| a.variant.cmp(&b.variant))
    });

    let mut variant_labels: Vec<String> = Vec::new();
    for m in &variants {
        let size = format_size_mb(m.size.as_ref().copied());
        let updated = m
            .last_modified
            .as_deref()
            .map(short_date)
            .map(|d| format!(" • updated {}", d))
            .unwrap_or_default();
        let variant_label = if m.variant == "default" {
            "default"
        } else {
            &m.variant
        };
        variant_labels.push(format!("{} ({}{})", variant_label, size, updated));
    }
    let variant_refs: Vec<&str> = variant_labels.iter().map(|s| s.as_str()).collect();
    let mut defaults = vec![false; variant_refs.len()];
    if !defaults.is_empty() {
        defaults[0] = true;
    }
    let picks = ui::prompt_multi_select(
        &format!("Select types for '{}'", chosen_base),
        &variant_refs,
        Some(&defaults),
    )?;

    if picks.is_empty() {
        ui::warn("No types selected; aborting.");
        ui::outro("No changes made.");
        return Ok(());
    }

    ui::println_above_bars("Downloading selected models...");

    let labels: Vec<String> = picks
        .iter()
        .map(|&i| {
            let m = &variants[i];
            format!("{} ({})", m.name, format_size_mb(m.size))
        })
        .collect();
    let mut pm = ui::progress::FileProgress::default_for_files(labels.len());
    pm.init_files(&labels);

    for (bar_idx, idx) in picks.into_iter().enumerate() {
        let picked = variants[idx].clone();
        pm.set_file_message(bar_idx, "downloading");
        let _path = ensure_model_available_noninteractive(&picked.name)?;
        pm.mark_file_done(bar_idx);
        ui::success(format!("Ready: {}", picked.name));
    }

    pm.finish_total("all done");
    ui::outro("Model selection complete.");
    Ok(())
}

pub fn update_local_models() -> Result<()> {
    use crate::ui;
    use std::collections::HashMap;

    let manifest = current_manifest()?;
    let dir = crate::models_dir_path();
    fs::create_dir_all(&dir).ok();

    ui::info("Checking locally available models, then verifying against the online manifest…");

    let mut by_file: HashMap<String, ModelEntry> = HashMap::new();
    let mut by_stem_or_name: HashMap<String, ModelEntry> = HashMap::new();
    for m in manifest {
        by_file.insert(m.file.to_ascii_lowercase(), m.clone());
        let stem = m
            .file
            .strip_suffix(".bin")
            .unwrap_or(&m.file)
            .to_ascii_lowercase();
        by_stem_or_name.insert(stem, m.clone());
        by_stem_or_name.insert(m.name.to_ascii_lowercase(), m);
    }

    let mut updated = 0usize;
    let mut up_to_date = 0usize;

    let rd = fs::read_dir(&dir).with_context(|| format!("reading models dir {}", dir.display()))?;
    let entries: Vec<_> = rd.flatten().collect();

    if entries.is_empty() {
        ui::info("No local models found.");
    } else {
        for ent in entries {
            let path = ent.path();
            if !path.is_file() {
                continue;
            }
            let is_bin = path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("bin"));
            if !is_bin {
                continue;
            }

            let file_name = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let file_lc = file_name.to_ascii_lowercase();
            let stem_lc = file_lc.strip_suffix(".bin").unwrap_or(&file_lc).to_string();

            let mut manifest_entry = by_file
                .get(&file_lc)
                .or_else(|| by_stem_or_name.get(&stem_lc))
                .cloned();

            let Some(mut m) = manifest_entry.take() else {
                ui::warn(format!(
                    "Skipping unknown local model (not in online manifest): {}",
                    path.display()
                ));
                continue;
            };

            let _ = enrich_entry_via_head(&mut m);

            let target_path = if m.file.eq_ignore_ascii_case(&file_name) {
                path.clone()
            } else {
                dir.join(&m.file)
            };

            if target_path.exists() && file_matches(&target_path, m.size, m.sha256.as_deref())? {
                crate::dlog!(1, "OK: {}", target_path.display());
                up_to_date += 1;
                continue;
            }

            if target_path == path && target_path.exists() {
                crate::ilog!("Updating {}", file_name);
                let _ = fs::remove_file(&target_path);
            } else if !target_path.exists() {
                crate::ilog!("Fetching latest for '{}' -> {}", file_name, m.file);
            } else {
                crate::ilog!("Refreshing {}", target_path.display());
            }

            download_with_progress(&target_path, &m)?;
            updated += 1;
        }

        if updated == 0 {
            ui::info(format!("All {} local model(s) are up to date.", up_to_date));
        } else {
            ui::info(format!("Updated {updated} local model(s)."));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_cache_bypass_environment() {
        unsafe {
            env::remove_var(ConfigService::ENV_NO_CACHE_MANIFEST);
        }
        assert!(!should_bypass_cache());

        unsafe {
            env::set_var(ConfigService::ENV_NO_CACHE_MANIFEST, "1");
        }
        assert!(should_bypass_cache());

        unsafe {
            env::remove_var(ConfigService::ENV_NO_CACHE_MANIFEST);
        }
    }

    #[test]
    fn test_cache_ttl_environment() {
        unsafe {
            env::remove_var(ConfigService::ENV_MANIFEST_TTL_SECONDS);
        }
        assert_eq!(
            get_cache_ttl(),
            ConfigService::DEFAULT_MANIFEST_CACHE_TTL_SECONDS
        );

        unsafe {
            env::set_var(ConfigService::ENV_MANIFEST_TTL_SECONDS, "3600");
        }
        assert_eq!(get_cache_ttl(), 3600);

        unsafe {
            env::remove_var(ConfigService::ENV_MANIFEST_TTL_SECONDS);
        }
    }

    #[test]
    fn test_cached_manifest_serialization() {
        let entries = vec![ModelEntry {
            name: "test".to_string(),
            file: "test.bin".to_string(),
            url: "https://example.com/test.bin".to_string(),
            size: Some(1024),
            sha256: Some("abc123".to_string()),
            last_modified: Some("2023-01-01T00:00:00Z".to_string()),
            base: "test".to_string(),
            variant: "default".to_string(),
        }];

        let cached = CachedManifest {
            fetched_at: 1234567890,
            etag: Some("etag123".to_string()),
            last_modified: Some("2023-01-01T00:00:00Z".to_string()),
            entries: entries.clone(),
        };

        let json = serde_json::to_string(&cached).unwrap();
        let deserialized: CachedManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.fetched_at, cached.fetched_at);
        assert_eq!(deserialized.etag, cached.etag);
        assert_eq!(deserialized.last_modified, cached.last_modified);
        assert_eq!(deserialized.entries.len(), entries.len());
        assert_eq!(deserialized.entries[0].name, entries[0].name);
    }
}
