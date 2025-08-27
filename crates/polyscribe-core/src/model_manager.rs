// SPDX-License-Identifier: MIT

use crate::prelude::*;
use crate::ui::BytesProgress;
use anyhow::{anyhow, Context};
use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use reqwest::header::{
    ACCEPT_RANGES, AUTHORIZATION, CONTENT_LENGTH, ETAG, IF_NONE_MATCH, LAST_MODIFIED, RANGE,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::min;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

const DEFAULT_CHUNK_SIZE: u64 = 8 * 1024 * 1024; // 8 MiB

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelRecord {
    pub alias: String,
    pub repo: String,
    pub file: String,
    pub revision: Option<String>, // ETag or commit hash
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
    pub quant: Option<String>,
    pub installed_at: Option<DateTime<Utc>>,
    pub last_used: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub models: BTreeMap<String, ModelRecord>, // key = alias
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub concurrency: usize,
    pub limit_rate: Option<u64>, // bytes/sec
    pub chunk_size: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            concurrency: 4,
            limit_rate: None,
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Paths {
    pub cache_dir: PathBuf,   // $XDG_CACHE_HOME/polyscribe/models
    pub config_path: PathBuf, // $XDG_CONFIG_HOME/polyscribe/models.json
}

impl Paths {
    pub fn resolve() -> Result<Self> {
        if let Ok(over) = std::env::var("POLYSCRIBE_CACHE_DIR") {
            if !over.is_empty() {
                let cache_dir = PathBuf::from(over).join("models");
                let config_path = std::env::var("POLYSCRIBE_CONFIG_DIR")
                    .map(|p| PathBuf::from(p).join("models.json"))
                    .unwrap_or_else(|_| default_config_path());
                return Ok(Self {
                    cache_dir,
                    config_path,
                });
            }
        }
        let cache_dir = default_cache_dir();
        let config_path = default_config_path();
        Ok(Self {
            cache_dir,
            config_path,
        })
    }
}

fn default_cache_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("polyscribe").join("models");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".cache")
                .join("polyscribe")
                .join("models");
        }
    }
    PathBuf::from("models")
}

fn default_config_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("polyscribe").join("models.json");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".config")
                .join("polyscribe")
                .join("models.json");
        }
    }
    PathBuf::from("models.json")
}

pub trait HttpClient: Send + Sync {
    fn head(&self, url: &str, etag: Option<&str>) -> Result<HeadMeta>;
    fn get_range(&self, url: &str, start: u64, end_inclusive: u64) -> Result<Vec<u8>>;
    fn get_whole_to(&self, url: &str, writer: &mut dyn Write) -> Result<()>;
    fn get_from_to(&self, url: &str, start: u64, writer: &mut dyn Write) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct ReqwestClient {
    client: Client,
    token: Option<String>,
}

impl ReqwestClient {
    pub fn new() -> Result<Self> {
        let token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
        let client = Client::builder()
            .user_agent(crate::config::ConfigService::user_agent())
            .build()?;
        Ok(Self { client, token })
    }

    fn auth(&self, mut req: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        if let Some(t) = &self.token {
            req = req.header(AUTHORIZATION, format!("Bearer {}", t));
        }
        req
    }
}

#[derive(Debug, Clone)]
pub struct HeadMeta {
    pub len: Option<u64>,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub accept_ranges: bool,
    pub not_modified: bool,
    pub status: u16,
}

impl HttpClient for ReqwestClient {
    fn head(&self, url: &str, etag: Option<&str>) -> Result<HeadMeta> {
        let mut req = self.client.head(url);
        if let Some(e) = etag {
            req = req.header(IF_NONE_MATCH, format!("\"{}\"", e));
        }
        let resp = self.auth(req).send()?;
        let status = resp.status().as_u16();
        if status == 304 {
            return Ok(HeadMeta {
                len: None,
                etag: etag.map(|s| s.to_string()),
                last_modified: None,
                accept_ranges: true,
                not_modified: true,
                status,
            });
        }
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
        let last_modified = resp
            .headers()
            .get(LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let accept_ranges = resp
            .headers()
            .get(ACCEPT_RANGES)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_ascii_lowercase().contains("bytes"))
            .unwrap_or(false);
        Ok(HeadMeta {
            len,
            etag,
            last_modified,
            accept_ranges,
            not_modified: false,
            status,
        })
    }

    fn get_range(&self, url: &str, start: u64, end_inclusive: u64) -> Result<Vec<u8>> {
        let range_val = format!("bytes={}-{}", start, end_inclusive);
        let resp = self
            .auth(self.client.get(url))
            .header(RANGE, range_val)
            .send()?;
        if !resp.status().is_success() && resp.status().as_u16() != 206 {
            return Err(anyhow!("HTTP {} for ranged GET", resp.status()).into());
        }
        let mut buf = Vec::new();
        let mut r = resp;
        r.copy_to(&mut buf)?;
        Ok(buf)
    }

    fn get_whole_to(&self, url: &str, writer: &mut dyn Write) -> Result<()> {
        let resp = self.auth(self.client.get(url)).send()?;
        if !resp.status().is_success() {
            return Err(anyhow!("HTTP {} for GET", resp.status()).into());
        }
        let mut r = resp;
        r.copy_to(writer)?;
        Ok(())
    }

    fn get_from_to(&self, url: &str, start: u64, writer: &mut dyn Write) -> Result<()> {
        let mut req = self.auth(self.client.get(url));
        if start > 0 {
            req = req.header(RANGE, format!("bytes={}-", start));
        }
        let resp = req.send()?;
        if !resp.status().is_success() && resp.status().as_u16() != 206 {
            return Err(anyhow!("HTTP {} for ranged GET from {}", resp.status(), start).into());
        }
        let mut r = resp;
        r.copy_to(writer)?;
        Ok(())
    }
}

pub struct ModelManager<C: HttpClient = ReqwestClient> {
    pub paths: Paths,
    pub settings: Settings,
    client: Arc<C>,
}

impl<C: HttpClient + 'static> ModelManager<C> {
    pub fn new_with_client(client: C, settings: Settings) -> Result<Self> {
        Ok(Self {
            paths: Paths::resolve()?,
            settings,
            client: Arc::new(client),
        })
    }

    pub fn new(settings: Settings) -> Result<Self>
    where
        C: Default,
    {
        Ok(Self {
            paths: Paths::resolve()?,
            settings,
            client: Arc::new(C::default()),
        })
    }

    fn load_manifest(&self) -> Result<Manifest> {
        let p = &self.paths.config_path;
        if !p.exists() {
            return Ok(Manifest::default());
        }
        let file = File::open(p).with_context(|| format!("open manifest {}", p.display()))?;
        let m: Manifest = serde_json::from_reader(file).context("parse manifest")?;
        Ok(m)
    }

    fn save_manifest(&self, m: &Manifest) -> Result<()> {
        let p = &self.paths.config_path;
        if let Some(dir) = p.parent() {
            fs::create_dir_all(dir)
                .with_context(|| format!("create config dir {}", dir.display()))?;
        }
        let tmp = p.with_extension("json.tmp");
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        serde_json::to_writer_pretty(f, m).context("serialize manifest")?;
        fs::rename(&tmp, p).with_context(|| format!("atomic rename {} -> {}", tmp.display(), p.display()))?;
        Ok(())
    }

    fn model_path(&self, file: &str) -> PathBuf {
        self.paths.cache_dir.join(file)
    }

    fn compute_sha256(path: &Path) -> Result<String> {
        let mut f = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn ls(&self) -> Result<Vec<ModelRecord>> {
        let m = self.load_manifest()?;
        Ok(m.models.values().cloned().collect())
    }

    pub fn rm(&self, alias: &str) -> Result<bool> {
        let mut m = self.load_manifest()?;
        if let Some(rec) = m.models.remove(alias) {
            let p = self.model_path(&rec.file);
            let _ = fs::remove_file(&p);
            self.save_manifest(&m)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn verify(&self, alias: &str) -> Result<bool> {
        let m = self.load_manifest()?;
        let Some(rec) = m.models.get(alias) else { return Ok(false) };
        let p = self.model_path(&rec.file);
        if !p.exists() { return Ok(false); }
        if let Some(expected) = &rec.sha256 {
            let actual = Self::compute_sha256(&p)?;
            return Ok(&actual == expected);
        }
        Ok(true)
    }

    pub fn gc(&self) -> Result<(usize, usize)> {
        // Remove files not referenced by manifest; also drop manifest entries whose file is missing
        fs::create_dir_all(&self.paths.cache_dir).ok();
        let mut m = self.load_manifest()?;
        let mut referenced = BTreeMap::new();
        for (alias, rec) in &m.models {
            referenced.insert(rec.file.clone(), alias.clone());
        }
        let mut removed_files = 0usize;
        if let Ok(rd) = fs::read_dir(&self.paths.cache_dir) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_file() {
                    let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if !referenced.contains_key(fname) {
                        let _ = fs::remove_file(&p);
                        removed_files += 1;
                    }
                }
            }
        }

        m.models.retain(|_, rec| self.model_path(&rec.file).exists());
        let removed_entries = referenced
            .keys()
            .filter(|f| !self.model_path(f).exists())
            .count();

        self.save_manifest(&m)?;
        Ok((removed_files, removed_entries))
    }

    pub fn add_or_update(
        &self,
        alias: &str,
        repo: &str,
        file: &str,
    ) -> Result<ModelRecord> {
        fs::create_dir_all(&self.paths.cache_dir)
            .with_context(|| format!("create cache dir {}", self.paths.cache_dir.display()))?;

        let url = format!("https://huggingface.co/{}/resolve/main/{}?download=true", repo, file);

        let mut manifest = self.load_manifest()?;
        let prev = manifest.models.get(alias).cloned();
        let prev_etag = prev.as_ref().and_then(|r| r.revision.clone());
        // Fetch remote meta (size/hash) when available to verify the download
        let (_api_size, api_sha) = hf_fetch_file_meta(repo, file).unwrap_or((None, None));
        let head = self.client.head(&url, prev_etag.as_deref())?;
        if head.not_modified {
            // up-to-date; ensure record present and touch last_used
            let mut rec = prev.ok_or_else(|| anyhow!("not installed yet but server says 304"))?;
            rec.last_used = Some(Utc::now());
            self.save_touch(&mut manifest, rec.clone())?;
            return Ok(rec);
        }

        // Quick check: if HEAD failed (e.g., 404), report a helpful error before attempting download
        if head.status >= 400 {
            return Err(anyhow!(
                "file not found or inaccessible: repo='{}' file='{}' (HTTP {})\nHint: run `polyscribe models search {} --query {}` to list available files",
                repo,
                file,
                head.status,
                repo,
                file
            ).into());
        }

        let total_len = head.len.ok_or_else(|| anyhow!("missing content-length (HEAD)"))?;
        let etag = head.etag.clone();
        let dest_tmp = self.model_path(&format!("{}.part", file));
        // If a previous cancelled download left a .part file, remove it to avoid clutter/resume.
        if dest_tmp.exists() { let _ = fs::remove_file(&dest_tmp); }
        // Guard to ensure .part is cleaned up on errors
        struct TempGuard { path: PathBuf, armed: bool }
        impl TempGuard { fn disarm(&mut self) { self.armed = false; } }
        impl Drop for TempGuard {
            fn drop(&mut self) {
                if self.armed {
                    let _ = fs::remove_file(&self.path);
                }
            }
        }
        let mut _tmp_guard = TempGuard { path: dest_tmp.clone(), armed: true };
        let dest_final = self.model_path(file);

        // Do not resume after cancellation; start fresh to avoid stale .part files
        let start_from = 0u64;

        // Open tmp for write
        let f = OpenOptions::new().create(true).write(true).read(true).open(&dest_tmp)?;
        f.set_len(total_len)?; // pre-allocate for random writes
        let f = Arc::new(Mutex::new(f));

        // Create progress bar
        let mut progress = BytesProgress::start(total_len, &format!("Downloading {}", file), start_from);

        // Create work chunks
        let chunk_size = self.settings.chunk_size;
        let mut chunks = Vec::new();
        let mut pos = start_from;
        while pos < total_len {
            let end = min(total_len - 1, pos + chunk_size - 1);
            chunks.push((pos, end));
            pos = end + 1;
        }

        // Attempt concurrent ranged download; on failure, fallback to streaming GET
        let mut ranged_failed = false;
        if head.accept_ranges && self.settings.concurrency > 1 {
            let (work_tx, work_rx) = mpsc::channel::<(u64, u64)>();
            let (prog_tx, prog_rx) = mpsc::channel::<u64>();
            for ch in chunks {
                work_tx.send(ch).unwrap();
            }
            drop(work_tx);
            let rx = Arc::new(Mutex::new(work_rx));

            let workers = self.settings.concurrency.max(1);
            let mut handles = Vec::new();
            for _ in 0..workers {
                let rx = rx.clone();
                let url = url.clone();
                let client = self.client.clone();
                let f = f.clone();
                let limit = self.settings.limit_rate;
                let prog_tx = prog_tx.clone();
                let handle = thread::spawn(move || -> Result<()> {
                    loop {
                        let next = {
                            let guard = rx.lock().unwrap();
                            guard.recv().ok()
                        };
                        let Some((start, end)) = next else { break; };
                        let data = client.get_range(&url, start, end)?;
                        if let Some(max_bps) = limit {
                            let dur = Duration::from_secs_f64((data.len() as f64) / (max_bps as f64));
                            if dur > Duration::from_millis(1) {
                                thread::sleep(dur);
                            }
                        }
                        let mut guard = f.lock().unwrap();
                        guard.seek(SeekFrom::Start(start))?;
                        guard.write_all(&data)?;
                        let _ = prog_tx.send(data.len() as u64);
                    }
                    Ok(())
                });
                handles.push(handle);
            }
            drop(prog_tx);

            for delta in prog_rx {
                progress.inc(delta);
            }

            let mut ranged_err: Option<anyhow::Error> = None;
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => { ranged_failed = true; if ranged_err.is_none() { ranged_err = Some(e.into()); } }
                    Err(_) => { ranged_failed = true; if ranged_err.is_none() { ranged_err = Some(anyhow!("worker panicked")); } }
                }
            }
        } else {
            ranged_failed = true;
        }

        if ranged_failed {
            // Restart progress if we are abandoning previous partial state
            if start_from > 0 {
                progress.stop("retrying as streaming");
                progress = BytesProgress::start(total_len, &format!("Downloading {}", file), 0);
            }
            // Fallback to streaming GET; try URL with and without ?download=true
            let mut try_urls = vec![url.clone()];
            if let Some((base, _qs)) = url.split_once('?') {
                try_urls.push(base.to_string());
            } else {
                try_urls.push(format!("{}?download=true", url));
            }
            // Fresh temp file for streaming
            let mut wf = OpenOptions::new().create(true).write(true).truncate(true).open(&dest_tmp)?;
            let mut ok = false;
            let mut last_err: Option<anyhow::Error> = None;
            // Counting writer to update progress inline
            struct CountingWriter<'a, 'b> {
                inner: &'a mut File,
                progress: &'b mut BytesProgress,
            }
            impl<'a, 'b> Write for CountingWriter<'a, 'b> {
                fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                    let n = self.inner.write(buf)?;
                    self.progress.inc(n as u64);
                    Ok(n)
                }
                fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
            }
            let mut cw = CountingWriter { inner: &mut wf, progress: &mut progress };
            for u in try_urls {
                // For fallback, stream from scratch to ensure integrity
                let res = self.client.get_whole_to(&u, &mut cw);
                match res {
                    Ok(()) => { ok = true; break; }
                    Err(e) => { last_err = Some(e.into()); }
                }
            }
            if !ok {
                if let Some(e) = last_err { return Err(e.into()); }
                return Err(anyhow!("download failed (ranged and streaming)").into());
            }
        }

        progress.stop("download complete");

        // Verify integrity
        let sha = Self::compute_sha256(&dest_tmp)?;
        if let Some(expected) = api_sha.as_ref() {
            if &sha != expected {
                return Err(anyhow!(
                    "sha256 mismatch (expected {}, got {})",
                    expected,
                    sha
                ).into());
            }
        } else if prev.as_ref().map(|r| r.file.eq(file)).unwrap_or(false) {
            if let Some(expected) = prev.as_ref().and_then(|r| r.sha256.as_ref()) {
                if &sha != expected {
                    return Err(anyhow!("sha256 mismatch").into());
                }
            }
        }
        // Atomic rename
        fs::rename(&dest_tmp, &dest_final).with_context(|| format!("rename {} -> {}", dest_tmp.display(), dest_final.display()))?;
        // Disarm guard; .part has been moved or cleaned
        _tmp_guard.disarm();

        let rec = ModelRecord {
            alias: alias.to_string(),
            repo: repo.to_string(),
            file: file.to_string(),
            revision: etag,
            sha256: Some(sha.clone()),
            size_bytes: Some(total_len),
            quant: infer_quant(file),
            installed_at: Some(Utc::now()),
            last_used: Some(Utc::now()),
        };
        self.save_touch(&mut manifest, rec.clone())?;
        Ok(rec)
    }

    fn save_touch(&self, manifest: &mut Manifest, rec: ModelRecord) -> Result<()> {
        manifest.models.insert(rec.alias.clone(), rec);
        self.save_manifest(manifest)
    }
}

fn infer_quant(file: &str) -> Option<String> {
    // Try to extract a Q* token, e.g. Q5_K_M from filename
    let lower = file.to_ascii_uppercase();
    if let Some(pos) = lower.find('Q') {
        let tail = &lower[pos..];
        let token: String = tail
            .chars()
            .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_' || *c == '-')
            .collect();
        if token.len() >= 2 {
            return Some(token);
        }
    }
    None
}

impl Default for ReqwestClient {
    fn default() -> Self {
        Self::new().expect("reqwest client")
    }
}

// Hugging Face API types for file metadata
#[derive(Debug, Deserialize)]
struct ApiHfLfs {
    oid: Option<String>,
    size: Option<u64>,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiHfFile {
    rfilename: String,
    size: Option<u64>,
    sha256: Option<String>,
    lfs: Option<ApiHfLfs>,
}

#[derive(Debug, Deserialize)]
struct ApiHfModelInfo {
    siblings: Option<Vec<ApiHfFile>>,
    files: Option<Vec<ApiHfFile>>,
}

fn pick_sha_from_file(f: &ApiHfFile) -> Option<String> {
    if let Some(s) = &f.sha256 { return Some(s.to_string()); }
    if let Some(l) = &f.lfs {
        if let Some(s) = &l.sha256 { return Some(s.to_string()); }
        if let Some(oid) = &l.oid { return oid.strip_prefix("sha256:").map(|s| s.to_string()); }
    }
    None
}

fn hf_fetch_file_meta(repo: &str, target: &str) -> Result<(Option<u64>, Option<String>)> {
    let token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
    let client = Client::builder()
        .user_agent(crate::config::ConfigService::user_agent())
        .build()?;
    let base = crate::config::ConfigService::hf_api_base_for(repo);
    let urls = [base.clone(), format!("{}?expand=files", base)];
    for url in urls {
        let mut req = client.get(&url);
        if let Some(t) = &token { req = req.header(AUTHORIZATION, format!("Bearer {}", t)); }
        let resp = req.send()?;
        if !resp.status().is_success() { continue; }
        let info: ApiHfModelInfo = resp.json()?;
        let list = info.files.or(info.siblings).unwrap_or_default();
        for f in list {
            let name = f.rfilename.rsplit('/').next().unwrap_or(&f.rfilename);
            if name.eq_ignore_ascii_case(target) {
                let sz = f.size.or_else(|| f.lfs.as_ref().and_then(|l| l.size));
                let sha = pick_sha_from_file(&f);
                return Ok((sz, sha));
            }
        }
    }
    Err(anyhow!("file not found in HF API").into())
}

/// Fetch remote metadata (size, sha256) for a single file in a HF repo.
pub fn fetch_file_meta(repo: &str, file: &str) -> Result<(Option<u64>, Option<String>)> {
    hf_fetch_file_meta(repo, file)
}

/// Search a Hugging Face repo for GGUF/BIN files via API. Returns file names only.
pub fn search_repo(repo: &str, query: Option<&str>) -> Result<Vec<String>> {
    let token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
    let client = Client::builder()
        .user_agent(crate::config::ConfigService::user_agent())
        .build()?;

    let base = crate::config::ConfigService::hf_api_base_for(repo);
    let mut urls = vec![base.clone(), format!("{}?expand=files", base)];
    let mut files = Vec::<String>::new();

    for url in urls.drain(..) {
        let mut req = client.get(&url);
        if let Some(t) = &token { req = req.header(AUTHORIZATION, format!("Bearer {}", t)); }
        let resp = req.send()?;
        if !resp.status().is_success() { continue; }
        let info: ApiHfModelInfo = resp.json()?;
        let list = info.files.or(info.siblings).unwrap_or_default();
        for f in list {
            if f.rfilename.ends_with(".gguf") || f.rfilename.ends_with(".bin") {
                let name = f.rfilename.rsplit('/').next().unwrap_or(&f.rfilename).to_string();
                if !files.contains(&name) { files.push(name); }
            }
        }
        if !files.is_empty() { break; }
    }

    if let Some(q) = query { let qlc = q.to_ascii_lowercase(); files.retain(|f| f.to_ascii_lowercase().contains(&qlc)); }
    files.sort();
    Ok(files)
}

/// List repo files with optional size metadata for GGUF/BIN entries.
pub fn list_repo_files_with_meta(repo: &str) -> Result<Vec<(String, Option<u64>)>> {
    let token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
    let client = Client::builder()
        .user_agent(crate::config::ConfigService::user_agent())
        .build()?;

    let base = crate::config::ConfigService::hf_api_base_for(repo);
    for url in [base.clone(), format!("{}?expand=files", base)] {
        let mut req = client.get(&url);
        if let Some(t) = &token { req = req.header(AUTHORIZATION, format!("Bearer {}", t)); }
        let resp = req.send()?;
        if !resp.status().is_success() { continue; }
        let info: ApiHfModelInfo = resp.json()?;
        let list = info.files.or(info.siblings).unwrap_or_default();
        let mut out = Vec::new();
        for f in list {
            if !(f.rfilename.ends_with(".gguf") || f.rfilename.ends_with(".bin")) { continue; }
            let name = f.rfilename.rsplit('/').next().unwrap_or(&f.rfilename).to_string();
            let size = f.size.or_else(|| f.lfs.as_ref().and_then(|l| l.size));
            out.push((name, size));
        }
        if !out.is_empty() { return Ok(out); }
    }
    Ok(Vec::new())
}

/// Fallback: HEAD request for a single file to retrieve Content-Length (size).
pub fn head_len_for_file(repo: &str, file: &str) -> Option<u64> {
    let token = std::env::var("HF_TOKEN").ok().filter(|s| !s.is_empty());
    let client = Client::builder()
        .user_agent(crate::config::ConfigService::user_agent())
        .build().ok()?;
    let mut urls = Vec::new();
    urls.push(format!("https://huggingface.co/{}/resolve/main/{}?download=true", repo, file));
    urls.push(format!("https://huggingface.co/{}/resolve/main/{}", repo, file));
    for url in urls {
        let mut req = client.head(&url);
        if let Some(t) = &token { req = req.header(AUTHORIZATION, format!("Bearer {}", t)); }
        if let Ok(resp) = req.send() {
            if resp.status().is_success() {
                if let Some(len) = resp.headers().get(CONTENT_LENGTH)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                { return Some(len); }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::TempDir;

    #[derive(Clone)]
    struct StubHttp {
        data: Arc<Vec<u8>>,
        etag: Arc<Option<String>>,
        accept_ranges: bool,
    }
    impl HttpClient for StubHttp {
        fn head(&self, _url: &str, etag: Option<&str>) -> Result<HeadMeta> {
            let not_modified = etag.is_some() && self.etag.as_ref().as_deref() == etag;
            Ok(HeadMeta {
                len: Some(self.data.len() as u64),
                etag: self.etag.as_ref().clone(),
                last_modified: None,
                accept_ranges: self.accept_ranges,
                not_modified,
                status: if not_modified { 304 } else { 200 },
            })
        }
        fn get_range(&self, _url: &str, start: u64, end_inclusive: u64) -> Result<Vec<u8>> {
            let s = start as usize;
            let e = (end_inclusive as usize) + 1;
            Ok(self.data[s..e].to_vec())
        }

        fn get_whole_to(&self, _url: &str, writer: &mut dyn Write) -> Result<()> {
            writer.write_all(&self.data)?;
            Ok(())
        }

        fn get_from_to(&self, _url: &str, start: u64, writer: &mut dyn Write) -> Result<()> {
            let s = start as usize;
            writer.write_all(&self.data[s..])?;
            Ok(())
        }
    }

    fn setup_env(cache: &Path, cfg: &Path) {
        unsafe {
            env::set_var("POLYSCRIBE_CACHE_DIR", cache.to_string_lossy().to_string());
            env::set_var(
                "POLYSCRIBE_CONFIG_DIR",
                cfg.parent().unwrap().to_string_lossy().to_string(),
            );
        }
    }

    #[test]
    fn test_manifest_roundtrip() {
        let temp = TempDir::new().unwrap();
        let cache = temp.path().join("cache");
        let cfg = temp.path().join("config").join("models.json");
        setup_env(&cache, &cfg);

        let client = StubHttp {
            data: Arc::new(vec![0u8; 1024]),
            etag: Arc::new(Some("etag123".into())),
            accept_ranges: true,
        };
        let mm: ModelManager<StubHttp> = ModelManager::new_with_client(client, Settings::default()).unwrap();
        let m = mm.load_manifest().unwrap();
        assert!(m.models.is_empty());

        let rec = ModelRecord {
            alias: "tiny".into(),
            repo: "foo/bar".into(),
            file: "gguf-tiny.bin".into(),
            revision: Some("etag123".into()),
            sha256: None,
            size_bytes: None,
            quant: None,
            installed_at: None,
            last_used: None,
        };

        let mut m2 = Manifest::default();
        mm.save_touch(&mut m2, rec.clone()).unwrap();
        let m3 = mm.load_manifest().unwrap();
        assert!(m3.models.contains_key("tiny"));
    }

    #[test]
    fn test_add_verify_update_gc() {
        let temp = TempDir::new().unwrap();
        let cache = temp.path().join("cache");
        let cfg_dir = temp.path().join("config");
        let cfg = cfg_dir.join("models.json");
        setup_env(&cache, &cfg);

        let data = (0..1024 * 1024u32).flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>();
        let etag = Some("abc123".to_string());
        let client = StubHttp { data: Arc::new(data), etag: Arc::new(etag), accept_ranges: true };
        let mm: ModelManager<StubHttp> = ModelManager::new_with_client(client.clone(), Settings{ concurrency: 3, ..Default::default() }).unwrap();

        // add
        let rec = mm.add_or_update("tiny", "gguf/models", "gguf-tiny-q4_0.bin").unwrap();
        assert_eq!(rec.alias, "tiny");
        assert!(mm.verify("tiny").unwrap());

        // update (304)
        let rec2 = mm.add_or_update("tiny", "gguf/models", "gguf-tiny-q4_0.bin").unwrap();
        assert_eq!(rec2.alias, "tiny");

        // gc (nothing to remove)
        let (files_removed, entries_removed) = mm.gc().unwrap();
        assert_eq!(files_removed, 0);
        assert_eq!(entries_removed, 0);

        // rm
        assert!(mm.rm("tiny").unwrap());
        assert!(!mm.rm("tiny").unwrap());
    }
}
