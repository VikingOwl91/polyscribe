use std::fs::{File, create_dir_all};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::env;
use std::collections::BTreeMap;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use chrono::Local;
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};
use std::time::Duration;

#[cfg(feature = "native-whisper")]
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Parser, Debug)]
#[command(name = "merge_transcripts", version, about = "Merge multiple JSON transcripts into one or transcribe audio using native whisper")]
struct Args {
    /// Input .json transcript files or audio files to merge/transcribe
    inputs: Vec<String>,

    /// Output file path base (date prefix will be added); if omitted, writes JSON to stdout
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,

    /// Language code to use for transcription (e.g., en, de). No auto-detection.
    #[arg(short, long, value_name = "LANG")]
    language: Option<String>,

    /// Launch interactive model downloader (list HF models, multi-select and download)
    #[arg(long)]
    download_models: bool,
}

#[derive(Debug, Deserialize)]
struct InputRoot {
    #[serde(default)]
    segments: Vec<InputSegment>,
}

#[derive(Debug, Deserialize)]
struct InputSegment {
    start: f64,
    end: f64,
    text: String,
    // other fields are ignored
}

#[derive(Debug, Serialize)]
struct OutputEntry {
    id: u64,
    speaker: String,
    start: f64,
    end: f64,
    text: String,
}

#[derive(Debug, Serialize)]
struct OutputRoot {
    items: Vec<OutputEntry>,
}

fn date_prefix() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as i64;
    let ms = (total_ms % 1000) as i64;
    let total_secs = total_ms / 1000;
    let s = (total_secs % 60) as i64;
    let m = ((total_secs / 60) % 60) as i64;
    let h = (total_secs / 3600) as i64;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

fn render_srt(items: &[OutputEntry]) -> String {
    let mut out = String::new();
    for (i, e) in items.iter().enumerate() {
        let idx = i + 1;
        out.push_str(&format!("{}\n", idx));
        out.push_str(&format!("{} --> {}\n", format_srt_time(e.start), format_srt_time(e.end)));
        if !e.speaker.is_empty() {
            out.push_str(&format!("{}: {}\n", e.speaker, e.text));
        } else {
            out.push_str(&format!("{}\n", e.text));
        }
        out.push('\n');
    }
    out
}

fn sanitize_speaker_name(raw: &str) -> String {
    if let Some((prefix, rest)) = raw.split_once('-') {
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            return rest.to_string();
        }
    }
    raw.to_string()
}

// --- Helpers for audio transcription ---
fn is_json_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()), Some(ext) if ext == "json")
}

fn is_audio_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()) {
        let exts = [
            "mp3","wav","m4a","mp4","aac","flac","ogg","wma","webm","mkv","mov","avi","m4b","3gp","opus","aiff","alac"
        ];
        return exts.contains(&ext.as_str());
    }
    false
}

fn normalize_lang_code(input: &str) -> Option<String> {
    let mut s = input.trim().to_lowercase();
    if s.is_empty() || s == "auto" || s == "c" || s == "posix" { return None; }
    if let Some((lhs, _)) = s.split_once('.') { s = lhs.to_string(); }
    if let Some((lhs, _)) = s.split_once('_') { s = lhs.to_string(); }
    let code = match s.as_str() {
        // ISO codes directly
        "en"=>"en","de"=>"de","es"=>"es","fr"=>"fr","it"=>"it","pt"=>"pt","nl"=>"nl","ru"=>"ru","pl"=>"pl",
        "uk"=>"uk","cs"=>"cs","sv"=>"sv","no"=>"no","da"=>"da","fi"=>"fi","hu"=>"hu","tr"=>"tr","el"=>"el",
        "zh"=>"zh","ja"=>"ja","ko"=>"ko","ar"=>"ar","he"=>"he","hi"=>"hi","ro"=>"ro","bg"=>"bg","sk"=>"sk",
        // Common English names
        "english"=>"en","german"=>"de","spanish"=>"es","french"=>"fr","italian"=>"it","portuguese"=>"pt",
        "dutch"=>"nl","russian"=>"ru","polish"=>"pl","ukrainian"=>"uk","czech"=>"cs","swedish"=>"sv",
        "norwegian"=>"no","danish"=>"da","finnish"=>"fi","hungarian"=>"hu","turkish"=>"tr","greek"=>"el",
        "chinese"=>"zh","japanese"=>"ja","korean"=>"ko","arabic"=>"ar","hebrew"=>"he","hindi"=>"hi",
        "romanian"=>"ro","bulgarian"=>"bg","slovak"=>"sk",
        _ => return None,
    };
    Some(code.to_string())
}



#[cfg(feature = "native-whisper")]
fn find_model_file() -> Result<PathBuf> {
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        create_dir_all(models_dir).with_context(|| format!("Failed to create models directory: {}", models_dir.display()))?;
    }

    // If env var WHISPER_MODEL is set and valid, prefer it
    if let Ok(env_model) = env::var("WHISPER_MODEL") {
        let p = PathBuf::from(env_model);
        if p.is_file() {
            // persist selection
            let _ = std::fs::write(models_dir.join(".last_model"), p.display().to_string());
            return Ok(p);
        }
    }

    // Enumerate local models
    let mut candidates: Vec<PathBuf> = Vec::new();
    let rd = std::fs::read_dir(models_dir)
        .with_context(|| format!("Failed to read models directory: {}", models_dir.display()))?;
    for entry in rd {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()) {
                if ext == "bin" {
                    candidates.push(path);
                }
            }
        }
    }

    if candidates.is_empty() {
        eprintln!("No Whisper model files (*.bin) found in {}.", models_dir.display());
        eprint!("Would you like to download models now? [Y/n]: ");
        io::stderr().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        let ans = input.trim().to_lowercase();
        if ans.is_empty() || ans == "y" || ans == "yes" {
            if let Err(e) = run_interactive_model_downloader() {
                eprintln!("Downloader failed: {:#}", e);
            }
            // Re-scan
            candidates.clear();
            let rd2 = std::fs::read_dir(models_dir)
                .with_context(|| format!("Failed to read models directory: {}", models_dir.display()))?;
            for entry in rd2 {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()) {
                        if ext == "bin" {
                            candidates.push(path);
                        }
                    }
                }
            }
        }
    }

    if candidates.is_empty() {
        return Err(anyhow!("No Whisper model files (*.bin) available in {}", models_dir.display()));
    }

    // If only one, persist and return it
    if candidates.len() == 1 {
        let only = candidates.remove(0);
        let _ = std::fs::write(models_dir.join(".last_model"), only.display().to_string());
        return Ok(only);
    }

    // If a previous selection exists and is still valid, use it
    let last_file = models_dir.join(".last_model");
    if let Ok(prev) = std::fs::read_to_string(&last_file) {
        let prev = prev.trim();
        if !prev.is_empty() {
            let p = PathBuf::from(prev);
            if p.is_file() {
                // Also ensure it's one of the candidates (same dir)
                if candidates.iter().any(|c| c == &p) {
                    eprintln!("Using previously selected model: {}", p.display());
                    return Ok(p);
                }
            }
        }
    }

    // Multiple models and no previous selection: prompt user to choose, then persist
    eprintln!("Multiple Whisper models found in {}:", models_dir.display());
    for (i, p) in candidates.iter().enumerate() {
        eprintln!("  {}) {}", i + 1, p.display());
    }
    eprint!("Select model by number [1-{}]: ", candidates.len());
    io::stderr().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).context("Failed to read selection")?;
    let sel: usize = input.trim().parse().map_err(|_| anyhow!("Invalid selection: {}", input.trim()))?;
    if sel == 0 || sel > candidates.len() {
        return Err(anyhow!("Selection out of range"));
    }
    let chosen = candidates.swap_remove(sel - 1);
    let _ = std::fs::write(models_dir.join(".last_model"), chosen.display().to_string());
    Ok(chosen)
}

#[cfg(feature = "native-whisper")]
fn decode_audio_to_pcm_f32_ffmpeg(audio_path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .arg("-i").arg(audio_path)
        .arg("-f").arg("f32le")
        .arg("-ac").arg("1")
        .arg("-ar").arg("16000")
        .arg("pipe:1")
        .output()
        .with_context(|| format!("Failed to execute ffmpeg for {}", audio_path.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg failed for {}: {}",
            audio_path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let bytes = output.stdout;
    if bytes.len() % 4 != 0 {
        // Truncate to nearest multiple of 4 bytes to avoid partial f32
        let truncated = bytes.len() - (bytes.len() % 4);
        let mut v = Vec::with_capacity(truncated / 4);
        for chunk in bytes[..truncated].chunks_exact(4) {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            v.push(f32::from_le_bytes(arr));
        }
        Ok(v)
    } else {
        let mut v = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            v.push(f32::from_le_bytes(arr));
        }
        Ok(v)
    }
}

#[cfg(feature = "native-whisper")]
fn transcribe_native(audio_path: &Path, speaker: &str, lang_opt: Option<&str>) -> Result<Vec<OutputEntry>> {
    let pcm = decode_audio_to_pcm_f32_ffmpeg(audio_path)?;
    let model = find_model_file()?;
    let is_en_only = model
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains(".en.") || s.ends_with(".en.bin"))
        .unwrap_or(false);
    if let Some(lang) = lang_opt {
        if is_en_only && lang != "en" {
            return Err(anyhow!(
                "Selected model is English-only ({}), but a non-English language hint '{}' was provided. Please use a multilingual model like models/ggml-base.bin or set WHISPER_MODEL accordingly.",
                model.display(),
                lang
            ));
        }
    }
    let model_str = model.to_str().ok_or_else(|| anyhow!("Model path not valid UTF-8: {}", model.display()))?;

    // Initialize Whisper with GPU preference
    let cparams = WhisperContextParameters::default();
    // Prefer GPU if available; default whisper.cpp already has use_gpu=true. If the wrapper exposes
    // a gpu_device field in the future, we could set it here from WHISPER_GPU_DEVICE.
    if let Ok(dev_str) = env::var("WHISPER_GPU_DEVICE") {
        let _ = dev_str.trim().parse::<i32>().ok();
    }
    // Even if we can't set fields explicitly (due to API differences), whisper.cpp defaults to GPU.
    let ctx = WhisperContext::new_with_params(model_str, cparams)
        .with_context(|| format!("Failed to load Whisper model at {}", model.display()))?;
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Failed to create Whisper state: {:?}", e))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let n_threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(1);
    params.set_n_threads(n_threads);
    params.set_translate(false);
    if let Some(lang) = lang_opt { params.set_language(Some(lang)); }

    state.full(params, &pcm)
        .map_err(|e| anyhow!("Whisper full() failed: {:?}", e))?;

    let num_segments = state.full_n_segments().map_err(|e| anyhow!("Failed to get segments: {:?}", e))?;
    let mut items = Vec::new();
    for i in 0..num_segments {
        let text = state.full_get_segment_text(i)
            .map_err(|e| anyhow!("Failed to get segment text: {:?}", e))?;
        let t0 = state.full_get_segment_t0(i).map_err(|e| anyhow!("Failed to get segment t0: {:?}", e))?;
        let t1 = state.full_get_segment_t1(i).map_err(|e| anyhow!("Failed to get segment t1: {:?}", e))?;
        let start = (t0 as f64) * 0.01;
        let end = (t1 as f64) * 0.01;
        items.push(OutputEntry { id: 0, speaker: speaker.to_string(), start, end, text: text.trim().to_string() });
    }
    Ok(items)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // If requested, run the interactive model downloader first. If no inputs were provided, exit after downloading.
    if args.download_models {
        if let Err(e) = run_interactive_model_downloader() {
            eprintln!("Model downloader failed: {:#}", e);
        }
        if args.inputs.is_empty() {
            return Ok(());
        }
    }

    // Determine inputs and optional output path
    let mut inputs = args.inputs;
    let mut output_path = args.output;
    if output_path.is_none() && inputs.len() >= 2 {
        if let Some(last) = inputs.last().cloned() {
            if !Path::new(&last).exists() {
                inputs.pop();
                output_path = Some(last);
            }
        }
    }
    if inputs.is_empty() {
        return Err(anyhow!("No input files provided"));
    }

    // Language must be provided via CLI when transcribing audio (no detection from JSON/env)
    let lang_hint: Option<String> = if let Some(ref l) = args.language {
        normalize_lang_code(l).or_else(|| Some(l.trim().to_lowercase()))
    } else {
        None
    };
    let any_audio = inputs.iter().any(|p| is_audio_file(Path::new(p)));
    if any_audio && lang_hint.is_none() {
        return Err(anyhow!("Please specify --language (e.g., --language en). Language detection was removed."));
    }

    let mut entries: Vec<OutputEntry> = Vec::new();

    for input_path in &inputs {
        let path = Path::new(input_path);
        let speaker = sanitize_speaker_name(
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("speaker")
        );

        let mut buf = String::new();
        if is_audio_file(path) {
            #[cfg(feature = "native-whisper")]
            {
                let items = transcribe_native(path, &speaker, lang_hint.as_deref())?;
                for e in items {
                    entries.push(e);
                }
                continue;
            }
            #[cfg(not(feature = "native-whisper"))]
            {
                return Err(anyhow!("Python transcription has been removed. Please build with --features native-whisper to transcribe audio."));
            }
        } else if is_json_file(path) {
            File::open(path)
                .with_context(|| format!("Failed to open: {}", input_path))?
                .read_to_string(&mut buf)
                .with_context(|| format!("Failed to read: {}", input_path))?;
        } else {
            return Err(anyhow!(format!("Unsupported input type (expected .json or audio media): {}", input_path)));
        }

        let root: InputRoot = serde_json::from_str(&buf)
            .with_context(|| format!("Invalid JSON transcript parsed from {}", input_path))?;

        for seg in root.segments {
            entries.push(OutputEntry {
                id: 0, // will be reassigned after sorting
                speaker: speaker.clone(),
                start: seg.start,
                end: seg.end,
                text: seg.text,
            });
        }
    }

    // Sort globally by (start, end)
    entries.sort_by(|a, b| {
        match a.start.partial_cmp(&b.start) {
            Some(std::cmp::Ordering::Equal) | None => {}
            Some(o) => return o,
        }
        a.end
            .partial_cmp(&b.end)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reassign contiguous ids by chronological order
    for (i, e) in entries.iter_mut().enumerate() {
        e.id = i as u64;
    }

    // Output as an object containing the array (valid JSON)
    // Schema equivalent to:
    // {
    //   [
    //     id: number,
    //     speaker: string,
    //     start: number,
    //     end: number,
    //     text: string
    //   ], ...
    // }
    let out = OutputRoot { items: entries };

    if let Some(path) = output_path {
        let base_path = Path::new(&path);
        let parent_opt = base_path.parent();
        if let Some(parent) = parent_opt {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).with_context(|| {
                    format!("Failed to create parent directory for output: {}", parent.display())
                })?;
            }
        }
        let stem = base_path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        let date = date_prefix();
        let base_name = format!("{}_{}", date, stem);
        let dir = parent_opt.unwrap_or(Path::new(""));
        let json_path = dir.join(format!("{}.json", &base_name));
        let toml_path = dir.join(format!("{}.toml", &base_name));
        let srt_path = dir.join(format!("{}.srt", &base_name));

        // JSON
        let mut json_file = File::create(&json_path)
            .with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
        serde_json::to_writer_pretty(&mut json_file, &out)?;
        writeln!(&mut json_file)?;

        // TOML
        let toml_str = toml::to_string_pretty(&out)?;
        let mut toml_file = File::create(&toml_path)
            .with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
        toml_file.write_all(toml_str.as_bytes())?;
        if !toml_str.ends_with('\n') {
            writeln!(&mut toml_file)?;
        }

        // SRT
        let srt_str = render_srt(&out.items);
        let mut srt_file = File::create(&srt_path)
            .with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
        srt_file.write_all(srt_str.as_bytes())?;
    } else {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, &out)?;
        writeln!(&mut handle)?;
    }
    Ok(())
}

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

#[derive(Clone, Debug)]
struct ModelEntry {
    // e.g. "tiny.en-q5_1"
    name: String,
    base: String,
    subtype: String,
    size: u64,
    sha256: Option<String>,
    repo: &'static str, // e.g. "ggerganov/whisper.cpp"
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
    if b >= GB { format!("{:.2} GiB", b / GB) }
    else if b >= MB { format!("{:.2} MiB", b / MB) }
    else if b >= KB { format!("{:.2} KiB", b / KB) }
    else { format!("{} B", bytes) }
}

fn to_hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes { s.push_str(&format!("{:02x}", b)); }
    s
}

fn expected_sha_from_sibling(s: &HFSibling) -> Option<String> {
    if let Some(h) = &s.sha256 { return Some(h.to_lowercase()); }
    if let Some(lfs) = &s.lfs {
        if let Some(h) = &lfs.sha256 { return Some(h.to_lowercase()); }
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
    if let Some(sz) = s.size { return Some(sz); }
    if let Some(lfs) = &s.lfs { return lfs.size; }
    None
}

fn expected_sha_from_tree(s: &HFTreeItem) -> Option<String> {
    if let Some(h) = &s.sha256 { return Some(h.to_lowercase()); }
    if let Some(lfs) = &s.lfs {
        if let Some(h) = &lfs.sha256 { return Some(h.to_lowercase()); }
        if let Some(oid) = &lfs.oid {
            if let Some(rest) = oid.strip_prefix("sha256:") {
                return Some(rest.to_lowercase().to_string());
            }
        }
    }
    None
}

fn size_from_tree(s: &HFTreeItem) -> Option<u64> {
    if let Some(sz) = s.size { return Some(sz); }
    if let Some(lfs) = &s.lfs { return lfs.size; }
    None
}

fn fill_meta_via_head(repo: &'static str, name: &str) -> (Option<u64>, Option<String>) {
    let head_client = match Client::builder()
        .user_agent("dialogue_merger/0.1 (+https://github.com/)")
        .redirect(Policy::none())
        .timeout(Duration::from_secs(30))
        .build() {
        Ok(c) => c,
        Err(_) => return (None, None),
    };
    let url = format!("https://huggingface.co/{}/resolve/main/ggml-{}.bin", repo, name);
    let resp = match head_client.head(url).send().and_then(|r| r.error_for_status()) {
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
    eprintln!("Fetching online data: listing models from {}...", repo);
    // Prefer the tree endpoint for reliable size/hash metadata, then fall back to model metadata
    let tree_url = format!("https://huggingface.co/api/models/{}/tree/main?recursive=1", repo);
    let mut out: Vec<ModelEntry> = Vec::new();

    match client.get(tree_url).send().and_then(|r| r.error_for_status()) {
        Ok(resp) => {
            match resp.json::<Vec<HFTreeItem>>() {
                Ok(items) => {
                    for it in items {
                        let path = it.path.clone();
                        if !(path.starts_with("ggml-") && path.ends_with(".bin")) { continue; }
                        let model_name = path.trim_start_matches("ggml-").trim_end_matches(".bin").to_string();
                        let (base, subtype) = split_model_name(&model_name);
                        let size = size_from_tree(&it).unwrap_or(0);
                        let sha256 = expected_sha_from_tree(&it);
                        out.push(ModelEntry { name: model_name, base, subtype, size, sha256, repo });
                    }
                }
                Err(_) => { /* fall back below */ }
            }
        }
        Err(_) => { /* fall back below */ }
    }

    if out.is_empty() {
        let url = format!("https://huggingface.co/api/models/{}", repo);
        let resp = client
            .get(url)
            .send()
            .and_then(|r| r.error_for_status())
            .context("Failed to query Hugging Face API")?;

        let info: HFRepoInfo = resp.json().context("Failed to parse Hugging Face API response")?;

        if let Some(files) = info.siblings {
            for s in files {
                let rf = s.rfilename.clone();
                if !(rf.starts_with("ggml-") && rf.ends_with(".bin")) { continue; }
                let model_name = rf.trim_start_matches("ggml-").trim_end_matches(".bin").to_string();
                let (base, subtype) = split_model_name(&model_name);
                let size = size_from_sibling(&s).unwrap_or(0);
                let sha256 = expected_sha_from_sibling(&s);
                out.push(ModelEntry { name: model_name, base, subtype, size, sha256, repo });
            }
        }
    }

    // Fill missing metadata (size/hash) via HEAD request if necessary
    if out.iter().any(|m| m.size == 0 || m.sha256.is_none()) {
        eprintln!("Fetching online data: completing metadata checks for models in {}...", repo);
    }
    for m in out.iter_mut() {
        if m.size == 0 || m.sha256.is_none() {
            let (sz, sha) = fill_meta_via_head(m.repo, &m.name);
            if m.size == 0 {
                if let Some(s) = sz { m.size = s; }
            }
            if m.sha256.is_none() {
                m.sha256 = sha;
            }
        }
    }

    // Sort by base then subtype then name for stable listing
    out.sort_by(|a, b| a.base.cmp(&b.base).then(a.subtype.cmp(&b.subtype)).then(a.name.cmp(&b.name)));
    Ok(out)
}


fn fetch_all_models(client: &Client) -> Result<Vec<ModelEntry>> {
    eprintln!("Fetching online data: aggregating available models from Hugging Face...");
    let mut v1 = hf_fetch_repo_models(client, "ggerganov/whisper.cpp")?; // main repo must succeed

    // Optional tinydiarize repo; ignore errors but log to stderr
    let mut v2: Vec<ModelEntry> = match hf_fetch_repo_models(client, "akashmjn/tinydiarize-whisper.cpp") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Warning: failed to fetch optional repo akashmjn/tinydiarize-whisper.cpp: {:#}", e);
            Vec::new()
        }
    };

    v1.append(&mut v2);

    // Deduplicate by name preferring ggerganov repo if duplicates
    let mut map: BTreeMap<String, ModelEntry> = BTreeMap::new();
    for m in v1 {
        map.entry(m.name.clone()).and_modify(|existing| {
            if existing.repo != "ggerganov/whisper.cpp" && m.repo == "ggerganov/whisper.cpp" {
                *existing = m.clone();
            }
        }).or_insert(m);
    }

    let mut list: Vec<ModelEntry> = map.into_values().collect();
    list.sort_by(|a, b| a.base.cmp(&b.base).then(a.subtype.cmp(&b.subtype)).then(a.name.cmp(&b.name)));
    Ok(list)
}

fn print_grouped_models(models: &[ModelEntry]) {
    let mut current = "".to_string();
    for m in models {
        if m.base != current {
            current = m.base.clone();
            println!("\n{}:", current);
        }
        let short_hash = m
            .sha256
            .as_ref()
            .map(|h| h.chars().take(8).collect::<String>())
            .unwrap_or_else(|| "-".to_string());
        println!("  - {}  [{} | {} | {}]", m.name, m.repo, human_size(m.size), short_hash);
    }
    println!("\nEnter selection by indices (e.g., 1 3 5-7), or 'all', '*' for all, 'q' to cancel.");
}

fn prompt_select_models(models: &[ModelEntry]) -> Result<Vec<ModelEntry>> {
    // Build a flat list but show group headers; indices count only models
    println!("Available ggml Whisper models:");
    let mut idx = 1usize;
    let mut current = "".to_string();
    // We'll record mapping from index -> position in models
    let mut index_map: Vec<usize> = Vec::with_capacity(models.len());
    for (pos, m) in models.iter().enumerate() {
        if m.base != current {
            current = m.base.clone();
            println!("\n{}:", current);
        }
        let short_hash = m
            .sha256
            .as_ref()
            .map(|h| h.chars().take(8).collect::<String>())
            .unwrap_or_else(|| "-".to_string());
        println!("  {}) {}  [{} | {} | {}]", idx, m.name, m.repo, human_size(m.size), short_hash);
        index_map.push(pos);
        idx += 1;
    }
    println!("\nEnter selection by indices (e.g., 1 3 5-7), or 'all', '*' for all, 'q' to cancel.");
    loop {
        eprint!("Selection: ");
        io::stderr().flush().ok();
        let mut line = String::new();
        io::stdin().read_line(&mut line).context("Failed to read selection")?;
        let s = line.trim().to_lowercase();
        if s == "q" || s == "quit" || s == "exit" { return Ok(Vec::new()); }
        let mut selected: Vec<usize> = Vec::new();
        if s == "all" || s == "*" {
            selected = (1..idx).collect();
        } else if !s.is_empty() {
            for part in s.split(|c| c == ',' || c == ' ' || c == ';') {
                let part = part.trim();
                if part.is_empty() { continue; }
                if let Some((a, b)) = part.split_once('-') {
                    if let (Ok(ia), Ok(ib)) = (a.parse::<usize>(), b.parse::<usize>()) {
                        if ia >= 1 && ib < idx && ia <= ib { selected.extend(ia..=ib); }
                    }
                } else if let Ok(i) = part.parse::<usize>() {
                    if i >= 1 && i < idx { selected.push(i); }
                }
            }
        }
        selected.sort_unstable();
        selected.dedup();
        if selected.is_empty() {
            eprintln!("No valid selection. Please try again or 'q' to cancel.");
            continue;
        }
        let chosen: Vec<ModelEntry> = selected.into_iter().map(|i| models[index_map[i - 1]].clone()).collect();
        return Ok(chosen);
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
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex_lower(&hasher.finalize()))
}

fn download_one_model(client: &Client, models_dir: &Path, entry: &ModelEntry) -> Result<()> {
    let final_path = models_dir.join(format!("ggml-{}.bin", entry.name));

    // If the model already exists, verify against online metadata
    if final_path.exists() {
        if let Some(expected) = &entry.sha256 {
            match compute_file_sha256_hex(&final_path) {
                Ok(local_hash) => {
                    if local_hash.eq_ignore_ascii_case(expected) {
                        eprintln!("Model {} is up-to-date (hash match).", final_path.display());
                        return Ok(());
                    } else {
                        eprintln!(
                            "Local model {} hash differs from online (local {}.., online {}..). Updating...",
                            final_path.display(),
                            &local_hash[..std::cmp::min(8, local_hash.len())],
                            &expected[..std::cmp::min(8, expected.len())]
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to hash existing {}: {}. Will re-download to ensure correctness.",
                        final_path.display(), e
                    );
                }
            }
        } else if entry.size > 0 {
            match std::fs::metadata(&final_path) {
                Ok(md) => {
                    if md.len() == entry.size {
                        eprintln!(
                            "Model {} appears up-to-date by size ({}).",
                            final_path.display(), entry.size
                        );
                        return Ok(());
                    } else {
                        eprintln!(
                            "Local model {} size ({}) differs from online ({}). Updating...",
                            final_path.display(), md.len(), entry.size
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to stat existing {}: {}. Will re-download to ensure correctness.",
                        final_path.display(), e
                    );
                }
            }
        } else {
            eprintln!(
                "Model {} exists but remote hash/size not available; will download to verify contents.",
                final_path.display()
            );
            // Fall through to download for content comparison
        }
    }

    let url = format!("https://huggingface.co/{}/resolve/main/ggml-{}.bin", entry.repo, entry.name);
    eprintln!("Downloading {} ({} | {})...", entry.name, human_size(entry.size), url);
    let mut resp = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .context("Failed to download model")?;

    let tmp_path = models_dir.join(format!("ggml-{}.bin.part", entry.name));
    if tmp_path.exists() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    let mut file = std::io::BufWriter::new(
        File::create(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?
    );

    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 128];
    loop {
        let n = resp.read(&mut buf).context("Network read error")?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n]).context("Write error")?;
    }
    file.flush().ok();

    let got = to_hex_lower(&hasher.finalize());
    if let Some(expected) = &entry.sha256 {
        if got != expected.to_lowercase() {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(anyhow!(
                "SHA-256 mismatch for {}: expected {}, got {}",
                entry.name, expected, got
            ));
        }
    } else {
        eprintln!("Warning: no SHA-256 available for {}. Skipping verification.", entry.name);
    }
    // Replace existing file safely
    if final_path.exists() {
        let _ = std::fs::remove_file(&final_path);
    }
    std::fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("Failed to move into place: {}", final_path.display()))?;
    eprintln!("Saved: {}", final_path.display());
    Ok(())
}

fn run_interactive_model_downloader() -> Result<()> {
    let models_dir = Path::new("models");
    if !models_dir.exists() { create_dir_all(models_dir).context("Failed to create models directory")?; }
    let client = Client::builder()
        .user_agent("dialogue_merger/0.1 (+https://github.com/)")
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("Failed to build HTTP client")?;

    eprintln!("Fetching online data: contacting Hugging Face to retrieve available models (this may take a moment)...");
    let models = fetch_all_models(&client)?;
    if models.is_empty() {
        eprintln!("No models found on Hugging Face listing. Please try again later.");
        return Ok(());
    }
    let selected = prompt_select_models(&models)?;
    if selected.is_empty() {
        eprintln!("No selection. Aborting download.");
        return Ok(());
    }
    for m in selected {
        if let Err(e) = download_one_model(&client, models_dir, &m) { eprintln!("Error: {:#}", e); }
    }
    Ok(())
}
