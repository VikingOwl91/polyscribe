use std::fs::{File, create_dir_all};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::env;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use chrono::Local;
use std::sync::atomic::{AtomicBool, Ordering};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

mod models;

static LAST_MODEL_WRITTEN: AtomicBool = AtomicBool::new(false);

fn models_dir_path() -> PathBuf {
    // Highest priority: explicit override
    if let Ok(p) = env::var("POLYSCRIBE_MODELS_DIR") {
        let pb = PathBuf::from(p);
        if !pb.as_os_str().is_empty() {
            return pb;
        }
    }
    // In debug builds, keep local ./models for convenience
    if cfg!(debug_assertions) {
        return PathBuf::from("models");
    }
    // In release builds, choose a user-writable data directory
    if let Ok(xdg) = env::var("XDG_DATA_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("polyscribe").join("models");
        }
    }
    if let Ok(home) = env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("polyscribe")
                .join("models");
        }
    }
    // Last resort fallback
    PathBuf::from("models")
}

#[derive(Parser, Debug)]
#[command(name = "PolyScribe", version, about = "Merge multiple JSON transcripts into one or transcribe audio using native whisper")]
struct Args {
    /// Input .json transcript files or audio files to merge/transcribe
    inputs: Vec<String>,

    /// Output file path base (date prefix will be added); if omitted, writes JSON to stdout
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,

    /// Merge all inputs into a single output; if not set, each input is written as a separate output
    #[arg(short = 'm', long = "merge")]
    merge: bool,

    /// Merge and also write separate outputs per input; requires -o OUTPUT_DIR
    #[arg(long = "merge-and-separate")]
    merge_and_separate: bool,

    /// Language code to use for transcription (e.g., en, de). No auto-detection.
    #[arg(short, long, value_name = "LANG")]
    language: Option<String>,

    /// Launch interactive model downloader (list HF models, multi-select and download)
    #[arg(long)]
    download_models: bool,

    /// Update local Whisper models by comparing hashes/sizes with remote manifest
    #[arg(long)]
    update_models: bool,
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

#[derive(Debug, Serialize, Clone)]
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



fn find_model_file() -> Result<PathBuf> {
    let models_dir_buf = models_dir_path();
    let models_dir = models_dir_buf.as_path();
    if !models_dir.exists() {
        create_dir_all(models_dir).with_context(|| format!("Failed to create models directory: {}", models_dir.display()))?;
    }

    // If env var WHISPER_MODEL is set and valid, prefer it
    if let Ok(env_model) = env::var("WHISPER_MODEL") {
        let p = PathBuf::from(env_model);
        if p.is_file() {
            // persist selection
            let _ = std::fs::write(models_dir.join(".last_model"), p.display().to_string());
            LAST_MODEL_WRITTEN.store(true, Ordering::Relaxed);
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
            if let Err(e) = models::run_interactive_model_downloader() {
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
        LAST_MODEL_WRITTEN.store(true, Ordering::Relaxed);
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
    LAST_MODEL_WRITTEN.store(true, Ordering::Relaxed);
    Ok(chosen)
}

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

struct LastModelCleanup {
    path: PathBuf,
}
impl Drop for LastModelCleanup {
    fn drop(&mut self) {
        // Ensure .last_model does not persist across program runs
        let _ = std::fs::remove_file(&self.path);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Defer cleanup of .last_model until program exit
    let models_dir_buf = models_dir_path();
    let last_model_path = models_dir_buf.join(".last_model");
    // Ensure cleanup at end of program, regardless of exit path
    let _last_model_cleanup = LastModelCleanup { path: last_model_path.clone() };

    // If requested, run the interactive model downloader first. If no inputs were provided, exit after downloading.
    if args.download_models {
        if let Err(e) = models::run_interactive_model_downloader() {
            eprintln!("Model downloader failed: {:#}", e);
        }
        if args.inputs.is_empty() {
            return Ok(());
        }
    }

    // If requested, update local models and exit unless inputs provided to continue
    if args.update_models {
        if let Err(e) = models::update_local_models() {
            eprintln!("Model update failed: {:#}", e);
            return Err(e);
        }
        // if only updating models and no inputs, exit
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

    if args.merge_and_separate {
        // Combined mode: write separate outputs per input and also a merged output set
        // Require an output directory
        let out_dir = match output_path.as_ref() {
            Some(p) => PathBuf::from(p),
            None => return Err(anyhow!("--merge-and-separate requires -o OUTPUT_DIR")),
        };
        if !out_dir.as_os_str().is_empty() {
            create_dir_all(&out_dir)
                .with_context(|| format!("Failed to create output directory: {}", out_dir.display()))?;
        }

        let mut merged_entries: Vec<OutputEntry> = Vec::new();

        for input_path in &inputs {
            let path = Path::new(input_path);
            let speaker = sanitize_speaker_name(
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("speaker")
            );

            // Collect entries per file and extend merged
            let mut entries: Vec<OutputEntry> = Vec::new();
            if is_audio_file(path) {
                let items = transcribe_native(path, &speaker, lang_hint.as_deref())?;
                entries.extend(items.into_iter());
            } else if is_json_file(path) {
                let mut buf = String::new();
                File::open(path)
                    .with_context(|| format!("Failed to open: {}", input_path))?
                    .read_to_string(&mut buf)
                    .with_context(|| format!("Failed to read: {}", input_path))?;
                let root: InputRoot = serde_json::from_str(&buf)
                    .with_context(|| format!("Invalid JSON transcript parsed from {}", input_path))?;
                for seg in root.segments {
                    entries.push(OutputEntry { id: 0, speaker: speaker.clone(), start: seg.start, end: seg.end, text: seg.text });
                }
            } else {
                return Err(anyhow!(format!("Unsupported input type (expected .json or audio media): {}", input_path)));
            }

            // Sort and reassign ids per file
            entries.sort_by(|a, b| {
                match a.start.partial_cmp(&b.start) { Some(std::cmp::Ordering::Equal) | None => {} Some(o) => return o }
                a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, e) in entries.iter_mut().enumerate() { e.id = i as u64; }

            // Write separate outputs to out_dir
            let out = OutputRoot { items: entries.clone() };
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            let date = date_prefix();
            let base_name = format!("{}_{}", date, stem);
            let json_path = out_dir.join(format!("{}.json", &base_name));
            let toml_path = out_dir.join(format!("{}.toml", &base_name));
            let srt_path = out_dir.join(format!("{}.srt", &base_name));

            let mut json_file = File::create(&json_path)
                .with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
            serde_json::to_writer_pretty(&mut json_file, &out)?; writeln!(&mut json_file)?;

            let toml_str = toml::to_string_pretty(&out)?;
            let mut toml_file = File::create(&toml_path)
                .with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
            toml_file.write_all(toml_str.as_bytes())?; if !toml_str.ends_with('\n') { writeln!(&mut toml_file)?; }

            let srt_str = render_srt(&out.items);
            let mut srt_file = File::create(&srt_path)
                .with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
            srt_file.write_all(srt_str.as_bytes())?;

            // Extend merged with per-file entries
            merged_entries.extend(out.items.into_iter());
        }

        // Now write merged output set into out_dir
        merged_entries.sort_by(|a, b| {
            match a.start.partial_cmp(&b.start) { Some(std::cmp::Ordering::Equal) | None => {} Some(o) => return o }
            a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, e) in merged_entries.iter_mut().enumerate() { e.id = i as u64; }
        let merged_out = OutputRoot { items: merged_entries };

        let date = date_prefix();
        let merged_base = format!("{}_merged", date);
        let m_json = out_dir.join(format!("{}.json", &merged_base));
        let m_toml = out_dir.join(format!("{}.toml", &merged_base));
        let m_srt = out_dir.join(format!("{}.srt", &merged_base));

        let mut mj = File::create(&m_json)
            .with_context(|| format!("Failed to create output file: {}", m_json.display()))?;
        serde_json::to_writer_pretty(&mut mj, &merged_out)?; writeln!(&mut mj)?;

        let m_toml_str = toml::to_string_pretty(&merged_out)?;
        let mut mt = File::create(&m_toml)
            .with_context(|| format!("Failed to create output file: {}", m_toml.display()))?;
        mt.write_all(m_toml_str.as_bytes())?; if !m_toml_str.ends_with('\n') { writeln!(&mut mt)?; }

        let m_srt_str = render_srt(&merged_out.items);
        let mut ms = File::create(&m_srt)
            .with_context(|| format!("Failed to create output file: {}", m_srt.display()))?;
        ms.write_all(m_srt_str.as_bytes())?;
    } else if args.merge {
        // MERGED MODE (previous default)
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
                let items = transcribe_native(path, &speaker, lang_hint.as_deref())?;
                for e in items { entries.push(e); }
                continue;
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
                    id: 0,
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
        for (i, e) in entries.iter_mut().enumerate() { e.id = i as u64; }
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

            let mut json_file = File::create(&json_path)
                .with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
            serde_json::to_writer_pretty(&mut json_file, &out)?; writeln!(&mut json_file)?;

            let toml_str = toml::to_string_pretty(&out)?;
            let mut toml_file = File::create(&toml_path)
                .with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
            toml_file.write_all(toml_str.as_bytes())?; if !toml_str.ends_with('\n') { writeln!(&mut toml_file)?; }

            let srt_str = render_srt(&out.items);
            let mut srt_file = File::create(&srt_path)
                .with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
            srt_file.write_all(srt_str.as_bytes())?;
        } else {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            serde_json::to_writer_pretty(&mut handle, &out)?; writeln!(&mut handle)?;
        }
    } else {
        // SEPARATE MODE (default now)
        // If writing to stdout with multiple inputs, not supported
        if output_path.is_none() && inputs.len() > 1 {
            return Err(anyhow!("Multiple inputs without --merge require -o OUTPUT_DIR to write separate files"));
        }

        // If output_path is provided, treat it as a directory. Create it.
        let out_dir: Option<PathBuf> = output_path.as_ref().map(|p| PathBuf::from(p));
        if let Some(dir) = &out_dir {
            if !dir.as_os_str().is_empty() {
                create_dir_all(dir).with_context(|| format!("Failed to create output directory: {}", dir.display()))?;
            }
        }

        for input_path in &inputs {
            let path = Path::new(input_path);
            let speaker = sanitize_speaker_name(
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("speaker")
            );

            // Collect entries per file
            let mut entries: Vec<OutputEntry> = Vec::new();
            if is_audio_file(path) {
                let items = transcribe_native(path, &speaker, lang_hint.as_deref())?;
                entries.extend(items);
            } else if is_json_file(path) {
                let mut buf = String::new();
                File::open(path)
                    .with_context(|| format!("Failed to open: {}", input_path))?
                    .read_to_string(&mut buf)
                    .with_context(|| format!("Failed to read: {}", input_path))?;
                let root: InputRoot = serde_json::from_str(&buf)
                    .with_context(|| format!("Invalid JSON transcript parsed from {}", input_path))?;
                for seg in root.segments {
                    entries.push(OutputEntry { id: 0, speaker: speaker.clone(), start: seg.start, end: seg.end, text: seg.text });
                }
            } else {
                return Err(anyhow!(format!("Unsupported input type (expected .json or audio media): {}", input_path)));
            }

            // Sort and reassign ids per file
            entries.sort_by(|a, b| {
                match a.start.partial_cmp(&b.start) { Some(std::cmp::Ordering::Equal) | None => {} Some(o) => return o }
                a.end.partial_cmp(&b.end).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, e) in entries.iter_mut().enumerate() { e.id = i as u64; }
            let out = OutputRoot { items: entries };

            if let Some(dir) = &out_dir {
                // Build file names using input stem
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
                let date = date_prefix();
                let base_name = format!("{}_{}", date, stem);
                let json_path = dir.join(format!("{}.json", &base_name));
                let toml_path = dir.join(format!("{}.toml", &base_name));
                let srt_path = dir.join(format!("{}.srt", &base_name));

                let mut json_file = File::create(&json_path)
                    .with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
                serde_json::to_writer_pretty(&mut json_file, &out)?; writeln!(&mut json_file)?;

                let toml_str = toml::to_string_pretty(&out)?;
                let mut toml_file = File::create(&toml_path)
                    .with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
                toml_file.write_all(toml_str.as_bytes())?; if !toml_str.ends_with('\n') { writeln!(&mut toml_file)?; }

                let srt_str = render_srt(&out.items);
                let mut srt_file = File::create(&srt_path)
                    .with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
                srt_file.write_all(srt_str.as_bytes())?;
            } else {
                // stdout (only single input reaches here)
                let stdout = io::stdout();
                let mut handle = stdout.lock();
                serde_json::to_writer_pretty(&mut handle, &out)?; writeln!(&mut handle)?;
            }
        }
    }

    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::env as std_env;
    use clap::CommandFactory;

    #[test]
    fn test_cli_name_polyscribe() {
        let cmd = Args::command();
        assert_eq!(cmd.get_name(), "PolyScribe");
    }

    #[test]
    fn test_last_model_cleanup_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        let last = tmp.path().join(".last_model");
        fs::write(&last, "dummy").unwrap();
        {
            let _cleanup = LastModelCleanup { path: last.clone() };
        }
        assert!(!last.exists(), ".last_model should be removed on drop");
    }
    use super::*;
    use std::path::Path;

    #[test]
    fn test_format_srt_time_basic_and_rounding() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.0), "00:00:01,000");
        assert_eq!(format_srt_time(61.0), "00:01:01,000");
        assert_eq!(format_srt_time(3661.789), "01:01:01,789");
        // rounding
        assert_eq!(format_srt_time(0.0014), "00:00:00,001");
        assert_eq!(format_srt_time(0.0015), "00:00:00,002");
    }

    #[test]
    fn test_render_srt_with_and_without_speaker() {
        let items = vec![
            OutputEntry { id: 0, speaker: "Alice".to_string(), start: 0.0, end: 1.0, text: "Hello".to_string() },
            OutputEntry { id: 1, speaker: String::new(), start: 1.0, end: 2.0, text: "World".to_string() },
        ];
        let srt = render_srt(&items);
        let expected = "1\n00:00:00,000 --> 00:00:01,000\nAlice: Hello\n\n2\n00:00:01,000 --> 00:00:02,000\nWorld\n\n";
        assert_eq!(srt, expected);
    }

    #[test]
    fn test_sanitize_speaker_name() {
        assert_eq!(sanitize_speaker_name("123-bob"), "bob");
        assert_eq!(sanitize_speaker_name("00123-alice"), "alice");
        assert_eq!(sanitize_speaker_name("abc-bob"), "abc-bob");
        assert_eq!(sanitize_speaker_name("123"), "123");
        assert_eq!(sanitize_speaker_name("-bob"), "-bob");
        assert_eq!(sanitize_speaker_name("123-"), "");
    }

    #[test]
    fn test_is_json_file_and_is_audio_file() {
        assert!(is_json_file(Path::new("foo.json")));
        assert!(is_json_file(Path::new("foo.JSON")));
        assert!(!is_json_file(Path::new("foo.txt")));
        assert!(!is_json_file(Path::new("foo")));

        assert!(is_audio_file(Path::new("a.mp3")));
        assert!(is_audio_file(Path::new("b.WAV")));
        assert!(is_audio_file(Path::new("c.m4a")));
        assert!(!is_audio_file(Path::new("d.txt")));
    }

    #[test]
    fn test_normalize_lang_code() {
        assert_eq!(normalize_lang_code("en"), Some("en".to_string()));
        assert_eq!(normalize_lang_code("German"), Some("de".to_string()));
        assert_eq!(normalize_lang_code("en_US.UTF-8"), Some("en".to_string()));
        assert_eq!(normalize_lang_code("AUTO"), None);
        assert_eq!(normalize_lang_code("  \t  "), None);
        assert_eq!(normalize_lang_code("zh"), Some("zh".to_string()));
    }

    #[test]
    fn test_date_prefix_format_shape() {
        let d = date_prefix();
        assert_eq!(d.len(), 10);
        let bytes = d.as_bytes();
        assert!(bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit() && bytes[2].is_ascii_digit() && bytes[3].is_ascii_digit());
        assert_eq!(bytes[4], b'-');
        assert!(bytes[5].is_ascii_digit() && bytes[6].is_ascii_digit());
        assert_eq!(bytes[7], b'-');
        assert!(bytes[8].is_ascii_digit() && bytes[9].is_ascii_digit());
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_models_dir_path_default_debug_and_env_override() {
        // clear override
        unsafe { std_env::remove_var("POLYSCRIBE_MODELS_DIR"); }
        assert_eq!(models_dir_path(), PathBuf::from("models"));
        // override
        let tmp = tempfile::tempdir().unwrap();
        unsafe { std_env::set_var("POLYSCRIBE_MODELS_DIR", tmp.path()); }
        assert_eq!(models_dir_path(), tmp.path().to_path_buf());
        // cleanup
        unsafe { std_env::remove_var("POLYSCRIBE_MODELS_DIR"); }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_models_dir_path_default_release() {
        // Ensure override is cleared
        unsafe { std_env::remove_var("POLYSCRIBE_MODELS_DIR"); }
        // Prefer XDG_DATA_HOME when set
        let tmp_xdg = tempfile::tempdir().unwrap();
        unsafe {
            std_env::set_var("XDG_DATA_HOME", tmp_xdg.path());
            std_env::remove_var("HOME");
        }
        assert_eq!(models_dir_path(), tmp_xdg.path().join("polyscribe").join("models"));
        // Else fall back to HOME/.local/share
        let tmp_home = tempfile::tempdir().unwrap();
        unsafe {
            std_env::remove_var("XDG_DATA_HOME");
            std_env::set_var("HOME", tmp_home.path());
        }
        assert_eq!(models_dir_path(), tmp_home.path().join(".local").join("share").join("polyscribe").join("models"));
        // Cleanup
        unsafe {
            std_env::remove_var("XDG_DATA_HOME");
            std_env::remove_var("HOME");
        }
    }
}
