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

#[derive(Parser, Debug)]
#[command(name = "PolyScribe", version, about = "Merge multiple JSON transcripts into one or transcribe audio using native whisper")]
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
        if LAST_MODEL_WRITTEN.load(Ordering::Relaxed) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Defer cleanup of .last_model until program exit (after all runs within this process)
    let models_dir = Path::new("models");
    let last_model_path = models_dir.join(".last_model");
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
            let items = transcribe_native(path, &speaker, lang_hint.as_deref())?;
            for e in items {
                entries.push(e);
            }
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

