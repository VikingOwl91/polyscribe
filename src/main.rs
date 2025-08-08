use std::fs::{File, create_dir_all};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use chrono::Local;

#[derive(Parser, Debug)]
#[command(name = "merge_transcripts", version, about = "Merge multiple JSON transcripts into one")]
struct Args {
    /// Input JSON files to merge
    #[arg(required = true)]
    inputs: Vec<String>,

    /// Output file path (if omitted, writes to stdout)
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,
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

// --- Helpers for audio transcription via openai-whisper ---
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

fn ensure_whisper_venv() -> Result<PathBuf> {
    let venv_dir = Path::new(".venv");
    let bin_dir = venv_dir.join("bin");
    let python_bin = bin_dir.join("python");
    // Create venv if needed
    if !python_bin.exists() {
        let status = Command::new("python")
            .arg("-m").arg("venv").arg(".venv")
            .status()
            .with_context(|| "Failed to execute: python -m venv .venv")?;
        if !status.success() {
            return Err(anyhow!("Creating virtual environment failed (python -m venv .venv)"));
        }
    }
    // Install openai-whisper if needed
    let whisper_bin = bin_dir.join("whisper");
    if !whisper_bin.exists() {
        let status = Command::new(&python_bin)
            .arg("-m").arg("pip").arg("install").arg("-U").arg("openai-whisper")
            .status()
            .with_context(|| "Failed to execute: .venv/bin/python -m pip install -U openai-whisper")?;
        if !status.success() {
            return Err(anyhow!("Installing openai-whisper failed"));
        }
    }
    Ok(whisper_bin)
}

fn run_whisper(audio_path: &Path, out_dir: &Path, whisper_bin: &Path) -> Result<PathBuf> {
    create_dir_all(out_dir).with_context(|| format!("Failed to create whisper output dir: {}", out_dir.display()))?;
    let out_dir_str = out_dir.to_string_lossy().to_string();

    // Try with -o first (as per requested command)
    let attempt = |use_short_o: bool| -> Result<()> {
        let mut cmd = Command::new(whisper_bin);
        cmd.arg(audio_path)
            .arg("--model").arg("turbo")
            .arg("--output_format").arg("json");
        if use_short_o {
            cmd.arg("-o").arg(&out_dir_str);
        } else {
            cmd.arg("--output_dir").arg(&out_dir_str);
        }
        let output = cmd.output()
            .with_context(|| format!("Failed to execute whisper for {}", audio_path.display()))?;
        if !output.status.success() {
            return Err(anyhow!(
                "Whisper transcription failed for {}: {}",
                audio_path.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(())
    };

    if let Err(_e) = attempt(true) {
        // Fallback to --output_dir if -o not supported by CLI installed
        attempt(false)?;
    }

    // Try to locate the resulting JSON file
    let stem = audio_path.file_stem().and_then(|s| s.to_str()).unwrap_or("transcript");
    let cand1 = out_dir.join(format!("{}.json", stem));
    if cand1.exists() { return Ok(cand1); }
    if let Some(file_name) = audio_path.file_name().and_then(|s| s.to_str()) {
        let cand2 = out_dir.join(format!("{}.json", file_name));
        if cand2.exists() { return Ok(cand2); }
    }
    Err(anyhow!("Could not find whisper JSON output in {} for input {}", out_dir.display(), audio_path.display()))
}

fn main() -> Result<()> {
    let args = Args::parse();

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

    // Determine directory for intermediate Whisper transcriptions
    let transcripts_dir: PathBuf = if let Some(ref out_path) = output_path {
        let base_path = Path::new(out_path);
        if let Some(parent) = base_path.parent() {
            if parent.as_os_str().is_empty() {
                PathBuf::from("output").join("transcripts")
            } else {
                parent.join("transcripts")
            }
        } else {
            PathBuf::from("output").join("transcripts")
        }
    } else {
        PathBuf::from("output").join("transcripts")
    };
    let mut whisper_bin_opt: Option<PathBuf> = None;

    let mut entries: Vec<OutputEntry> = Vec::new();

    for input_path in &inputs {
        let path = Path::new(input_path);
        let speaker = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("speaker")
            .to_string();

        let mut buf = String::new();
        if is_audio_file(path) {
            // Ensure whisper is available
            if whisper_bin_opt.is_none() {
                whisper_bin_opt = Some(ensure_whisper_venv()?);
            }
            let whisper_bin = whisper_bin_opt.as_ref().unwrap();
            let json_path = run_whisper(path, &transcripts_dir, whisper_bin)?;
            File::open(&json_path)
                .with_context(|| format!("Failed to open Whisper JSON: {}", json_path.display()))?
                .read_to_string(&mut buf)
                .with_context(|| format!("Failed to read Whisper JSON: {}", json_path.display()))?;
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
