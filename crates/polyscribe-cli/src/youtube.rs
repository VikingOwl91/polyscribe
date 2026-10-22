// SPDX-License-Identifier: MIT

//! YouTube caption retrieval command.
//!
//! Uses yt-dlp as an external runtime dependency to discover caption
//! tracks and download WebVTT content. All subprocess invocations use
//! explicit argument vectors (no shell).

use anyhow::{Context, Result, anyhow};
use polyscribe_core::youtube::{
    CaptionTrack, VideoMetadata, dedup_rolling, parse_vtt, render_srt, render_txt,
    select_caption_track, validate_youtube_url,
};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

const YTDLP_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024; // 50 MiB

/// Run the youtube command.
pub fn run(
    url: String,
    language: Option<String>,
    format: String,
    output: Option<PathBuf>,
    prefer_manual: bool,
) -> Result<()> {
    // Validate URL first (no network).
    let yt_url = validate_youtube_url(&url).context("invalid YouTube URL")?;

    // Check yt-dlp is available.
    let ytdlp = find_ytdlp()?;

    // Retrieve metadata and caption track listings.
    let (meta, manual_tracks, auto_tracks) = fetch_metadata(&ytdlp, &yt_url.canonical_url)?;

    // Select caption track.
    let track = select_caption_track(
        &manual_tracks,
        &auto_tracks,
        language.as_deref(),
        prefer_manual,
    )
    .ok_or_else(|| {
        if let Some(lang) = &language {
            anyhow!("no caption track found for language '{lang}'")
        } else {
            anyhow!("no caption tracks available for this video")
        }
    })?;

    // Download the caption content (WebVTT).
    let vtt_content = download_captions(&ytdlp, &track, &yt_url.canonical_url)?;

    // Parse WebVTT.
    let doc = parse_vtt(&vtt_content).context("failed to parse WebVTT captions")?;

    // Deduplicate rolling captions.
    let cues = dedup_rolling(&doc.cues);

    // Render output.
    let rendered = match format.as_str() {
        "txt" => render_txt(&cues),
        "srt" => render_srt(&cues),
        "json" => {
            let json_out = serde_json::json!({
                "source": "youtube-captions",
                "video_id": yt_url.video_id,
                "title": meta.title,
                "uploader": meta.uploader,
                "caption_language": track.language,
                "is_automatic": track.is_automatic,
                "is_translated": track.is_translated,
                "cue_count": cues.len(),
                "cues": cues,
            });
            serde_json::to_string_pretty(&json_out).context("failed to serialize JSON")?
        }
        other => return Err(anyhow!("unsupported format: {other}")),
    };

    // Write output.
    if let Some(path) = output {
        // Validate output path (no traversal).
        validate_output_path(&path)?;
        std::fs::write(&path, &rendered)
            .with_context(|| format!("failed to write output to {}", path.display()))?;
        eprintln!("Wrote {} cues to {}", cues.len(), path.display());
    } else {
        // Write to stdout.
        let mut stdout = std::io::stdout();
        stdout.write_all(rendered.as_bytes())?;
        if !rendered.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }
        stdout.flush()?;
    }

    // Report provenance to stderr.
    eprintln!(
        "source: youtube-captions, video: {}, language: {}, track: {}, cues: {}",
        yt_url.video_id,
        track.language,
        if track.is_automatic {
            "automatic"
        } else {
            "manual"
        },
        cues.len()
    );

    Ok(())
}

fn find_ytdlp() -> Result<PathBuf> {
    // Look for yt-dlp on PATH.
    let path = std::env::var("PATH").unwrap_or_default();
    for dir in path.split(':') {
        let candidate = std::path::Path::new(dir).join("yt-dlp");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    // Also check common locations.
    for candidate in [
        "/usr/bin/yt-dlp",
        "/usr/local/bin/yt-dlp",
        "/opt/yt-dlp/yt-dlp",
    ] {
        let p = std::path::PathBuf::from(candidate);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(anyhow!(
        "yt-dlp not found on PATH. Install it with: pip install yt-dlp"
    ))
}

fn fetch_metadata(
    ytdlp: &std::path::Path,
    url: &str,
) -> Result<(VideoMetadata, Vec<CaptionTrack>, Vec<CaptionTrack>)> {
    // Use yt-dlp to fetch metadata and caption track listings.
    // --skip-download: no media download
    // --print: output JSON fields
    let output = run_ytdlp(
        ytdlp,
        &[
            "--no-playlist",
            "--skip-download",
            "--print",
            "%(subtitles)j",
            "--print",
            "%(automatic_captions)j",
            "--print",
            "%(id)s",
            "--print",
            "%(title)s",
            "--print",
            "%(uploader)s",
            "--print",
            "%(duration)s",
            url,
        ],
    )?;

    let lines: Vec<&str> = output.lines().collect();
    // First two lines are subtitles and automatic_captions JSON.
    // Remaining lines are metadata fields.
    let subs_json = lines.first().copied().unwrap_or("{}");
    let auto_json = lines.get(1).copied().unwrap_or("{}");

    let subs: serde_json::Value = serde_json::from_str(subs_json).unwrap_or(serde_json::json!({}));
    let auto: serde_json::Value = serde_json::from_str(auto_json).unwrap_or(serde_json::json!({}));

    let manual_tracks = parse_tracks(&subs, false);
    let auto_tracks = parse_tracks(&auto, true);

    // Parse metadata fields.
    let id = lines.get(2).copied().unwrap_or("").to_string();
    let title = lines
        .get(3)
        .copied()
        .filter(|s| *s != "NA" && !s.is_empty())
        .map(String::from);
    let uploader = lines
        .get(4)
        .copied()
        .filter(|s| *s != "NA" && !s.is_empty())
        .map(String::from);
    let duration = lines.get(5).copied().and_then(|s| s.parse::<f64>().ok());

    let meta = VideoMetadata {
        id,
        title,
        uploader,
        duration,
        subtitles: subs,
        automatic_captions: auto,
    };

    Ok((meta, manual_tracks, auto_tracks))
}

fn parse_tracks(json: &serde_json::Value, is_automatic: bool) -> Vec<CaptionTrack> {
    let mut tracks = Vec::new();
    if let Some(obj) = json.as_object() {
        for (lang, formats) in obj {
            if let Some(arr) = formats.as_array() {
                for fmt in arr {
                    let url = fmt.get("url").and_then(|v| v.as_str()).map(String::from);
                    let ext = fmt.get("ext").and_then(|v| v.as_str()).map(String::from);
                    let name = fmt.get("name").and_then(|v| v.as_str()).map(String::from);
                    // Only keep VTT tracks to match YTT's pick_vtt preference.
                    if ext.as_deref() != Some("vtt") {
                        continue;
                    }
                    // Detect translated tracks: URL contains tlang= parameter.
                    let is_translated = url.as_ref().map(|u| u.contains("tlang=")).unwrap_or(false);
                    tracks.push(CaptionTrack {
                        language: lang.clone(),
                        name,
                        url,
                        ext,
                        is_automatic,
                        is_translated,
                    });
                }
            }
        }
    }
    tracks
}

fn download_captions(
    ytdlp: &std::path::Path,
    track: &CaptionTrack,
    video_url: &str,
) -> Result<String> {
    let tmp_dir = std::env::temp_dir().join(format!("polyscribe-yt-{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).context("failed to create temporary directory")?;
    let output_template = tmp_dir.join("caption").to_string_lossy().to_string();

    let mut args: Vec<String> = vec![
        "--no-playlist".to_string(),
        "--skip-download".to_string(),
        "--write-sub".to_string(),
        "--sub-format".to_string(),
        "vtt".to_string(),
        "-o".to_string(),
        output_template,
        "--sub-lang".to_string(),
        track.language.clone(),
    ];

    if track.is_automatic {
        args.push("--write-auto-sub".to_string());
    }

    args.push(video_url.to_string());

    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let _output = run_ytdlp(ytdlp, &args_refs)?;

    // Find the .vtt file in the temp directory.
    let mut vtt_path = None;
    for entry in std::fs::read_dir(&tmp_dir).context("failed to read temp dir")? {
        let entry = entry.context("failed to read dir entry")?;
        let path = entry.path();
        if path.extension().map(|e| e == "vtt").unwrap_or(false) {
            vtt_path = Some(path);
            break;
        }
    }

    let vtt_path = vtt_path.ok_or_else(|| anyhow!("yt-dlp did not produce a VTT file"))?;
    let content = std::fs::read_to_string(&vtt_path)
        .with_context(|| format!("failed to read caption file: {}", vtt_path.display()))?;

    // Clean up.
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(content)
}

fn run_ytdlp(ytdlp: &std::path::Path, args: &[&str]) -> Result<String> {
    let mut cmd = Command::new(ytdlp);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().context("failed to spawn yt-dlp")?;

    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    // Read stdout and stderr in separate threads to avoid pipe deadlock.
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut out) = stdout_handle {
            use std::io::Read;
            out.read_to_end(&mut buf).ok();
        }
        buf
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut err) = stderr_handle {
            use std::io::Read;
            err.read_to_end(&mut buf).ok();
        }
        buf
    });

    // Wait with timeout.
    use std::time::Instant;
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() > YTDLP_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(anyhow!(
                        "yt-dlp timed out after {}s",
                        YTDLP_TIMEOUT.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(anyhow!("failed to wait for yt-dlp: {e}"));
            }
        }
    };

    let stdout_bytes = stdout_thread.join().unwrap_or_default();
    let stderr_bytes = stderr_thread.join().unwrap_or_default();

    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr_bytes);
        let stderr = if stderr.len() > 4096 {
            &stderr[..4096]
        } else {
            &stderr
        };
        return Err(anyhow!("yt-dlp failed: {stderr}"));
    }

    if stdout_bytes.len() > MAX_OUTPUT_BYTES {
        return Err(anyhow!("yt-dlp output too large"));
    }

    let stdout = String::from_utf8_lossy(&stdout_bytes);
    let stderr = String::from_utf8_lossy(&stderr_bytes);
    if !stderr.is_empty() && polyscribe_core::verbose_level() > 0 {
        eprintln!("yt-dlp stderr: {}", &stderr[..stderr.len().min(512)]);
    }

    Ok(stdout.to_string())
}

fn validate_output_path(path: &std::path::Path) -> Result<()> {
    // Reject paths with traversal or control characters.
    let path_str = path.to_string_lossy();
    if path_str.contains("..") {
        return Err(anyhow!("output path must not contain '..'"));
    }
    for ch in path_str.chars() {
        if ch.is_control() {
            return Err(anyhow!("output path must not contain control characters"));
        }
    }
    Ok(())
}
