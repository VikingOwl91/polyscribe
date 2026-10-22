// SPDX-License-Identifier: MIT

//! YouTube caption retrieval, parsing, and rendering.
//!
//! This module provides the data types and pure functions for working with
//! YouTube captions: URL validation, WebVTT parsing, rolling-caption
//! deduplication, and output rendering. Network access and subprocess
//! orchestration live in the CLI layer.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// A validated YouTube URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YouTubeUrl {
    pub video_id: String,
    pub canonical_url: String,
}

/// Validate and extract the video ID from a YouTube URL.
///
/// Only HTTPS URLs on known YouTube hosts are accepted. Credentials,
/// query parameters that look like option injection, and non-YouTube
/// hosts are rejected.
pub fn validate_youtube_url(input: &str) -> Result<YouTubeUrl> {
    let input = input.trim();
    if input.is_empty() {
        return Err(anyhow!("the URL must not be empty"));
    }

    // Reject anything that is not an HTTPS URL.
    if !input.starts_with("https://") {
        return Err(anyhow!("only HTTPS URLs are supported"));
    }

    // Parse the URL manually to avoid pulling in a URL crate.
    let after_scheme = &input["https://".len()..];
    let (host, rest) = match after_scheme.find('/') {
        Some(idx) => (&after_scheme[..idx], &after_scheme[idx..]),
        None => (after_scheme, "/"),
    };

    // Strip port if present.
    let host = host.split(':').next().unwrap_or(host);

    // YouTube host allowlist.
    let allowed_hosts = [
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    ];
    if !allowed_hosts.contains(&host) {
        return Err(anyhow!("unsupported host: {host}"));
    }

    // Reject credential-bearing URLs.
    if rest.contains('@') {
        return Err(anyhow!("credential-bearing URLs are not supported"));
    }

    // Extract video ID.
    let video_id = if host == "youtu.be" || host == "www.youtu.be" {
        // https://youtu.be/VIDEO_ID
        let path = rest.trim_start_matches('/');
        let id = path.split('?').next().unwrap_or(path);
        let id = id.split('&').next().unwrap_or(id);
        id.to_string()
    } else {
        // https://www.youtube.com/watch?v=VIDEO_ID
        // or https://www.youtube.com/embed/VIDEO_ID
        if rest.starts_with("/watch") {
            // Extract v= parameter.
            let query = rest.find('?').map(|i| &rest[i + 1..]).unwrap_or("");
            let mut id = None;
            for param in query.split('&') {
                if let Some(val) = param.strip_prefix("v=") {
                    id = Some(val.to_string());
                    break;
                }
            }
            id.ok_or_else(|| anyhow!("missing video ID in URL"))?
        } else if let Some(rest) = rest.strip_prefix("/embed/") {
            let id = rest.split('?').next().unwrap_or(rest);
            let id = id.split('&').next().unwrap_or(id);
            id.to_string()
        } else if let Some(rest) = rest.strip_prefix("/shorts/") {
            let id = rest.split('?').next().unwrap_or(rest);
            let id = id.split('&').next().unwrap_or(id);
            id.to_string()
        } else {
            return Err(anyhow!("unsupported YouTube URL path"));
        }
    };

    // Validate video ID format (alphanumeric, underscore, hyphen, 11 chars typical).
    if video_id.is_empty() || video_id.len() > 32 {
        return Err(anyhow!("invalid video ID"));
    }
    if !video_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(anyhow!("invalid video ID characters"));
    }

    let canonical_url = format!("https://www.youtube.com/watch?v={video_id}");
    Ok(YouTubeUrl {
        video_id,
        canonical_url,
    })
}

/// A caption cue with start/end times and text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionCue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// A parsed WebVTT file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VttDocument {
    pub cues: Vec<CaptionCue>,
}

/// Parse a WebVTT string into cues.
pub fn parse_vtt(input: &str) -> Result<VttDocument> {
    let mut cues = Vec::new();
    // Normalize line endings.
    let input = input.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;

    // Skip header.
    while i < lines.len() && !lines[i].trim().is_empty() {
        i += 1;
    }
    // Skip blank lines after header.
    while i < lines.len() && lines[i].trim().is_empty() {
        i += 1;
    }

    while i < lines.len() {
        // Skip blank lines between cues.
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }

        // Optional cue identifier.
        let mut line = lines[i].trim();
        if !line.contains("-->") {
            // This might be a cue ID; skip it.
            i += 1;
            if i >= lines.len() {
                break;
            }
            line = lines[i].trim();
        }

        // Parse timestamp line.
        if !line.contains("-->") {
            // Malformed: skip.
            i += 1;
            continue;
        }

        let parts: Vec<&str> = line.split("-->").collect();
        if parts.len() < 2 {
            i += 1;
            continue;
        }

        let start = parse_vtt_timestamp(parts[0].trim());
        let end_part = parts[1].trim();
        // End timestamp may have settings after it.
        let end_str = end_part.split_whitespace().next().unwrap_or(end_part);
        let end = parse_vtt_timestamp(end_str);

        if start.is_none() || end.is_none() {
            i += 1;
            continue;
        }
        let start_ms = start.unwrap();
        let end_ms = end.unwrap();

        // Collect cue text (until blank line).
        i += 1;
        let mut text_lines = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() {
            text_lines.push(lines[i]);
            i += 1;
        }

        let raw_text = text_lines.join("\n");
        let text = clean_vtt_text(&raw_text);

        if !text.is_empty() {
            cues.push(CaptionCue {
                start_ms,
                end_ms: end_ms.max(start_ms),
                text,
            });
        }
    }

    Ok(VttDocument { cues })
}

fn parse_vtt_timestamp(s: &str) -> Option<u64> {
    // Formats: HH:MM:SS.mmm or MM:SS.mmm
    let s = s.trim();
    let parts: Vec<&str> = s.split(':').collect();
    let (hours, mins, secs) = match parts.len() {
        3 => {
            let h: u64 = parts[0].parse().ok()?;
            let m: u64 = parts[1].parse().ok()?;
            let s: f64 = parts[2].parse().ok()?;
            (h, m, s)
        }
        2 => {
            let m: u64 = parts[0].parse().ok()?;
            let s: f64 = parts[1].parse().ok()?;
            (0, m, s)
        }
        _ => return None,
    };
    let total_ms = hours * 3_600_000 + mins * 60_000 + (secs * 1000.0).round() as u64;
    Some(total_ms)
}

fn clean_vtt_text(raw: &str) -> String {
    // Remove HTML tags, voice tags, and normalize whitespace.
    let mut result = String::new();
    let mut in_tag = false;
    for ch in raw.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    // Decode common HTML entities.
    let result = result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'");
    // Normalize whitespace: collapse multiple spaces/newlines.
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Deduplicate rolling captions.
///
/// YouTube automatic captions often progressively extend earlier text:
/// "The quick" -> "The quick brown" -> "The quick brown fox".
/// This function keeps only the final version of each rolling sequence.
pub fn dedup_rolling(cues: &[CaptionCue]) -> Vec<CaptionCue> {
    if cues.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<CaptionCue> = Vec::new();

    for cue in cues {
        if let Some(last) = result.last_mut() {
            // Check if this cue is a rolling extension of the last one.
            let last_text = last.text.to_lowercase();
            let curr_text = cue.text.to_lowercase();

            if curr_text.starts_with(&last_text) && curr_text.len() > last_text.len() {
                // Rolling extension: replace the last cue's text and end time.
                last.text = cue.text.clone();
                last.end_ms = cue.end_ms.max(last.end_ms);
                continue;
            }

            // Check if this cue is an exact duplicate at the same time.
            // Identical text at different timestamps is legitimate repetition
            // and must be kept.
            if curr_text == last_text && cue.start_ms == last.start_ms && cue.end_ms == last.end_ms
            {
                continue;
            }
        }
        result.push(cue.clone());
    }

    result
}

/// Metadata retrieved from yt-dlp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub id: String,
    pub title: Option<String>,
    pub uploader: Option<String>,
    pub duration: Option<f64>,
    pub subtitles: serde_json::Value,
    pub automatic_captions: serde_json::Value,
}

/// Caption track information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionTrack {
    pub language: String,
    pub name: Option<String>,
    pub url: Option<String>,
    pub ext: Option<String>,
    pub is_automatic: bool,
    pub is_translated: bool,
}

/// Select the best caption track given available tracks and preferences.
pub fn select_caption_track(
    manual: &[CaptionTrack],
    automatic: &[CaptionTrack],
    language: Option<&str>,
    prefer_manual: bool,
) -> Option<CaptionTrack> {
    // Filter out live-chat and translated tracks unless explicitly requested.
    let manual: Vec<&CaptionTrack> = manual
        .iter()
        .filter(|t| {
            if t.language.to_lowercase() == "live_chat" {
                return false;
            }
            if t.is_translated && language.is_none() {
                return false;
            }
            true
        })
        .collect();
    let automatic: Vec<&CaptionTrack> = automatic
        .iter()
        .filter(|t| {
            if t.language.to_lowercase() == "live_chat" {
                return false;
            }
            if t.is_translated && language.is_none() {
                return false;
            }
            true
        })
        .collect();

    let lang_norm = |code: &str| -> String {
        code.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase()
    };

    // 1. Manual track matching explicitly requested language.
    if let Some(lang) = language {
        let want = lang_norm(lang);
        for track in &manual {
            if lang_norm(&track.language) == want
                || lang_norm(track.name.as_deref().unwrap_or("")) == want
            {
                return Some((*track).clone());
            }
        }
    }

    // 2. Single unambiguous manual track (no language specified).
    if language.is_none() && manual.len() == 1 {
        return Some((*manual[0]).clone());
    }

    // 3. Automatic ASR track: requested language if given.
    if let Some(lang) = language {
        let want = lang_norm(lang);
        for track in &automatic {
            if !track.is_translated && lang_norm(&track.language) == want {
                return Some((*track).clone());
            }
        }
        // 3b. Translated automatic track, only if explicitly requested.
        for track in &automatic {
            if track.is_translated && lang_norm(&track.language) == want {
                return Some((*track).clone());
            }
        }
        // No match: do not substitute.
        return None;
    }

    // 4. Automatic: unique source track (no tlang/translated).
    let auto_sources: Vec<&CaptionTrack> = automatic
        .iter()
        .filter(|t| !t.is_translated)
        .copied()
        .collect();
    if auto_sources.len() == 1 {
        return Some((*auto_sources[0]).clone());
    }

    // 5. Default selection: prefer English, then first available.
    // Manual tracks first (if prefer_manual).
    if prefer_manual && !manual.is_empty() {
        // Prefer English.
        for track in &manual {
            if track.language.to_lowercase().starts_with("en") {
                return Some((*track).clone());
            }
        }
        // Fall back to first manual track.
        return Some((*manual[0]).clone());
    }

    // 6. Automatic tracks: prefer English source, then first source.
    if !automatic.is_empty() {
        for track in &automatic {
            if !track.is_translated && track.language.to_lowercase().starts_with("en") {
                return Some((*track).clone());
            }
        }
        // Fall back to first non-translated auto track.
        if let Some(t) = auto_sources.first() {
            return Some((**t).clone());
        }
        // Last resort: first auto track.
        return Some((*automatic[0]).clone());
    }

    None
}

/// Render cues as plain text.
pub fn render_txt(cues: &[CaptionCue]) -> String {
    cues.iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render cues as SRT.
pub fn render_srt(cues: &[CaptionCue]) -> String {
    let mut out = String::new();
    for (i, cue) in cues.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(cue.start_ms),
            format_srt_time(cue.end_ms)
        ));
        out.push_str(&cue.text);
        out.push_str("\n\n");
    }
    out
}

fn format_srt_time(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, millis)
}

/// Sanitize a string for use as a filename.
pub fn sanitize_filename(input: &str) -> String {
    let mut result = String::new();
    for ch in input.chars() {
        if ch.is_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            result.push(ch);
        } else if ch == ' ' {
            result.push('_');
        } else {
            // Skip other characters.
        }
    }
    // Remove leading dots.
    while result.starts_with('.') {
        result.remove(0);
    }
    if result.is_empty() {
        result = "transcript".to_string();
    }
    // Limit length.
    if result.len() > 100 {
        result.truncate(100);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_youtube_url_watch() {
        let url = validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ").unwrap();
        assert_eq!(url.video_id, "dQw4w9WgXcQ");
    }

    #[test]
    fn test_validate_youtube_url_youtu_be() {
        let url = validate_youtube_url("https://youtu.be/dQw4w9WgXcQ").unwrap();
        assert_eq!(url.video_id, "dQw4w9WgXcQ");
    }

    #[test]
    fn test_validate_youtube_url_rejects_http() {
        assert!(validate_youtube_url("http://www.youtube.com/watch?v=dQw4w9WgXcQ").is_err());
    }

    #[test]
    fn test_validate_youtube_url_rejects_non_youtube() {
        assert!(validate_youtube_url("https://example.com/watch?v=dQw4w9WgXcQ").is_err());
    }

    #[test]
    fn test_validate_youtube_url_rejects_injection() {
        assert!(
            validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ;rm -rf /").is_err()
        );
    }

    #[test]
    fn test_parse_vtt_basic() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world\n";
        let doc = parse_vtt(vtt).unwrap();
        assert_eq!(doc.cues.len(), 1);
        assert_eq!(doc.cues[0].text, "Hello world");
        assert_eq!(doc.cues[0].start_ms, 1000);
        assert_eq!(doc.cues[0].end_ms, 3000);
    }

    #[test]
    fn test_parse_vtt_rolling() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nThe quick\n\n00:00:02.000 --> 00:00:03.000\nThe quick brown\n\n00:00:03.000 --> 00:00:04.000\nThe quick brown fox\n";
        let doc = parse_vtt(vtt).unwrap();
        let deduped = dedup_rolling(&doc.cues);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].text, "The quick brown fox");
    }

    #[test]
    fn test_dedup_preserves_repeated() {
        let cues = vec![
            CaptionCue {
                start_ms: 1000,
                end_ms: 2000,
                text: "Hello".to_string(),
            },
            CaptionCue {
                start_ms: 3000,
                end_ms: 4000,
                text: "Hello".to_string(),
            },
        ];
        let deduped = dedup_rolling(&cues);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_render_srt() {
        let cues = vec![CaptionCue {
            start_ms: 1000,
            end_ms: 3000,
            text: "Hello".to_string(),
        }];
        let srt = render_srt(&cues);
        assert!(srt.contains("1\n"));
        assert!(srt.contains("00:00:01,000 --> 00:00:03,000"));
        assert!(srt.contains("Hello"));
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("Hello World!"), "Hello_World");
        assert_eq!(sanitize_filename(""), "transcript");
        assert_eq!(sanitize_filename("../../../etc/passwd"), "etcpasswd");
    }

    #[test]
    fn test_select_caption_track_manual_priority() {
        let manual = vec![CaptionTrack {
            language: "en".to_string(),
            name: Some("English".to_string()),
            url: None,
            ext: Some("vtt".to_string()),
            is_automatic: false,
            is_translated: false,
        }];
        let auto = vec![CaptionTrack {
            language: "en".to_string(),
            name: Some("English (auto)".to_string()),
            url: None,
            ext: Some("vtt".to_string()),
            is_automatic: true,
            is_translated: false,
        }];
        let selected = select_caption_track(&manual, &auto, None, true).unwrap();
        assert!(!selected.is_automatic);
    }

    #[test]
    fn test_select_caption_track_no_substitution() {
        let manual = vec![CaptionTrack {
            language: "en".to_string(),
            name: Some("English".to_string()),
            url: None,
            ext: Some("vtt".to_string()),
            is_automatic: false,
            is_translated: false,
        }];
        let auto = vec![];
        let selected = select_caption_track(&manual, &auto, Some("de"), true);
        assert!(selected.is_none());
    }
}
