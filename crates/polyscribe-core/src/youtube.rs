// SPDX-License-Identifier: MIT

//! YouTube URL validation, video metadata, transcript cues, and rendering.
//!
//! Network access, audio acquisition, caching, and subprocess orchestration
//! live in the CLI layer.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

pub const YOUTUBE_NORMALIZATION_VERSION: u32 = 4;

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
        return Err(anyhow!("unsupported YouTube host"));
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

/// A timed transcript cue with text and optional speaker information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionCue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub speaker_marker: bool,
}

fn is_false(value: &bool) -> bool {
    !value
}

/// Cue-normalization counters stored with a YouTube transcript.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub merged_updates: usize,
    pub removed_duplicate_cues: usize,
    pub removed_overlap_tokens: usize,
}

/// Video fields retained from the bounded `yt-dlp` metadata response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub id: String,
    pub title: Option<String>,
    pub uploader: Option<String>,
    pub duration: Option<f64>,
}

/// Render cues as plain text.
pub fn render_txt(cues: &[CaptionCue]) -> String {
    cues.iter()
        .map(rendered_cue_text)
        .collect::<Vec<_>>()
        .join(" ")
}

fn rendered_cue_text(cue: &CaptionCue) -> String {
    if let Some(speaker) = &cue.speaker {
        format!("{speaker}: {}", cue.text)
    } else if cue.speaker_marker {
        format!(">> {}", cue.text)
    } else {
        cue.text.clone()
    }
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
        out.push_str(&rendered_cue_text(cue));
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
    fn test_validate_youtube_url_does_not_echo_untrusted_host() {
        let url = "https://\u{1b}]52;c;aGVsbG8=\u{7}.example/watch?v=dQw4w9WgXcQ";
        let error = validate_youtube_url(url).unwrap_err().to_string();
        assert_eq!(error, "unsupported YouTube host");
        assert!(!error.chars().any(char::is_control));
    }

    #[test]
    fn test_validate_youtube_url_rejects_injection() {
        assert!(
            validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ;rm -rf /").is_err()
        );
    }

    #[test]
    fn test_render_srt() {
        let cues = vec![CaptionCue {
            start_ms: 1000,
            end_ms: 3000,
            text: "Hello".to_string(),
            speaker: None,
            speaker_marker: false,
        }];
        let srt = render_srt(&cues);
        assert!(srt.contains("1\n"));
        assert!(srt.contains("00:00:01,000 --> 00:00:03,000"));
        assert!(srt.contains("Hello"));
    }
}
