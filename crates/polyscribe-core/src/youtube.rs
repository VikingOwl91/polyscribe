// SPDX-License-Identifier: MIT

//! YouTube URL validation, video metadata, transcript cues, and rendering.
//!
//! Network access, audio acquisition, caching, and subprocess orchestration
//! live in the CLI layer.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Version of the cue normalization. Bumped from 4 to 5 when the fixed
/// `speaker1` label was removed from the local Whisper path: entries cached
/// under version 4 may still carry that label in their cues, so they are
/// discarded (never served) instead of being delivered as-is.
pub const YOUTUBE_NORMALIZATION_VERSION: u32 = 5;

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
///
/// Missing fields stay `None`; no placeholder such as "NA" or "unknown" is
/// ever stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub id: String,
    pub title: Option<String>,
    pub uploader: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub channel_url: Option<String>,
    /// Upload date as ISO `YYYY-MM-DD`; yt-dlp reports `YYYYMMDD` and an
    /// invalid value becomes `None`.
    #[serde(default)]
    pub upload_date: Option<String>,
    pub duration: Option<f64>,
}

/// Convert a yt-dlp `YYYYMMDD` upload date to ISO `YYYY-MM-DD`.
///
/// Anything that is not a valid calendar date in that exact shape (including
/// yt-dlp's `NA`) yields `None`.
pub fn iso_upload_date(raw: &str) -> Option<String> {
    chrono::NaiveDate::parse_from_str(raw.trim(), "%Y%m%d")
        .ok()
        .map(|date| date.format("%Y-%m-%d").to_string())
}

/// Quote a string as a YAML double-quoted scalar.
///
/// The escapes are the JSON ones (`\"`, `\\`, `\n`, `\uXXXX`, ...), which are
/// a subset of the YAML double-quoted style, so any title survives: colons,
/// quotes, newlines, and emoji stay inside one quoted scalar. Control
/// characters, DEL, and the C1 range are always escaped because YAML does not
/// allow them raw; non-ASCII text such as emoji is kept literal.
pub fn yaml_quote(value: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || (0x7f..=0x9f).contains(&(c as u32)) => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn yaml_field(out: &mut String, indent: &str, key: &str, value: Option<&str>) {
    out.push_str(indent);
    out.push_str(key);
    out.push_str(": ");
    match value {
        Some(value) => out.push_str(&yaml_quote(value)),
        None => out.push_str("null"),
    }
    out.push('\n');
}

/// Inputs for the YAML frontmatter block written above a plain-text transcript.
pub struct TranscriptFrontmatter<'a> {
    pub title: Option<&'a str>,
    pub video_id: &'a str,
    pub channel: Option<&'a str>,
    pub channel_url: Option<&'a str>,
    pub upload_date: Option<&'a str>,
    pub duration_seconds: Option<f64>,
    pub language: Option<&'a str>,
    pub transcribed_at: &'a str,
    pub engine: &'a str,
    pub model: &'a str,
    pub backend: Option<&'a str>,
}

/// Build the YAML frontmatter block (ending in `---\n`) for a txt transcript.
///
/// Mirrors yt-transcript's `build_frontmatter`: `source` is the canonical
/// watch URL, `duration` is whole seconds and is omitted when unknown, and
/// the `transcription:` block records the engine/model/backend actually used.
pub fn build_frontmatter(frontmatter: &TranscriptFrontmatter<'_>) -> String {
    let mut out = String::from("---\n");
    yaml_field(&mut out, "", "title", frontmatter.title);
    let source = format!("https://www.youtube.com/watch?v={}", frontmatter.video_id);
    yaml_field(&mut out, "", "source", Some(&source));
    yaml_field(&mut out, "", "video_id", Some(frontmatter.video_id));
    yaml_field(&mut out, "", "channel", frontmatter.channel);
    yaml_field(&mut out, "", "channel_url", frontmatter.channel_url);
    yaml_field(&mut out, "", "upload_date", frontmatter.upload_date);
    if let Some(duration) = frontmatter
        .duration_seconds
        .filter(|duration| duration.is_finite() && *duration >= 0.0)
    {
        out.push_str(&format!("duration: {}\n", duration.round() as i64));
    }
    yaml_field(&mut out, "", "language", frontmatter.language);
    yaml_field(
        &mut out,
        "",
        "transcribed_at",
        Some(frontmatter.transcribed_at),
    );
    out.push_str("transcription:\n");
    yaml_field(&mut out, "  ", "engine", Some(frontmatter.engine));
    yaml_field(&mut out, "  ", "model", Some(frontmatter.model));
    if let Some(backend) = frontmatter.backend {
        yaml_field(&mut out, "  ", "backend", Some(backend));
    }
    out.push_str("---\n");
    out
}

/// Make a metadata value safe as a single file-name component.
///
/// Slashes are replaced (so no path separator and no `..` traversal survives),
/// control characters are dropped, leading dots are removed, and the result
/// is capped at `max_bytes` UTF-8 bytes on a character boundary.
pub fn fs_safe(text: &str, max_bytes: usize) -> String {
    let mut cleaned = String::with_capacity(text.len());
    for ch in text.chars() {
        let ch = match ch {
            '/' => '\u{2215}',
            '\\' => '\u{2216}',
            c if (c as u32) < 0x20 || c as u32 == 0x7f => continue,
            c => c,
        };
        if ch == '.' && cleaned.ends_with('.') {
            continue;
        }
        cleaned.push(ch);
    }
    let cleaned = cleaned.trim().trim_start_matches('.');
    let mut end = cleaned.len();
    if cleaned.len() > max_bytes {
        end = cleaned
            .char_indices()
            .take_while(|(index, ch)| index + ch.len_utf8() <= max_bytes)
            .map(|(index, ch)| index + ch.len_utf8())
            .last()
            .unwrap_or(0);
    }
    let trimmed = cleaned[..end].trim_end_matches(['.', ' ']);
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Slugify a title for use in a file name: umlauts are transliterated, only
/// ASCII letters and digits survive, everything else collapses to `-`.
pub fn slugify(text: &str, max_len: usize) -> String {
    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            'ä' => transliterated.push_str("ae"),
            'ö' => transliterated.push_str("oe"),
            'ü' => transliterated.push_str("ue"),
            'ß' => transliterated.push_str("ss"),
            'Ä' => transliterated.push_str("Ae"),
            'Ö' => transliterated.push_str("Oe"),
            'Ü' => transliterated.push_str("Ue"),
            c => transliterated.push(c),
        }
    }
    let mut out = String::new();
    for ch in transliterated.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let slug: String = out
        .trim_matches('-')
        .chars()
        .take(max_len)
        .collect::<String>()
        .trim_end_matches('-')
        .to_string();
    if slug.is_empty() {
        "video".to_string()
    } else {
        slug
    }
}

/// Canonical transcript file name: `YYYY-MM-DD_<channel>_<video-id>_<title-slug>.<ext>`.
///
/// Every component comes from `fs_safe`/`slugify`, so the result can never
/// contain `..`, a path separator, or a control character.
pub fn canonical_file_name(
    upload_date: Option<&str>,
    channel: Option<&str>,
    video_id: &str,
    title: Option<&str>,
    extension: &str,
) -> String {
    let date = upload_date
        .filter(|date| chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").is_ok())
        .unwrap_or("unknown-date");
    let channel = fs_safe(channel.unwrap_or(""), 50).replace(' ', "_");
    let video_id = fs_safe(video_id, 50);
    let slug = slugify(title.unwrap_or(""), 70);
    let extension = fs_safe(extension, 10);
    format!("{date}_{channel}_{video_id}_{slug}.{extension}")
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

    #[test]
    fn iso_upload_date_only_accepts_real_yyyymmdd_dates() {
        assert_eq!(iso_upload_date("20240131"), Some("2024-01-31".to_string()));
        assert_eq!(iso_upload_date("20240229"), Some("2024-02-29".to_string()));
        assert_eq!(
            iso_upload_date(" 20240131 "),
            Some("2024-01-31".to_string())
        );
        assert_eq!(iso_upload_date("20230229"), None);
        assert_eq!(iso_upload_date("20241301"), None);
        assert_eq!(iso_upload_date("2024-01-31"), None);
        assert_eq!(iso_upload_date("NA"), None);
        assert_eq!(iso_upload_date(""), None);
    }

    #[test]
    fn yaml_quote_keeps_hostile_titles_on_one_line_and_roundtrips() {
        let hostile = "A: \"quoted\" \\ line\nnext\t🎉 \u{1} \u{7f} é";
        let quoted = yaml_quote(hostile);
        assert!(quoted.starts_with('"') && quoted.ends_with('"'));
        assert!(!quoted[1..quoted.len() - 1].contains('\n'));
        assert!(quoted.contains("🎉"));
        assert!(quoted.contains("é"));
        assert!(quoted.contains("\\u0001"));
        assert!(quoted.contains("\\u007f"));
        let parsed: String = serde_json::from_str(&quoted).unwrap();
        assert_eq!(parsed, hostile);
        assert_eq!(yaml_quote("plain"), "\"plain\"");
    }

    #[test]
    fn frontmatter_contains_all_fields_and_quotes_them() {
        let head = build_frontmatter(&TranscriptFrontmatter {
            title: Some("Title: with \"quotes\"\nand newline"),
            video_id: "dQw4w9WgXcQ",
            channel: Some("Cool Channel"),
            channel_url: Some("https://www.youtube.com/@cool"),
            upload_date: Some("2024-01-31"),
            duration_seconds: Some(83.6),
            language: Some("en"),
            transcribed_at: "2026-01-02T03:04:05+00:00",
            engine: "whisper",
            model: "ggml-base.bin",
            backend: Some("cpu"),
        });
        assert_eq!(
            head,
            "---\n\
             title: \"Title: with \\\"quotes\\\"\\nand newline\"\n\
             source: \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\"\n\
             video_id: \"dQw4w9WgXcQ\"\n\
             channel: \"Cool Channel\"\n\
             channel_url: \"https://www.youtube.com/@cool\"\n\
             upload_date: \"2024-01-31\"\n\
             duration: 84\n\
             language: \"en\"\n\
             transcribed_at: \"2026-01-02T03:04:05+00:00\"\n\
             transcription:\n\
             \x20 engine: \"whisper\"\n\
             \x20 model: \"ggml-base.bin\"\n\
             \x20 backend: \"cpu\"\n\
             ---\n"
        );
    }

    #[test]
    fn frontmatter_writes_null_for_missing_fields_and_omits_duration() {
        let head = build_frontmatter(&TranscriptFrontmatter {
            title: None,
            video_id: "dQw4w9WgXcQ",
            channel: None,
            channel_url: None,
            upload_date: None,
            duration_seconds: None,
            language: None,
            transcribed_at: "2026-01-02T03:04:05+00:00",
            engine: "voxtral",
            model: "fake-model",
            backend: None,
        });
        assert!(head.contains("\ntitle: null\n"));
        assert!(head.contains("\nchannel: null\n"));
        assert!(head.contains("\nchannel_url: null\n"));
        assert!(head.contains("\nupload_date: null\n"));
        assert!(head.contains("\nlanguage: null\n"));
        assert!(!head.contains("duration:"));
        assert!(!head.contains("backend:"));
        assert!(!head.contains("NA"));
        assert!(!head.contains("unknown"));
        assert!(head.ends_with("---\n"));
    }

    #[test]
    fn fs_safe_removes_separators_control_chars_and_dotdot() {
        let cleaned = fs_safe("../../etc\\passwd\n\u{1}💣", 60);
        assert!(!cleaned.contains('/'));
        assert!(!cleaned.contains('\\'));
        assert!(!cleaned.contains(".."));
        assert!(!cleaned.chars().any(|c| c.is_control()));
        assert!(!cleaned.starts_with('.'));
        assert_eq!(fs_safe("", 60), "unknown");
        assert_eq!(fs_safe("...", 60), "unknown");
        assert_eq!(fs_safe("  .hidden  ", 60), "hidden");
        assert_eq!(fs_safe("a..b", 60), "a.b");
    }

    #[test]
    fn fs_safe_caps_utf8_bytes_on_char_boundaries() {
        let capped = fs_safe("ääääääääää", 5);
        assert!(capped.len() <= 5);
        assert_eq!(capped, "ää");
        assert!(capped.is_char_boundary(capped.len()));
    }

    #[test]
    fn slugify_keeps_only_ascii_alnum_and_dashes() {
        assert_eq!(slugify("Hello, World! 🎉", 70), "hello-world");
        assert_eq!(slugify("Grüße aus Köln", 70), "gruesse-aus-koeln");
        assert_eq!(slugify("../etc/passwd", 70), "etc-passwd");
        assert_eq!(slugify("", 70), "video");
        assert_eq!(slugify("!!!", 70), "video");
        assert!(slugify(&"a".repeat(200), 70).len() <= 70);
    }

    #[test]
    fn canonical_file_name_is_a_single_safe_component() {
        let name = canonical_file_name(
            Some("2024-01-31"),
            Some("Cool Channel"),
            "dQw4w9WgXcQ",
            Some("Full: \"Meta\" 🎉"),
            "txt",
        );
        assert_eq!(name, "2024-01-31_Cool_Channel_dQw4w9WgXcQ_full-meta.txt");

        let hostile = canonical_file_name(
            Some("not-a-date"),
            Some("../../x"),
            "dQw4w9WgXcQ",
            Some("../etc/passwd\n💣"),
            "txt",
        );
        assert!(hostile.starts_with("unknown-date_"));
        assert!(hostile.ends_with(".txt"));
        assert!(!hostile.contains('/'));
        assert!(!hostile.contains(".."));
        assert!(!hostile.chars().any(|c| c.is_control()));
        assert_eq!(hostile.matches('_').count(), 3);

        let missing = canonical_file_name(None, None, "dQw4w9WgXcQ", None, "srt");
        assert_eq!(missing, "unknown-date_unknown_dQw4w9WgXcQ_video.srt");
    }
}
