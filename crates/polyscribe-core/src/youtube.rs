// SPDX-License-Identifier: MIT

//! YouTube caption retrieval, parsing, and rendering.
//!
//! This module provides the data types and pure functions for working with
//! YouTube captions: URL validation, WebVTT parsing, rolling-caption
//! deduplication, and output rendering. Network access and subprocess
//! orchestration live in the CLI layer.

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub speaker_marker: bool,
}

fn is_false(value: &bool) -> bool {
    !value
}

/// A parsed WebVTT file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VttDocument {
    pub cues: Vec<CaptionCue>,
    pub malformed_cues: usize,
    pub empty_cues: usize,
}

/// Parse a WebVTT string into cues.
pub fn parse_vtt(input: &str) -> Result<VttDocument> {
    let mut cues = Vec::new();
    let mut malformed_cues = 0;
    let mut empty_cues = 0;
    let normalized = input
        .trim_start_matches('\u{feff}')
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let mut blocks = normalized.split("\n\n");
    let header = blocks.next().unwrap_or_default();
    let header_line = header.lines().next().unwrap_or_default().trim_start();
    if !(header_line == "WEBVTT"
        || header_line.starts_with("WEBVTT ")
        || header_line.starts_with("WEBVTT\t"))
    {
        return Err(anyhow!("missing WebVTT header"));
    }

    for block in blocks {
        let lines: Vec<&str> = block.lines().collect();
        let Some(first) = lines.iter().position(|line| !line.trim().is_empty()) else {
            continue;
        };
        let first_line = lines[first].trim();
        if first_line == "STYLE" || first_line == "REGION" || first_line.starts_with("NOTE") {
            continue;
        }
        let time_index = if first_line.contains("-->") {
            first
        } else {
            first + 1
        };
        let Some(time_line) = lines.get(time_index).map(|line| line.trim()) else {
            malformed_cues += 1;
            continue;
        };
        let Some((start_raw, end_raw)) = time_line.split_once("-->") else {
            malformed_cues += 1;
            continue;
        };
        let Some(start_ms) = parse_vtt_timestamp(start_raw.trim()) else {
            malformed_cues += 1;
            continue;
        };
        let end_raw = end_raw.split_whitespace().next().unwrap_or_default();
        let Some(end_ms) = parse_vtt_timestamp(end_raw) else {
            malformed_cues += 1;
            continue;
        };
        if end_ms < start_ms {
            malformed_cues += 1;
            continue;
        }

        let raw_text = lines[time_index + 1..].join("\n");
        let (text, speaker, speaker_marker) = clean_vtt_text(&raw_text);
        if text.is_empty() {
            empty_cues += 1;
            continue;
        }
        cues.push(CaptionCue {
            start_ms,
            end_ms,
            text,
            speaker,
            speaker_marker,
        });
    }

    Ok(VttDocument {
        cues,
        malformed_cues,
        empty_cues,
    })
}

fn parse_vtt_timestamp(s: &str) -> Option<u64> {
    let s = s.trim();
    let parts: Vec<&str> = s.split(':').collect();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [hours, minutes, seconds] => (
            hours.parse::<u64>().ok()?,
            minutes.parse::<u64>().ok()?,
            *seconds,
        ),
        [minutes, seconds] => (0, minutes.parse::<u64>().ok()?, *seconds),
        _ => return None,
    };
    if minutes > 59 {
        return None;
    }
    let (whole_seconds, fraction) = seconds.split_once('.')?;
    let whole_seconds = whole_seconds.parse::<u64>().ok()?;
    if whole_seconds > 59
        || fraction.is_empty()
        || fraction.len() > 3
        || !fraction.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let mut millis = fraction.parse::<u64>().ok()?;
    for _ in fraction.len()..3 {
        millis *= 10;
    }
    hours
        .checked_mul(3_600_000)?
        .checked_add(minutes.checked_mul(60_000)?)?
        .checked_add(whole_seconds.checked_mul(1000)?)?
        .checked_add(millis)
}

fn clean_vtt_text(raw: &str) -> (String, Option<String>, bool) {
    let mut text = String::new();
    let mut speaker = None;
    let mut inside_tag = false;
    let mut tag = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '<' {
            inside_tag = true;
            tag.clear();
        } else if ch == '>' && inside_tag {
            inside_tag = false;
            if let Some(name) = tag.strip_prefix('v')
                && let Some(name) = name
                    .chars()
                    .next()
                    .filter(|c| c.is_whitespace())
                    .map(|_| name.trim())
                && !name.is_empty()
            {
                let cleaned: String = name.chars().filter(|c| !c.is_control()).collect();
                let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
                if !cleaned.is_empty() {
                    speaker = Some(cleaned);
                }
            }
        } else if inside_tag {
            tag.push(ch);
        } else if ch == '&' {
            let mut entity = String::new();
            while let Some(next) = chars.peek().copied() {
                if next == ';' || entity.len() >= 12 {
                    break;
                }
                if next.is_whitespace() || next == '&' || next == '<' {
                    break;
                }
                entity.push(next);
                chars.next();
            }
            if chars.peek() == Some(&';') {
                chars.next();
                if let Some(decoded) = decode_entity(&entity) {
                    text.push(decoded);
                } else {
                    text.push('&');
                    text.push_str(&entity);
                    text.push(';');
                }
            } else {
                text.push('&');
                text.push_str(&entity);
            }
        } else {
            text.push(ch);
        }
    }
    let text: String = text.chars().filter(|c| !c.is_control()).collect();
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let (text, marker) = if let Some(text) = text.strip_prefix(">>") {
        (text.trim_start().to_string(), true)
    } else {
        (text, false)
    };
    (text, speaker, marker)
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        "nbsp" => Some('\u{00a0}'),
        "hellip" => Some('…'),
        _ if entity.starts_with("#x") || entity.starts_with("#X") => {
            char::from_u32(u32::from_str_radix(&entity[2..], 16).ok()?)
        }
        _ if entity.starts_with('#') => char::from_u32(entity[1..].parse().ok()?),
        _ => None,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub merged_updates: usize,
    pub removed_duplicate_cues: usize,
    pub removed_overlap_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedTranscript {
    pub cues: Vec<CaptionCue>,
    pub stats: NormalizationStats,
}

#[derive(Debug, Clone, Copy)]
struct WordSpan {
    start: usize,
}

fn words(text: &str) -> Vec<(WordSpan, String)> {
    let mut result = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if !ch.is_whitespace() {
            start.get_or_insert(index);
        } else if let Some(word_start) = start.take() {
            let raw = &text[word_start..index];
            let normalized: String = raw
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect();
            if normalized.is_empty() {
                continue;
            }
            result.push((WordSpan { start: word_start }, normalized));
        }
    }
    if let Some(word_start) = start {
        let raw = &text[word_start..];
        let normalized: String = raw
            .chars()
            .filter(|ch| ch.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        if !normalized.is_empty() {
            result.push((WordSpan { start: word_start }, normalized));
        }
    }
    result
}

fn novel_suffix(text: &str, words: &[(WordSpan, String)], overlap: usize) -> String {
    if overlap >= words.len() {
        return String::new();
    }
    text[words[overlap].0.start..].trim().to_string()
}

fn temporal_neighbors(previous: &CaptionCue, current: &CaptionCue) -> bool {
    const MAX_GAP_MS: u64 = 250;
    const MAX_START_DELTA_MS: u64 = 3_000;
    current.start_ms <= previous.start_ms.saturating_add(MAX_START_DELTA_MS)
        && current.start_ms <= previous.end_ms.saturating_add(MAX_GAP_MS)
        && current.end_ms >= previous.start_ms
}

fn canonical_spacing(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn overlap_count(previous: &[String], current: &[String]) -> usize {
    (1..=previous.len().min(current.len()))
        .rev()
        .find(|count| previous[previous.len() - count..] == current[..*count])
        .unwrap_or(0)
}

/// Normalize only adjacent, temporally related rolling captions.
///
/// Complete words are compared case-insensitively after punctuation is
/// ignored. At least two words and overlapping or near-touching cue times are
/// required. Ambiguous one-word matches, speaker changes, corrections, and
/// distant repetitions are retained. Cue order is made chronological for
/// rendering; original parsed cues remain available to the caller.
pub fn normalize_rolling_cues(cues: &[CaptionCue]) -> NormalizedTranscript {
    let mut ordered: Vec<(usize, &CaptionCue)> = cues.iter().enumerate().collect();
    ordered.sort_by_key(|(index, cue)| (cue.start_ms, cue.end_ms, *index));

    let mut output: Vec<CaptionCue> = Vec::new();
    let mut stats = NormalizationStats::default();
    let mut previous: Option<&CaptionCue> = None;
    let mut previous_output_index: Option<usize> = None;

    for position in 0..ordered.len() {
        let current = ordered[position].1;
        let next_cue = if position + 1 < ordered.len() {
            Some(ordered[position + 1].1)
        } else {
            None
        };

        let current_words = words(&current.text);
        if current_words.is_empty() || current.end_ms < current.start_ms {
            previous = Some(current);
            previous_output_index = None;
            continue;
        }

        // Skip exact rolling-caption duplicates even when their display
        // windows are not within the tight 250 ms tolerance, as long as
        // they are adjacent in time. This handles YouTube's commit cues.
        if let Some(prior) = previous {
            let prior_words = words(&prior.text);
            let left: Vec<String> = prior_words.iter().map(|(_, word)| word.clone()).collect();
            let right: Vec<String> = current_words.iter().map(|(_, word)| word.clone()).collect();
            if !left.is_empty()
                && left == right
                && temporal_neighbors(prior, current)
                && prior.speaker == current.speaker
            {
                stats.removed_duplicate_cues += 1;
                if let Some(index) = previous_output_index {
                    output[index].speaker_marker = current.speaker_marker;
                }
                previous = Some(current);
                continue;
            }
        }

        // Skip fragment/commit cues whose text is already contained in an
        // adjacent following cue. These are intermediate rolling displays
        // that add no new transcript content.
        if let Some(next) = next_cue
            && temporal_neighbors(current, next)
        {
            let current_normalized = current
                .text
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            let next_normalized = next
                .text
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            let current_words_normalized: Vec<String> =
                current_words.iter().map(|(_, word)| word.clone()).collect();
            let next_words = words(&next.text);
            let next_word_strings: Vec<String> =
                next_words.iter().map(|(_, word)| word.clone()).collect();
            let is_prefix_of_next = next_word_strings.starts_with(&current_words_normalized);
            if !current_normalized.is_empty()
                && next_normalized.contains(&current_normalized)
                && next_normalized.len() > current_normalized.len()
                && !is_prefix_of_next
            {
                continue;
            }
        }

        let mut consumed = false;
        if let Some(prior) = previous
            && temporal_neighbors(prior, current)
            && prior.speaker == current.speaker
        {
            let prior_words = words(&prior.text);
            let left: Vec<String> = prior_words.iter().map(|(_, word)| word.clone()).collect();
            let right: Vec<String> = current_words.iter().map(|(_, word)| word.clone()).collect();
            if left.len() >= 2 && right.starts_with(&left) {
                let extra_candidate_words = &current_words[left.len()..];
                let prior_word_strings: Vec<String> =
                    prior_words.iter().map(|(_, word)| word.clone()).collect();
                let extra_word_strings: Vec<String> = extra_candidate_words
                    .iter()
                    .map(|(_, word)| word.clone())
                    .collect();
                let boundary_overlap = overlap_count(&prior_word_strings, &extra_word_strings);
                let extra =
                    novel_suffix(&current.text, &current_words, left.len() + boundary_overlap);
                if !extra.is_empty() {
                    if let Some(index) = previous_output_index {
                        let previous_output = &mut output[index];
                        if !previous_output.text.is_empty() {
                            previous_output.text.push(' ');
                        }
                        previous_output.text.push_str(&canonical_spacing(&extra));
                        previous_output.end_ms = previous_output.end_ms.max(current.end_ms);
                        previous_output.speaker_marker = current.speaker_marker;
                        stats.merged_updates += 1;
                        stats.removed_overlap_tokens += left.len() + boundary_overlap;
                        consumed = true;
                    }
                } else if prior.start_ms.abs_diff(current.start_ms) <= 250
                    && prior.end_ms.abs_diff(current.end_ms) <= 250
                {
                    if let Some(index) = previous_output_index {
                        output[index].speaker_marker = current.speaker_marker;
                    }
                    stats.removed_duplicate_cues += 1;
                    consumed = true;
                }
            }

            if !consumed {
                let overlap = overlap_count(&left, &right);
                if overlap >= 1 && overlap < right.len() {
                    // Rolling captions often repeat the boundary word once
                    // more at the start of the new content; strip that
                    // repeated boundary word from the novel suffix.
                    let mut boundary_duplicates = 0usize;
                    while boundary_duplicates < right.len() - overlap
                        && boundary_duplicates < overlap
                        && right[overlap + boundary_duplicates]
                            == left[left.len() - 1 - boundary_duplicates]
                    {
                        boundary_duplicates += 1;
                    }
                    let extra =
                        novel_suffix(&current.text, &current_words, overlap + boundary_duplicates);
                    if !extra.is_empty() {
                        let mut normalized = current.clone();
                        normalized.text = canonical_spacing(&extra);
                        output.push(normalized);
                        previous_output_index = Some(output.len() - 1);
                        stats.removed_overlap_tokens += overlap + boundary_duplicates;
                        consumed = true;
                    }
                } else if overlap >= 2
                    && overlap == right.len()
                    && prior.start_ms.abs_diff(current.start_ms) <= 250
                    && prior.end_ms.abs_diff(current.end_ms) <= 250
                {
                    if let Some(index) = previous_output_index {
                        output[index].speaker_marker = current.speaker_marker;
                    }
                    stats.removed_duplicate_cues += 1;
                    consumed = true;
                }
            }
        }

        if !consumed {
            let mut normalized = current.clone();
            normalized.text = canonical_spacing(&normalized.text);
            output.push(normalized);
            previous_output_index = Some(output.len() - 1);
        }
        previous = Some(current);
    }

    NormalizedTranscript {
        cues: output,
        stats,
    }
}

/// Compatibility helper returning only normalized cues.
pub fn dedup_rolling(cues: &[CaptionCue]) -> Vec<CaptionCue> {
    normalize_rolling_cues(cues).cues
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

    // A manual track is still usable when automatic captions are absent,
    // even if the caller prefers automatic captions when both kinds exist.
    if !manual.is_empty() {
        for track in &manual {
            if track.language.to_lowercase().starts_with("en") {
                return Some((*track).clone());
            }
        }
        return Some((*manual[0]).clone());
    }

    None
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

    fn cue(start_ms: u64, end_ms: u64, text: &str) -> CaptionCue {
        CaptionCue {
            start_ms,
            end_ms,
            text: text.to_string(),
            speaker: None,
            speaker_marker: false,
        }
    }

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
        assert_eq!(doc.malformed_cues, 0);
        assert_eq!(doc.empty_cues, 0);
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
    fn fixture_normalizes_overlap_without_dropping_the_novel_phrase() {
        let vtt = include_str!("../tests/fixtures/youtube/rolling-overlap.vtt");
        let parsed = parse_vtt(vtt).unwrap();
        let normalized = normalize_rolling_cues(&parsed.cues);
        assert_eq!(parsed.cues.len(), 5);
        assert_eq!(normalized.cues.len(), 3);
        assert_eq!(
            normalized.cues[0].text,
            "That's supposed to be confidential, but"
        );
        assert!(normalized.cues[0].speaker_marker);
        assert_eq!(
            normalized.cues[1].text,
            "yeah, I did report you to HR. I sent"
        );
        assert_eq!(normalized.cues[2].text, "the report yesterday.");
        assert_eq!(normalized.stats.removed_duplicate_cues, 2);
        assert_eq!(normalized.stats.removed_overlap_tokens, 2);
        assert_eq!(
            render_txt(&normalized.cues),
            ">> That's supposed to be confidential, but yeah, I did report you to HR. I sent the report yesterday."
        );
    }

    #[test]
    fn sliding_overlap_requires_two_complete_words_and_preserves_repeated_lyrics() {
        let normalized = normalize_rolling_cues(&[
            cue(0, 2000, "We can try again."),
            cue(1500, 3500, "can try again, tomorrow."),
            cue(8000, 9500, "Never gonna give you up."),
            cue(12_000, 13_500, "Never gonna give you up."),
        ]);
        assert_eq!(normalized.cues.len(), 4);
        assert_eq!(normalized.cues[1].text, "tomorrow.");
        assert_eq!(normalized.cues[2].text, "Never gonna give you up.");
        assert_eq!(normalized.cues[3].text, "Never gonna give you up.");
        assert_eq!(
            normalize_rolling_cues(&[cue(10_000, 11_000, "We can."), cue(10_100, 11_100, "can."),])
                .cues
                .len(),
            2
        );
    }

    #[test]
    fn punctuation_whitespace_and_sentence_boundary_updates_are_normalized() {
        let normalized = normalize_rolling_cues(&[
            cue(1000, 2000, "Well,   this is the first sentence"),
            cue(1800, 3000, "WELL this is the first sentence. And then"),
            cue(4000, 5000, "That's a phrase"),
            cue(4800, 6000, "Thats a phrase, continued"),
            cue(7000, 20_000, "We will be back again."),
            cue(15_000, 16_000, "We will be back again."),
        ]);
        assert_eq!(normalized.cues.len(), 4);
        assert_eq!(
            normalized.cues[0].text,
            "Well, this is the first sentence And then"
        );
        assert_eq!(normalized.cues[0].start_ms, 1000);
        assert_eq!(normalized.cues[0].end_ms, 3000);
        assert_eq!(normalized.cues[1].text, "That's a phrase continued");
        assert_eq!(normalized.cues[2].text, "We will be back again.");
        assert_eq!(normalized.cues[3].text, "We will be back again.");
    }

    #[test]
    fn speaker_changes_are_not_merged_and_voice_markup_is_preserved() {
        let document = parse_vtt(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<v Alice>Hello there.</v>\n\n00:00:01.500 --> 00:00:02.500\n<v Bob>Hello there.</v>\n",
        )
        .unwrap();
        assert_eq!(document.cues[0].speaker.as_deref(), Some("Alice"));
        assert_eq!(document.cues[1].speaker.as_deref(), Some("Bob"));
        assert_eq!(normalize_rolling_cues(&document.cues).cues.len(), 2);
        assert_eq!(
            render_txt(&document.cues),
            "Alice: Hello there. Bob: Hello there."
        );
    }

    #[test]
    fn parser_decodes_entities_handles_crlf_and_reports_bad_or_empty_cues() {
        let vtt = "WEBVTT\r\n\r\n00:00:01.0 --> 00:00:02.00\r\n< b >Unicode &amp; punctuation &#x1F642; &nbsp;終わり</b>\r\n\r\n00:00:03.000 --> nope\r\nBad time\r\n\r\n00:00:04.000 --> 00:00:05.000\r\n<b></b>\r\n";
        let document = parse_vtt(vtt).unwrap();
        assert_eq!(document.cues.len(), 1);
        assert_eq!(document.cues[0].start_ms, 1000);
        assert_eq!(document.cues[0].text, "Unicode & punctuation 🙂 終わり");
        assert_eq!(document.malformed_cues, 1);
        assert_eq!(document.empty_cues, 1);
    }

    #[test]
    fn out_of_order_and_overlapping_cues_render_with_monotonic_starts() {
        let normalized = normalize_rolling_cues(&[
            cue(3000, 5000, "A different phrase"),
            cue(1000, 4000, "First phrase"),
            cue(2000, 4500, "First phrase repeats"),
        ]);
        assert_eq!(normalized.cues[0].start_ms, 1000);
        assert!(
            normalized
                .cues
                .windows(2)
                .all(|pair| pair[0].start_ms <= pair[1].start_ms)
        );
        assert!(normalized.cues.iter().all(|cue| cue.end_ms >= cue.start_ms));
        assert_eq!(normalized.cues.len(), 2);
    }

    #[test]
    fn test_dedup_preserves_repeated() {
        let cues = vec![
            CaptionCue {
                start_ms: 1000,
                end_ms: 2000,
                text: "Hello".to_string(),
                speaker: None,
                speaker_marker: false,
            },
            CaptionCue {
                start_ms: 3000,
                end_ms: 4000,
                text: "Hello".to_string(),
                speaker: None,
                speaker_marker: false,
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
            speaker: None,
            speaker_marker: false,
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
    fn selection_falls_back_to_manual_when_automatic_is_preferred_but_absent() {
        let manual = vec![
            CaptionTrack {
                language: "de".to_string(),
                name: Some("German".to_string()),
                url: None,
                ext: Some("vtt".to_string()),
                is_automatic: false,
                is_translated: false,
            },
            CaptionTrack {
                language: "en".to_string(),
                name: Some("English".to_string()),
                url: None,
                ext: Some("vtt".to_string()),
                is_automatic: false,
                is_translated: false,
            },
        ];
        let selected = select_caption_track(&manual, &[], None, false).unwrap();
        assert_eq!(selected.language, "en");
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
