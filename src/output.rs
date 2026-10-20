// SPDX-License-Identifier: MIT
// Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.

//! Transcript data types and output rendering (JSON, TOML, SRT).
//!
//! This module owns the on-disk transcript schema and everything needed to
//! turn a list of [`OutputEntry`] segments into the files PolyScribe writes:
//! pretty-printed JSON, TOML, and SRT subtitles, or JSON on stdout.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use chrono::Local;
use serde::{Deserialize, Serialize};

/// Transcript entry for a single segment.
#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct OutputEntry {
    /// Sequential id in output ordering.
    pub id: u64,
    /// Speaker label associated with the segment.
    pub speaker: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Text content.
    pub text: String,
}

/// Root object of a PolyScribe transcript file (JSON and TOML outputs).
#[derive(Debug, Serialize)]
pub struct OutputRoot {
    /// Transcript segments in output order.
    pub items: Vec<OutputEntry>,
}

/// Return a YYYY-MM-DD date prefix string for output file naming.
pub fn date_prefix() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

/// Format a floating-point number of seconds as SRT timestamp (HH:MM:SS,mmm).
pub fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as i64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let m = (total_secs / 60) % 60;
    let h = total_secs / 3600;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Render a list of transcript entries to SRT format.
pub fn render_srt(items: &[OutputEntry]) -> String {
    let mut out = String::new();
    for (i, e) in items.iter().enumerate() {
        let idx = i + 1;
        out.push_str(&format!("{idx}\n"));
        out.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(e.start),
            format_srt_time(e.end)
        ));
        if !e.speaker.is_empty() {
            out.push_str(&format!("{}: {}\n", e.speaker, e.text));
        } else {
            out.push_str(&format!("{}\n", e.text));
        }
        out.push('\n');
    }
    out
}

/// Sort entries by (start, end) and reassign sequential ids starting at 0.
///
/// Entries with non-finite timestamps sort as if equal on that key, matching
/// the historical comparator behavior.
pub fn sort_and_reindex(entries: &mut [OutputEntry]) {
    entries.sort_by(|a, b| {
        match a.start.partial_cmp(&b.start) {
            Some(std::cmp::Ordering::Equal) | None => {}
            Some(o) => return o,
        }
        a.end
            .partial_cmp(&b.end)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, e) in entries.iter_mut().enumerate() {
        e.id = i as u64;
    }
}

/// Write a transcript as `{base_name}.json`, `{base_name}.toml`, and
/// `{base_name}.srt` inside `dir`.
pub fn write_output_files(dir: &Path, base_name: &str, out: &OutputRoot) -> Result<()> {
    let json_path = dir.join(format!("{base_name}.json"));
    let toml_path = dir.join(format!("{base_name}.toml"));
    let srt_path = dir.join(format!("{base_name}.srt"));

    let mut json_file = File::create(&json_path)
        .with_context(|| format!("Failed to create output file: {}", json_path.display()))?;
    serde_json::to_writer_pretty(&mut json_file, out)?;
    writeln!(&mut json_file)?;

    let toml_str = toml::to_string_pretty(out)?;
    let mut toml_file = File::create(&toml_path)
        .with_context(|| format!("Failed to create output file: {}", toml_path.display()))?;
    toml_file.write_all(toml_str.as_bytes())?;
    if !toml_str.ends_with('\n') {
        writeln!(&mut toml_file)?;
    }

    let srt_str = render_srt(&out.items);
    let mut srt_file = File::create(&srt_path)
        .with_context(|| format!("Failed to create output file: {}", srt_path.display()))?;
    srt_file.write_all(srt_str.as_bytes())?;
    Ok(())
}

/// Write a transcript as pretty-printed JSON to stdout.
pub fn write_json_stdout(out: &OutputRoot) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, out)?;
    writeln!(&mut handle)?;
    Ok(())
}

/// On-disk shape of a PolyScribe input transcript (`{"segments": [...]}`).
#[derive(Debug, Deserialize)]
struct InputRoot {
    #[serde(default)]
    segments: Vec<InputSegment>,
}

/// A single segment of an input transcript file.
#[derive(Debug, Deserialize)]
struct InputSegment {
    start: f64,
    end: f64,
    text: String,
    // other fields are ignored
}

/// Load a JSON transcript in the input format (`{"segments": [...]}`) and
/// return its segments as entries with id 0 and an empty speaker label.
pub fn load_json_transcript(path: &Path) -> Result<Vec<OutputEntry>> {
    let mut buf = String::new();
    File::open(path)
        .with_context(|| format!("Failed to open: {}", path.display()))?
        .read_to_string(&mut buf)
        .with_context(|| format!("Failed to read: {}", path.display()))?;
    let root: InputRoot = serde_json::from_str(&buf)
        .with_context(|| format!("Invalid JSON transcript parsed from {}", path.display()))?;
    Ok(root
        .segments
        .into_iter()
        .map(|seg| OutputEntry {
            id: 0,
            speaker: String::new(),
            start: seg.start,
            end: seg.end,
            text: seg.text,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, start: f64, end: f64, text: &str) -> OutputEntry {
        OutputEntry {
            id,
            speaker: String::new(),
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn test_sort_and_reindex_orders_by_start_then_end() {
        let mut entries = vec![
            entry(9, 5.0, 6.0, "c"),
            entry(9, 1.0, 3.0, "b2"),
            entry(9, 1.0, 2.0, "b1"),
            entry(9, 0.0, 9.0, "a"),
        ];
        sort_and_reindex(&mut entries);
        let order: Vec<&str> = entries.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(order, ["a", "b1", "b2", "c"]);
        let ids: Vec<u64> = entries.iter().map(|e| e.id).collect();
        assert_eq!(ids, [0, 1, 2, 3]);
    }

    #[test]
    fn test_sort_and_reindex_empty() {
        let mut entries: Vec<OutputEntry> = Vec::new();
        sort_and_reindex(&mut entries);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_write_output_files_creates_json_toml_srt() {
        let tmp = tempfile::tempdir().unwrap();
        let out = OutputRoot {
            items: vec![OutputEntry {
                id: 0,
                speaker: "Alice".to_string(),
                start: 0.0,
                end: 1.5,
                text: "Hello".to_string(),
            }],
        };
        write_output_files(tmp.path(), "2026-01-02_test", &out).unwrap();

        let json = std::fs::read_to_string(tmp.path().join("2026-01-02_test.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["items"][0]["speaker"], "Alice");
        assert_eq!(parsed["items"][0]["start"], 0.0);
        assert!(json.ends_with('\n'));

        let toml_str = std::fs::read_to_string(tmp.path().join("2026-01-02_test.toml")).unwrap();
        assert!(toml_str.contains("Hello"));
        assert!(toml_str.ends_with('\n'));

        let srt = std::fs::read_to_string(tmp.path().join("2026-01-02_test.srt")).unwrap();
        assert_eq!(srt, "1\n00:00:00,000 --> 00:00:01,500\nAlice: Hello\n\n");
    }

    #[test]
    fn test_load_json_transcript_roundtrip_and_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("in.json");
        std::fs::write(
            &path,
            r#"{"segments": [{"start": 1.0, "end": 2.0, "text": "hi", "extra": 1}]}"#,
        )
        .unwrap();
        let entries = load_json_transcript(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start, 1.0);
        assert_eq!(entries[0].end, 2.0);
        assert_eq!(entries[0].text, "hi");
        assert_eq!(entries[0].id, 0);
        assert!(entries[0].speaker.is_empty());

        // Missing segments key is fine (defaults to empty)
        let path2 = tmp.path().join("empty.json");
        std::fs::write(&path2, r#"{"other": true}"#).unwrap();
        assert!(load_json_transcript(&path2).unwrap().is_empty());

        // Invalid JSON errors with context
        let path3 = tmp.path().join("bad.json");
        std::fs::write(&path3, "not json").unwrap();
        let err = load_json_transcript(&path3).unwrap_err();
        assert!(format!("{err:#}").contains("Invalid JSON transcript"));

        // Missing file errors with context
        let err = load_json_transcript(&tmp.path().join("nope.json")).unwrap_err();
        assert!(format!("{err:#}").contains("Failed to open"));
    }
}
