// SPDX-License-Identifier: MIT

//! Transcript output handling: sorting, JSON/TOML/SRT serialization, and
//! writing transcript files (or JSON to stdout).

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::OutputEntry;

/// Root object of a PolyScribe transcript file (JSON and TOML outputs).
#[derive(Debug, Serialize)]
pub struct OutputRoot {
    /// Transcript segments in output order.
    pub items: Vec<OutputEntry>,
}

/// Determine whether a path has a JSON transcript extension (case-insensitive).
pub fn is_json_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()), Some(ext) if ext == "json")
}

/// Determine whether a path has a supported audio/video extension (case-insensitive).
pub fn is_audio_file(path: &Path) -> bool {
    if let Some(ext) = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
    {
        let exts = [
            "mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mkv", "mov", "avi",
            "m4b", "3gp", "opus", "aiff", "alac",
        ];
        return exts.contains(&ext.as_str());
    }
    false
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

    let srt_str = crate::render_srt(&out.items);
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

    #[test]
    fn test_is_json_file_and_is_audio_file() {
        assert!(is_json_file(Path::new("foo.json")));
        assert!(is_json_file(Path::new("foo.JSON")));
        assert!(!is_json_file(Path::new("foo.txt")));
        assert!(!is_json_file(Path::new("foo")));

        assert!(is_audio_file(Path::new("a.mp3")));
        assert!(is_audio_file(Path::new("b.WAV")));
        assert!(is_audio_file(Path::new("c.m4a")));
        assert!(is_audio_file(Path::new("video.mp4")));
        assert!(is_audio_file(Path::new("clip.WEBM")));
        assert!(!is_audio_file(Path::new("d.txt")));
    }
}
