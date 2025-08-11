use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Context;

use crate::render_srt;
use crate::OutputRoot;

/// Which formats to write.
pub struct OutputFormats {
    pub json: bool,
    pub toml: bool,
    pub srt: bool,
}

impl OutputFormats {
    pub fn all() -> Self {
        Self { json: true, toml: true, srt: true }
    }
}

/// Write outputs for the given base path (without extension).
/// This will create files named `base.json`, `base.toml`, and `base.srt`
/// according to the `formats` flags. JSON and TOML will always end with a trailing newline.
pub fn write_outputs(base: &Path, root: &OutputRoot, formats: &OutputFormats) -> anyhow::Result<()> {
    if formats.json {
        let json_path = base.with_extension("json");
        let mut json_file = File::create(&json_path).with_context(|| {
            format!("Failed to create output file: {}", json_path.display())
        })?;
        serde_json::to_writer_pretty(&mut json_file, root)?;
        // ensure trailing newline
        writeln!(&mut json_file)?;
    }

    if formats.toml {
        let toml_path = base.with_extension("toml");
        let toml_str = toml::to_string_pretty(root)?;
        let mut toml_file = File::create(&toml_path).with_context(|| {
            format!("Failed to create output file: {}", toml_path.display())
        })?;
        toml_file.write_all(toml_str.as_bytes())?;
        if !toml_str.ends_with('\n') {
            writeln!(&mut toml_file)?;
        }
    }

    if formats.srt {
        let srt_path = base.with_extension("srt");
        let srt_str = render_srt(&root.items);
        let mut srt_file = File::create(&srt_path).with_context(|| {
            format!("Failed to create output file: {}", srt_path.display())
        })?;
        srt_file.write_all(srt_str.as_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputEntry;

    #[test]
    fn write_outputs_creates_files_and_newlines() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("test_base");
        let items = vec![OutputEntry { id: 0, speaker: "Alice".to_string(), start: 0.0, end: 1.23, text: "Hello".to_string() }];
        let root = OutputRoot { items };

        write_outputs(&base, &root, &OutputFormats::all()).unwrap();

        let json_path = base.with_extension("json");
        let toml_path = base.with_extension("toml");
        let srt_path = base.with_extension("srt");

        assert!(json_path.exists(), "json file should exist");
        assert!(toml_path.exists(), "toml file should exist");
        assert!(srt_path.exists(), "srt file should exist");

        let json = std::fs::read_to_string(&json_path).unwrap();
        let toml = std::fs::read_to_string(&toml_path).unwrap();

        assert!(json.ends_with('\n'), "json should end with newline");
        assert!(toml.ends_with('\n'), "toml should end with newline");
    }
}
