use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

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

fn any_target_exists(base: &Path, formats: &OutputFormats) -> bool {
    (formats.json && base.with_extension("json").exists())
        || (formats.toml && base.with_extension("toml").exists())
        || (formats.srt && base.with_extension("srt").exists())
}

fn with_suffix(base: &Path, n: usize) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let name = base.file_name().and_then(|s| s.to_str()).unwrap_or("out");
    parent.join(format!("{}_{}", name, n))
}

fn resolve_base(base: &Path, formats: &OutputFormats, force: bool) -> PathBuf {
    if force {
        return base.to_path_buf();
    }
    if !any_target_exists(base, formats) {
        return base.to_path_buf();
    }
    let mut n = 1usize;
    loop {
        let candidate = with_suffix(base, n);
        if !any_target_exists(&candidate, formats) {
            return candidate;
        }
        n += 1;
    }
}

/// Write outputs for the given base path (without extension).
/// This will create files named `base.json`, `base.toml`, and `base.srt`
/// according to the `formats` flags. JSON and TOML will always end with a trailing newline.
pub fn write_outputs(base: &Path, root: &OutputRoot, formats: &OutputFormats, force: bool) -> anyhow::Result<()> {
    let base = resolve_base(base, formats, force);

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

        write_outputs(&base, &root, &OutputFormats::all(), false).unwrap();

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

    #[test]
    fn suffix_is_added_when_file_exists_unless_forced() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("run");

        // Precreate a toml file for base to simulate existing output
        let pre_path = base.with_extension("toml");
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(&pre_path, b"existing\n").unwrap();

        let items = vec![OutputEntry { id: 0, speaker: "A".to_string(), start: 0.0, end: 1.0, text: "Hi".to_string() }];
        let root = OutputRoot { items };
        let fmts = OutputFormats { json: false, toml: true, srt: false };

        // Without force, should write to run_1.toml
        write_outputs(&base, &root, &fmts, false).unwrap();
        assert!(base.with_file_name("run_1").with_extension("toml").exists());

        // If run_1.toml also exists, next should be run_2.toml
        std::fs::write(base.with_file_name("run_1").with_extension("toml"), b"x\n").unwrap();
        write_outputs(&base, &root, &fmts, false).unwrap();
        assert!(base.with_file_name("run_2").with_extension("toml").exists());

        // With force, should overwrite the base.toml
        write_outputs(&base, &root, &fmts, true).unwrap();
        let content = std::fs::read_to_string(pre_path).unwrap();
        assert!(content.ends_with('\n'));
    }
}
