use anyhow::{Context, Result};
use serde::Deserialize;
use std::process::Stdio;
use std::{env, fs, os::unix::fs::PermissionsExt, path::{Path, PathBuf}};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::{Child as TokioChild, Command},
};

#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Default)]
pub struct PluginManager;

impl PluginManager {
    pub fn list(&self) -> Result<Vec<PluginInfo>> {
        let mut plugins = Vec::new();

        // 1) Scan PATH entries for executables starting with "polyscribe-plugin-"
        if let Ok(path) = env::var("PATH") {
            for dir in env::split_paths(&path) {
                scan_dir_for_plugins(&dir, &mut plugins);
            }
        }

        // 2) Scan XDG data dir: $XDG_DATA_HOME/polyscribe/plugins or platform equiv
        if let Some(dirs) = directories::ProjectDirs::from("dev", "polyscribe", "polyscribe") {
            let plugin_dir = dirs.data_dir().join("plugins");
            scan_dir_for_plugins(&plugin_dir, &mut plugins);
        }

        // 3) De-duplicate by binary path
        plugins.sort_by(|a, b| a.path.cmp(&b.path));
        plugins.dedup_by(|a, b| a.path == b.path);
        Ok(plugins)
    }

    pub fn info(&self, name: &str) -> Result<serde_json::Value> {
        let bin = self.resolve(name)?;
        let out = std::process::Command::new(&bin)
            .arg("info")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("spawning plugin info")?
            .wait_with_output()
            .context("waiting for plugin info")?;

        let val: serde_json::Value =
            serde_json::from_slice(&out.stdout).context("parsing plugin info JSON")?;
        Ok(val)
    }

    pub fn spawn(&self, name: &str, command: &str) -> Result<TokioChild> {
        let bin = self.resolve(name)?;
        let mut cmd = Command::new(&bin);
        cmd.arg("run")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let child = cmd.spawn().context("spawning plugin run")?;
        Ok(child)
    }

    pub async fn forward_stdio(&self, child: &mut TokioChild) -> Result<std::process::ExitStatus> {
        if let Some(stdout) = child.stdout.take() {
            let mut reader = BufReader::new(stdout).lines();
            while let Some(line) = reader.next_line().await? {
                println!("{line}");
            }
        }
        Ok(child.wait().await?)
    }

    fn resolve(&self, name: &str) -> Result<String> {
        let bin = format!("polyscribe-plugin-{name}");
        let path =
            which::which(&bin).with_context(|| format!("plugin not found in PATH: {bin}"))?;
        Ok(path.to_string_lossy().to_string())
    }
}

fn is_executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        if let Ok(meta) = fs::metadata(path) {
            let mode = meta.permissions().mode();
            // if any execute bit is set
            return mode & 0o111 != 0;
        }
    }
    // Fallback for non-unix (treat files as candidates)
    true
}

fn scan_dir_for_plugins(dir: &Path, out: &mut Vec<PluginInfo>) {
    if let Ok(read_dir) = fs::read_dir(dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if let Some(fname) = path.file_name().and_then(|s| s.to_str())
                && fname.starts_with("polyscribe-plugin-")
                && is_executable(&path)
            {
                let name = fname.trim_start_matches("polyscribe-plugin-").to_string();
                out.push(PluginInfo {
                    name,
                    path: path.to_string_lossy().to_string(),
                });
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Capability {
    command: String,
    summary: String,
}
