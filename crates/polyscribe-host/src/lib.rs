// SPDX-License-Identifier: MIT

use anyhow::{anyhow, Context, Result};
use cliclack as ui; // reuse for minimal logging
use directories::BaseDirs;
use serde_json::Value;
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use polyscribe_protocol as psp;

#[derive(Debug, Clone)]
pub struct Plugin {
    pub name: String,
    pub path: PathBuf,
}

/// Discover plugins on PATH and in the user's data dir (XDG) under polyscribe/plugins.
pub fn discover() -> Result<Vec<Plugin>> {
    let mut found: BTreeMap<String, PathBuf> = BTreeMap::new();

    // Scan PATH directories
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            if dir.as_os_str().is_empty() { continue; }
            if let Ok(rd) = fs::read_dir(&dir) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if !is_executable(&p) { continue; }
                    if let Some(fname) = p.file_name().and_then(OsStr::to_str) {
                        if let Some(name) = fname.strip_prefix("polyscribe-plugin-") {
                            found.entry(name.to_string()).or_insert(p);
                        }
                    }
                }
            }
        }
    }

    // Scan user data dir
    if let Some(base) = BaseDirs::new() {
        let user_plugins = PathBuf::from(base.data_dir()).join("polyscribe").join("plugins");
        if let Ok(rd) = fs::read_dir(&user_plugins) {
            for ent in rd.flatten() {
                let p = ent.path();
                if !is_executable(&p) { continue; }
                if let Some(fname) = p.file_name().and_then(OsStr::to_str) {
                    let name = fname.strip_prefix("polyscribe-plugin-")
                        .map(|s| s.to_string())
                        .or_else(|| Some(fname.to_string()))
                        .unwrap();
                    found.entry(name).or_insert(p);
                }
            }
        }
    }

    Ok(found
        .into_iter()
        .map(|(name, path)| Plugin { name, path })
        .collect())
}

fn is_executable(p: &Path) -> bool {
    if !p.is_file() { return false; }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(md) = fs::metadata(p) {
            let mode = md.permissions().mode();
            return (mode & 0o111) != 0;
        }
        false
    }
    #[cfg(not(unix))]
    {
        // On Windows, consider .exe, .bat, .cmd
        matches!(p.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()), Some(ext) if matches!(ext.as_str(), "exe"|"bat"|"cmd"))
    }
}

/// Query plugin capabilities by invoking `--capabilities`.
pub fn capabilities(plugin_path: &Path) -> Result<psp::Capabilities> {
    let out = Command::new(plugin_path)
        .arg("--capabilities")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .with_context(|| format!("Failed to execute plugin: {}", plugin_path.display()))?;
    if !out.status.success() {
        return Err(anyhow!("Plugin --capabilities failed: {}", plugin_path.display()));
    }
    let s = String::from_utf8(out.stdout).context("capabilities stdout not utf-8")?;
    let caps: psp::Capabilities = serde_json::from_str(s.trim()).context("invalid capabilities JSON")?;
    Ok(caps)
}

/// Run a single method via `--serve`, writing one JSON-RPC request and streaming until result.
pub fn run_method<F>(plugin_path: &Path, method: &str, params: Value, mut on_progress: F) -> Result<Value>
where
    F: FnMut(psp::Progress),
{
    let mut child = Command::new(plugin_path)
        .arg("--serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("Failed to spawn plugin: {}", plugin_path.display()))?;

    let mut stdin = child.stdin.take().ok_or_else(|| anyhow!("failed to open plugin stdin"))?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("failed to open plugin stdout"))?;

    // Send request line
    let req = psp::JsonRpcRequest { jsonrpc: "2.0".into(), id: "1".into(), method: method.to_string(), params: Some(params) };
    let line = serde_json::to_string(&req)? + "\n";
    stdin.write_all(line.as_bytes())?;
    stdin.flush()?;

    // Read response lines
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        // Try parse StreamItem; if that fails, try parse JsonRpcResponse directly
        if let Ok(item) = serde_json::from_str::<psp::StreamItem>(&line) {
            match item {
                psp::StreamItem::Progress(p) => {
                    on_progress(p);
                }
                psp::StreamItem::Result(resp) => {
                    match resp.outcome {
                        psp::JsonRpcOutcome::Ok { result } => return Ok(result),
                        psp::JsonRpcOutcome::Err { error } => return Err(anyhow!("{} ({})", error.message, error.code)),
                    }
                }
            }
        } else if let Ok(resp) = serde_json::from_str::<psp::JsonRpcResponse>(&line) {
            match resp.outcome {
                psp::JsonRpcOutcome::Ok { result } => return Ok(result),
                psp::JsonRpcOutcome::Err { error } => return Err(anyhow!("{} ({})", error.message, error.code)),
            }
        } else {
            let _ = ui::log::warning(format!("Unrecognized plugin output: {}", line));
        }
    }

    // If we exited loop without returning, wait for child
    let status = child.wait()?;
    if status.success() {
        Err(anyhow!("Plugin terminated without sending a result"))
    } else {
        Err(anyhow!("Plugin exited with status: {:?}", status))
    }
}

/// Helper: find a plugin by name using discovery
pub fn find_plugin_by_name(name: &str) -> Result<Plugin> {
    let plugins = discover()?;
    plugins
        .into_iter()
        .find(|p| p.name == name)
        .ok_or_else(|| anyhow!("Plugin '{}' not found", name))
}
