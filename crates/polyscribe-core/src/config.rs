// SPDX-License-Identifier: MIT
// Simple ConfigService with XDG/system/workspace merge and atomic writes

use anyhow::{Context, Result};
use directories::BaseDirs;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Generic configuration represented as TOML table
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config(pub toml::value::Table);

impl Config {
    /// Get a mutable reference to a top-level table under the given key, creating
    /// an empty table if it does not exist yet.
    pub fn get_table_mut(&mut self, key: &str) -> &mut toml::value::Table {
        let needs_init = !matches!(self.0.get(key), Some(toml::Value::Table(_)));
        if needs_init {
            self.0.insert(key.to_string(), toml::Value::Table(Default::default()));
        }
        match self.0.get_mut(key) {
            Some(toml::Value::Table(t)) => t,
            _ => unreachable!(),
        }
    }
}

fn merge_tables(base: &mut toml::value::Table, overlay: &toml::value::Table) {
    for (k, v) in overlay.iter() {
        match (base.get_mut(k), v) {
            (Some(toml::Value::Table(bsub)), toml::Value::Table(osub)) => {
                merge_tables(bsub, osub);
            }
            _ => {
                base.insert(k.clone(), v.clone());
            }
        }
    }
}

fn read_toml(path: &Path) -> Result<toml::value::Table> {
    let s = fs::read_to_string(path).with_context(|| format!("Failed to read config: {}", path.display()))?;
    let v: toml::Value = toml::from_str(&s).with_context(|| format!("Invalid TOML in {}", path.display()))?;
    Ok(v.as_table().cloned().unwrap_or_default())
}

fn write_toml_atomic(path: &Path, tbl: &toml::value::Table) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Failed to create config dir: {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp).with_context(|| format!("Failed to create temp file: {}", tmp.display()))?;
    let s = toml::to_string_pretty(&toml::Value::Table(tbl.clone()))?;
    f.write_all(s.as_bytes())?;
    if !s.ends_with('\n') { f.write_all(b"\n")?; }
    drop(f);
    fs::rename(&tmp, path).with_context(|| format!("Failed to atomically replace config: {}", path.display()))?;
    Ok(())
}

fn system_config_path() -> PathBuf {
    if cfg!(unix) { PathBuf::from("/etc").join("polyscribe").join("config.toml") } else { default_user_config_path() }
}

fn default_user_config_path() -> PathBuf {
    if let Some(base) = BaseDirs::new() {
        return PathBuf::from(base.config_dir()).join("polyscribe").join("config.toml");
    }
    PathBuf::from(".polyscribe").join("config.toml")
}

fn workspace_config_path() -> PathBuf {
    PathBuf::from(".polyscribe").join("config.toml")
}

/// Service responsible for loading and saving PolyScribe configuration
#[derive(Debug, Default, Clone)]
pub struct ConfigService;

impl ConfigService {
    /// Load configuration, merging system < user < workspace < env overrides.
    pub fn load(&self) -> Result<Config> {
        let mut accum = toml::value::Table::default();
        let sys = system_config_path();
        if sys.exists() {
            merge_tables(&mut accum, &read_toml(&sys)?);
        }
        let user = default_user_config_path();
        if user.exists() {
            merge_tables(&mut accum, &read_toml(&user)?);
        }
        let ws = workspace_config_path();
        if ws.exists() {
            merge_tables(&mut accum, &read_toml(&ws)?);
        }
        // Env overrides: POLYSCRIBE__SECTION__KEY=value
        let mut env_over = toml::value::Table::default();
        for (k, v) in env::vars() {
            if let Some(rest) = k.strip_prefix("POLYSCRIBE__") {
                let parts: Vec<&str> = rest.split("__").collect();
                if parts.is_empty() { continue; }
                let val: toml::Value = toml::Value::String(v);
                // Build nested tables
                let mut current = &mut env_over;
                for (i, part) in parts.iter().enumerate() {
                    if i == parts.len() - 1 {
                        current.insert(part.to_lowercase(), val.clone());
                    } else {
                        current = current.entry(part.to_lowercase()).or_insert_with(|| toml::Value::Table(Default::default()))
                            .as_table_mut().expect("table");
                    }
                }
            }
        }
        merge_tables(&mut accum, &env_over);
        Ok(Config(accum))
    }

    /// Ensure user config exists with sensible defaults, return loaded config
    pub fn ensure_user_config(&self) -> Result<Config> {
        let path = default_user_config_path();
        if !path.exists() {
            let mut defaults = toml::value::Table::default();
            defaults.insert("ui".into(), toml::Value::Table({
                let mut t = toml::value::Table::default();
                t.insert("theme".into(), toml::Value::String("auto".into()));
                t
            }));
            write_toml_atomic(&path, &defaults)?;
        }
        self.load()
    }

    /// Save to user config atomically, merging over existing user file.
    pub fn save_user(&self, new_values: &toml::value::Table) -> Result<()> {
        let path = default_user_config_path();
        let mut base = if path.exists() { read_toml(&path)? } else { Default::default() };
        merge_tables(&mut base, new_values);
        write_toml_atomic(&path, &base)
    }

    /// Paths used for debugging/information
    pub fn paths(&self) -> (PathBuf, PathBuf, PathBuf) {
        (system_config_path(), default_user_config_path(), workspace_config_path())
    }
}
