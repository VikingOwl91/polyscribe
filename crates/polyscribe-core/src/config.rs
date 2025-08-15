// SPDX-License-Identifier: MIT

use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

pub struct ConfigService;

impl ConfigService {
    pub const ENV_NO_CACHE_MANIFEST: &'static str = "POLYSCRIBE_NO_CACHE_MANIFEST";
    pub const ENV_MANIFEST_TTL_SECONDS: &'static str = "POLYSCRIBE_MANIFEST_TTL_SECONDS";
    pub const ENV_MODELS_DIR: &'static str = "POLYSCRIBE_MODELS_DIR";
    pub const ENV_USER_AGENT: &'static str = "POLYSCRIBE_USER_AGENT";
    pub const ENV_HTTP_TIMEOUT_SECS: &'static str = "POLYSCRIBE_HTTP_TIMEOUT_SECS";
    pub const ENV_HF_REPO: &'static str = "POLYSCRIBE_HF_REPO";
    pub const ENV_CACHE_FILENAME: &'static str = "POLYSCRIBE_MANIFEST_CACHE_FILENAME";

    pub const DEFAULT_USER_AGENT: &'static str = "polyscribe/0.1";
    pub const DEFAULT_DOWNLOADER_UA: &'static str = "polyscribe-model-downloader/1";
    pub const DEFAULT_HF_REPO: &'static str = "ggerganov/whisper.cpp";
    pub const DEFAULT_CACHE_FILENAME: &'static str = "hf_manifest_whisper_cpp.json";
    pub const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 8;
    pub const DEFAULT_MANIFEST_CACHE_TTL_SECONDS: u64 = 24 * 60 * 60;

    pub fn project_dirs() -> Option<directories::ProjectDirs> {
        directories::ProjectDirs::from("dev", "polyscribe", "polyscribe")
    }

    pub fn default_models_dir() -> Option<PathBuf> {
        Self::project_dirs().map(|d| d.data_dir().join("models"))
    }

    pub fn default_plugins_dir() -> Option<PathBuf> {
        Self::project_dirs().map(|d| d.data_dir().join("plugins"))
    }

    pub fn manifest_cache_dir() -> Option<PathBuf> {
        Self::project_dirs().map(|d| d.cache_dir().join("manifest"))
    }

    pub fn bypass_manifest_cache() -> bool {
        env::var(Self::ENV_NO_CACHE_MANIFEST).is_ok()
    }

    pub fn manifest_cache_ttl_seconds() -> u64 {
        env::var(Self::ENV_MANIFEST_TTL_SECONDS)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(Self::DEFAULT_MANIFEST_CACHE_TTL_SECONDS)
    }

    pub fn manifest_cache_filename() -> String {
        env::var(Self::ENV_CACHE_FILENAME)
            .unwrap_or_else(|_| Self::DEFAULT_CACHE_FILENAME.to_string())
    }

    pub fn models_dir(cfg: Option<&Config>) -> Option<PathBuf> {
        if let Ok(env_dir) = env::var(Self::ENV_MODELS_DIR) {
            if !env_dir.is_empty() {
                return Some(PathBuf::from(env_dir));
            }
        }
        if let Some(c) = cfg {
            if let Some(dir) = c.models_dir.clone() {
                return Some(dir);
            }
        }
        Self::default_models_dir()
    }

    pub fn user_agent() -> String {
        env::var(Self::ENV_USER_AGENT).unwrap_or_else(|_| Self::DEFAULT_USER_AGENT.to_string())
    }

    pub fn downloader_user_agent() -> String {
        env::var(Self::ENV_USER_AGENT).unwrap_or_else(|_| Self::DEFAULT_DOWNLOADER_UA.to_string())
    }

    pub fn http_timeout_secs() -> u64 {
        env::var(Self::ENV_HTTP_TIMEOUT_SECS)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(Self::DEFAULT_HTTP_TIMEOUT_SECS)
    }

    pub fn hf_repo() -> String {
        env::var(Self::ENV_HF_REPO).unwrap_or_else(|_| Self::DEFAULT_HF_REPO.to_string())
    }

    pub fn hf_api_base_for(repo: &str) -> String {
        format!("https://huggingface.co/api/models/{}", repo)
    }

    pub fn manifest_cache_path() -> Option<PathBuf> {
        let dir = Self::manifest_cache_dir()?;
        Some(dir.join(Self::manifest_cache_filename()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub models_dir: Option<PathBuf>,
    pub plugins_dir: Option<PathBuf>,
}
