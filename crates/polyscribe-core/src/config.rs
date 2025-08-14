use crate::prelude::*;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

const ENV_PREFIX: &str = "POLYSCRIBE";

/// Configuration for the Polyscribe application
///
/// Contains paths to models and plugins directories that can be customized
/// through configuration files or environment variables.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Directory path where ML models are stored
    pub models_dir: Option<PathBuf>,
    /// Directory path where plugins are stored
    pub plugins_dir: Option<PathBuf>,
}

// Default is derived

/// Service for managing Polyscribe configuration
///
/// Provides functionality to load, save, and access configuration settings
/// from disk or environment variables.
pub struct ConfigService;

impl ConfigService {
    /// Loads configuration from disk or returns default values if not found
    ///
    /// This function attempts to read the configuration file from disk. If the file
    /// doesn't exist or can't be parsed, it falls back to default values.
    /// Environment variable overrides are then applied to the configuration.
    pub fn load_or_default() -> Result<Config> {
        let mut cfg = Self::read_disk().unwrap_or_default();
        Self::apply_env_overrides(&mut cfg)?;
        Ok(cfg)
    }

    /// Saves the configuration to disk
    ///
    /// This function serializes the configuration to TOML format and writes it
    /// to the standard configuration directory for the application.
    /// Returns an error if writing fails or if project directories cannot be determined.
    pub fn save(cfg: &Config) -> Result<()> {
        let Some(dirs) = Self::dirs() else {
            return Err(Error::Other("unable to get project dirs".into()));
        };
        let cfg_dir = dirs.config_dir();
        fs::create_dir_all(cfg_dir)?;
        let path = cfg_dir.join("config.toml");
        let s = toml::to_string_pretty(cfg)?;
        fs::write(path, s)?;
        Ok(())
    }

    fn read_disk() -> Option<Config> {
        let dirs = Self::dirs()?;
        let path = dirs.config_dir().join("config.toml");
        let s = fs::read_to_string(path).ok()?;
        toml::from_str(&s).ok()
    }

    fn apply_env_overrides(cfg: &mut Config) -> Result<()> {
        // POLYSCRIBE__SECTION__KEY format reserved for future nested config.
        if let Ok(v) = std::env::var(format!("{ENV_PREFIX}_MODELS_DIR")) {
            cfg.models_dir = Some(PathBuf::from(v));
        }
        if let Ok(v) = std::env::var(format!("{ENV_PREFIX}_PLUGINS_DIR")) {
            cfg.plugins_dir = Some(PathBuf::from(v));
        }
        Ok(())
    }

    /// Returns the standard project directories for the application
    ///
    /// This function creates a ProjectDirs instance with the appropriate
    /// organization and application names for Polyscribe.
    /// Returns None if the project directories cannot be determined.
    pub fn dirs() -> Option<ProjectDirs> {
        ProjectDirs::from("dev", "polyscribe", "polyscribe")
    }

    /// Returns the default directory path for storing ML models
    ///
    /// This function determines the standard data directory for the application
    /// and appends a 'models' subdirectory to it.
    /// Returns None if the project directories cannot be determined.
    pub fn default_models_dir() -> Option<PathBuf> {
        Self::dirs().map(|d| d.data_dir().join("models"))
    }

    /// Returns the default directory path for storing plugins
    ///
    /// This function determines the standard data directory for the application
    /// and appends a 'plugins' subdirectory to it.
    /// Returns None if the project directories cannot be determined.
    pub fn default_plugins_dir() -> Option<PathBuf> {
        Self::dirs().map(|d| d.data_dir().join("plugins"))
    }
}
