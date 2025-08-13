// SPDX-License-Identifier: MIT

use thiserror::Error;

/// The common error type for the polyscribe core crate.
/// Add more domain-specific variants as needed.
#[derive(Debug, Error)]
pub enum Error {
    /// Wrapper for any boxed dynamic error. Useful as a temporary catch-all.
    #[error("anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),

    /// IO-related error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// UTF-8 conversion error.
    #[error("utf8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    /// Environment variable error.
    #[error("env var error: {0}")]
    Var(#[from] std::env::VarError),

    /// TOML de serialization error.
    #[error("toml de error: {0}")]
    TomlDe(#[from] toml::de::Error),

    /// Configuration parsing error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Placeholder for not-yet-implemented backends or features.
    #[error("unimplemented: {0}")]
    Unimplemented(&'static str),
}

/// Convenient result alias for the polyscribe core crate.
pub type Result<T> = std::result::Result<T, Error>;
