use thiserror::Error;

/// Error types for the polyscribe-core crate.
#[derive(Debug, Error)]
///
/// This enum represents various error conditions that can occur during
/// operations in this crate, including I/O errors, serialization/deserialization
/// errors, and environment variable access errors.
pub enum Error {
    #[error("I/O error: {0}")]
    /// Represents an I/O error that occurred during file or stream operations
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    /// Represents a JSON serialization or deserialization error
    Serde(#[from] serde_json::Error),

    #[error("toml error: {0}")]
    /// Represents a TOML deserialization error
    Toml(#[from] toml::de::Error),

    #[error("toml ser error: {0}")]
    /// Represents a TOML serialization error
    TomlSer(#[from] toml::ser::Error),

    #[error("env var error: {0}")]
    /// Represents an error that occurred during environment variable access
    EnvVar(#[from] std::env::VarError),

    #[error("http error: {0}")]
    /// Represents an HTTP client error from reqwest
    Http(#[from] reqwest::Error),

    #[error("other: {0}")]
    /// Represents a general error condition with a custom message
    Other(String),
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Error::Other(e.to_string())
    }
}
