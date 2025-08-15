use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("toml error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("toml ser error: {0}")]
    TomlSer(#[from] toml::ser::Error),

    #[error("env var error: {0}")]
    EnvVar(#[from] std::env::VarError),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("other: {0}")]
    Other(String),
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        Error::Other(e.to_string())
    }
}
