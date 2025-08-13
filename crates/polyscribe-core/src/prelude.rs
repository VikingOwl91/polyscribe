// rust
//! Commonly used exports for convenient glob-imports in binaries and tests.
//! Usage: `use polyscribe_core::prelude::*;`

pub use crate::backend::*;
pub use crate::config::*;
pub use crate::error::Error;
pub use crate::models::*;

// If you frequently use UI helpers across binaries/tests, export them too.
// Keep this lean to avoid pulling UI everywhere unintentionally.
#[allow(unused_imports)]
pub use crate::ui::*;

/// A convenient alias for `std::result::Result` with the error type defaulting to [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
