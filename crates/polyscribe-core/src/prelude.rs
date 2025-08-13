// rust
//! Commonly used exports for convenient glob-imports in binaries and tests.
//! Usage: `use polyscribe_core::prelude::*;`

pub use crate::backend::*;
pub use crate::config::*;
pub use crate::error::{Error, Result};
pub use crate::models::*;

// If you frequently use UI helpers across binaries/tests, export them too.
// Keep this lean to avoid pulling UI everywhere unintentionally.
#[allow(unused_imports)]
pub use crate::ui::*;
