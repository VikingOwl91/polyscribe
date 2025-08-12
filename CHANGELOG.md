# PolyScribe Refactor toward Rust 2024 — Incremental Patches

This changelog documents each incremental step applied to keep the build green while moving the codebase toward Rust 2024 idioms.

## 1) Formatting only (rustfmt)
- Ran `cargo fmt` across the repository.
- No semantic changes.
- Build status: OK (`cargo build` succeeded).

## 2) Lints — initial fixes (non-pedantic)
- Adjusted crate lint policy in `src/lib.rs`:
  - Replaced `#![warn(clippy::pedantic, clippy::nursery, clippy::cargo)]` with `#![warn(clippy::all)]` to align with the plan (skip pedantic/nursery for now).
  - Added comment/TODO to revisit stricter lints in a later pass.
- Fixed several clippy warnings that were causing `cargo clippy --all-targets` to error under tests:
  - `src/backend.rs`: conditionally import `libloading::Library` only for non-test builds and mark `names` parameter as used in test cfg to avoid unused warnings; keep `check_lib()` side‑effect free during tests.
  - `src/models.rs`: removed an unused `std::io::Write` import in test module.
  - `src/main.rs` (unit tests): imported `polyscribe::format_srt_time` explicitly and removed a duplicate `use super::*;` to fix unresolved name and unused import warnings under clippy test builds.
- Build/Clippy status:
  - `cargo build`: OK.
  - `cargo clippy --all-targets`: OK (only warnings remain; no errors).

## 3) Module hygiene
- Verified crate structure:
  - Library crate (`src/lib.rs`) exposes a coherent API and re‑exports `backend` and `models` via `pub mod`.
  - Binary (`src/main.rs`) consumes the library API through `polyscribe::...` paths.
- No structural changes required. Build status: OK.

## 4) Edition
- The project already targets `edition = "2024"` in Cargo.toml.
- Verified that the project compiles under Rust 2024. No changes needed.
- TODO: If stricter lints or new features from 2024 edition introduce issues in future steps, document blockers here.

## 5) Error handling
- The codebase already returns `anyhow::Result` in the binary and uses contextual errors widely.
- No `unwrap`/`expect` usages in production paths required attention in this pass.
- Build status: OK.

## Next planned steps (not yet applied in this changelog)
- Gradually fix remaining clippy warnings (e.g., `uninlined_format_args`, small style nits) in small, compile‑green patches.
- Optionally re‑enable `clippy::pedantic`, `clippy::nursery`, and `clippy::cargo` once warnings are significantly reduced, then address non‑breaking warnings.
