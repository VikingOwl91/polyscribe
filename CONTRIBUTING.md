# Contributing to PolyScribe

Thanks for your interest in contributing! This guide explains the workflow and the checklist to follow before opening a Pull Request.

The repository is a Cargo workspace: `polyscribe-core` (engine), `polyscribe-cli` (the `polyscribe` binary), `polyscribe-host` / `polyscribe-protocol` (plugin system), and `plugins/polyscribe-plugin-tubescribe` (reference plugin).

Workflow (fork → branch → PR)
1) Fork the repository to your account.
2) Create a branch off the integration branch (`dev`); `main` only carries published releases:
   - `git checkout -b feature/short-description` (or `fix/…`, `chore/…`)
3) Make changes with focused commits and good messages (conventional-commit style, e.g. `[feat] …`, `[fix] …`, `[refactor] …`, `[docs] …`, `[test] …`, `[build] …`).
4) Run the checklist below.
5) Push and open a Pull request against `dev` on Gitea, the source of truth.

The branch model, the release procedure and the platform roles are described
in `docs/releases.md`. In short: `dev` is the integration branch, `main`
carries the latest published release, feature branches are short-lived, pull
requests need a review and green CI, and `main` is never force-pushed.

Developer checklist (before opening a PR)
- Format:
  - `cargo fmt --all -- --check` (clean)
- Build:
  - `cargo build --workspace --all-targets` (no warnings)
- Tests:
  - `cargo test --workspace` (all tests pass; the suite needs no network, models, or GPU — run it with `--workspace` so the plugin end-to-end tests are included)
- Lints:
  - `cargo clippy --workspace --all-targets -- -D warnings` (fix warnings)
- MSRV:
  - `cargo +1.88 check --workspace --all-targets` if you touched dependencies or used new std APIs
- Documentation:
  - Update README/docs for user-visible changes
  - Update `CHANGELOG.md` for notable changes
- Tests for changes:
  - Add or update tests for bug fixes and new features where reasonable

Local development tips
- Use `cargo run -p polyscribe-cli -- <args>` during development.
- Keep functions small and focused; prefer clear error messages with context.
- Prerequisites: ffmpeg on `PATH`; for GPU builds, the matching runtime (CUDA/ROCm) and `--features gpu-cuda` / `--features gpu-hip` on `polyscribe-cli`.
- Some tests simulate GPU detection using env vars (`POLYSCRIBE_TEST_FORCE_*`); do not rely on these flags in production code paths.
- The workspace lints (`[workspace.lints]`) deny unused imports; CI enforces clippy with warnings as errors.

Code of conduct
- Be respectful and constructive. Assume good intent.
