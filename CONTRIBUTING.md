# Contributing to PolyScribe

Thanks for your interest in contributing! This guide explains the workflow and the checklist to follow before opening a Pull Request.

Workflow (fork → branch → PR)
1) Fork the repository to your account.
2) Create a feature branch:
- `git checkout -b feat/short-description` (branch off `dev`, the integration branch; `main` is protected)
3) Make changes with focused commits and good messages (conventional-commit style, e.g. `[feat] …`, `[fix] …`, `[refactor] …`, `[docs] …`, `[test] …`, `[build] …`).
4) Run the checklist below.
5) Push and open a Pull Request against `dev`.

Developer checklist (before opening a PR)
- Format:
  - `cargo fmt --all -- --check` (clean)
- Build:
  - `cargo build --all-targets` (no warnings)
- Tests:
  - `cargo test` (all tests pass; integration tests use the fixtures in `tests/fixtures/` and need no network, models, or GPU)
- Lints:
  - `cargo clippy --all-targets -- -D warnings` (fix warnings)
- MSRV:
  - `cargo +1.88 check --all-targets` if you touched dependencies or used new std APIs
- Documentation:
  - Update README/docs for user-visible changes
- Tests for changes:
  - Add or update tests for bug fixes and new features where reasonable

Local development tips
- Use `cargo run -- <args>` during development.
- For faster feedback, keep examples in the `examples/` folder handy.
- Keep functions small and focused; prefer clear error messages with context.
- The test suite uses env-var overrides (`POLYSCRIBE_TEST_FORCE_*`) to simulate
  GPU detection; do not rely on them in production code paths.

Code of conduct
- Be respectful and constructive. Assume good intent.
