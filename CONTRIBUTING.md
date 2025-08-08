# Contributing to PolyScribe

Thanks for your interest in contributing! This guide explains the workflow and the checklist to follow before opening a Pull Request.

Workflow (fork → branch → PR)
1) Fork the repository to your account.
2) Create a feature branch:
- git checkout -b feat/short-description
3) Make changes with focused commits and good messages.
4) Run the checklist below.
5) Push and open a Pull Request against the main repository.

Developer checklist (before opening a PR)
- Build:
  - cargo build (preferably without warnings)
- Tests:
  - cargo test (all tests pass)
- Lints:
  - cargo clippy --all-targets -- -D warnings (fix warnings)
- Documentation:
  - Update README/docs for user-visible changes
  - Update CHANGELOG.md if applicable
- Tests for changes:
  - Add or update tests for bug fixes and new features where reasonable

Local development tips
- Use `cargo run -- <args>` during development.
- For faster feedback, keep examples in the examples/ folder handy.
- Keep functions small and focused; prefer clear error messages with context.

Code of conduct
- Be respectful and constructive. Assume good intent.
