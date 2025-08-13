# PolyScribe - TODO

This file tracks the development backlog for PolyScribe.

## High Priority

- **Set up CI Pipeline:**
  - Create a GitHub Actions workflow to run `cargo test` and `cargo clippy -- -D warnings` on all pull requests. This will automate the checks currently listed in `CONTRIBUTING.md`.

- **Improve Test Coverage:**
  - Review existing tests for transcription and plugin functionality.
  - Add integration tests for common CLI commands and edge cases (e.g., invalid inputs, file errors).

## Medium Priority

- **Flesh out Plugin Developer Documentation:**
  - Create a new document in `docs/` that explains the plugin architecture in detail, how to create a new plugin, and the protocol specification (`polyscribe-protocol`).

- **Support Additional Output Formats:**
  - Implement a `--format` flag to support output formats beyond the default JSON and SRT.
  - Good candidates include plain text (`.txt`), VTT (`.vtt`), or other common caption formats.

## Low Priority

- **Dependency Audit:**
  - Review `Cargo.toml` files in all crates to ensure dependencies are up-to-date and remove any unused ones.
