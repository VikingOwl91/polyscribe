# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## Unreleased

### Changed
- Docs: Replace `--download-models`/`--update-models` flags with `models download`/`models update` subcommands in `README.md`, `docs/usage.md`, and `docs/development.md`.
- Host: Plugin discovery now scans `$XDG_DATA_HOME/polyscribe/plugins` (platform equivalent via `directories`) in addition to `PATH`.
- CI: Add GitHub Actions workflow to run fmt, clippy (warnings as errors), and tests for pushes and PRs.


