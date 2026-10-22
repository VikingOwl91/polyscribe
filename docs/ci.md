# CI checklist and job outline

## Platform roles and merge gate

- Gitea is the canonical Git repository for branches, pull requests and
  releases. GitHub is the synchronized mirror.
- GitHub Actions is authoritative for merge decisions. All required checks
  must pass on the exact commit at the PR head; a result from an earlier head
  does not count.
- Gitea Actions checks are informational while no functional Gitea runner is
  available. A missing Gitea runner does not justify bypassing a failing
  GitHub check.
- Changes to CI authority or required checks need review and must be reflected
  in this document and `docs/releases.md`.

Checklist to keep docs and code healthy in CI (all commands run at the workspace root):

- Format: `cargo fmt --all -- --check`
- Build: `cargo build --workspace --all-targets --locked`
- Tests: `cargo test --workspace --locked` (includes the plugin end-to-end tests; no network, models, or GPU needed)
- Lints: `cargo clippy --workspace --all-targets -- -D warnings`
- MSRV: `cargo +1.88 check --workspace --all-targets --locked`
- E2E: the `e2e-transcribe` job downloads a SHA-256-pinned `ggml-tiny.en.bin` (the checksum is verified on every run), then runs `e2e_transcribe`, `integration_stream`, `integration_boundary`, `integration_live`, `integration_record`, and two tiny-compatible `integration_recovery` tests. These suites use real CPU inference; the record suite replaces `pw-record` with a fake program. The skipped-region file-recovery fixture needs a larger model and is not run in CI. Locally, set `POLYSCRIBE_E2E_MODEL=/path/to/ggml-tiny.en.bin` and run the corresponding `cargo test -p polyscribe-cli --test <suite>` command. Run the full `integration_recovery` suite with a large model to verify that fixture.
- Optional: check README and docs snippets (basic smoke run of the examples inline)

Example GitHub Actions job (outline; the actual workflow is `.github/workflows/ci.yml`):

- name: Rust
  runs-on: ubuntu-latest
  steps:
  - uses: actions/checkout@v4
  - uses: dtolnay/rust-toolchain@stable
  - name: Format
    run: cargo fmt --all -- --check
  - name: Build
    run: cargo build --workspace --all-targets --locked
  - name: Test
    run: cargo test --workspace --locked
  - name: Clippy
    run: cargo clippy --workspace --all-targets -- -D warnings

Notes

- The test suite builds whisper.cpp (cmake + a C/C++ toolchain + libclang, all preinstalled on GitHub runners) and uses the fixtures in `crates/polyscribe-cli/tests/fixtures/`.
- The `test` and `e2e-transcribe` jobs install ffmpeg when it is missing (current ubuntu-latest runners no longer ship it); ffmpeg-dependent tests skip themselves with a note when it is absent.
- Model-backed tests skip themselves with a note when `POLYSCRIBE_E2E_MODEL` is unset, so `cargo test --workspace` still needs no model; the `e2e-transcribe` job provides one.
- The reference plugin (`plugins/polyscribe-plugin-tubescribe`) is a workspace member, so it is built and exercised by the test job.
- For GPU features, set up appropriate runners and add `--features polyscribe-cli/gpu-cuda` or `--features polyscribe-cli/gpu-hip` where applicable.
- For docs-only changes, jobs still build/test to ensure doctests and examples compile when enabled.
