# CI checklist and job outline

Checklist to keep docs and code healthy in CI (all commands run at the workspace root):

- Format: `cargo fmt --all -- --check`
- Build: `cargo build --workspace --all-targets --locked`
- Tests: `cargo test --workspace --locked` (includes the plugin end-to-end tests; no network, models, or GPU needed)
- Lints: `cargo clippy --workspace --all-targets -- -D warnings`
- MSRV: `cargo +1.88 check --workspace --all-targets --locked`
- E2E: the `e2e-transcribe` job downloads a SHA-256-pinned `ggml-tiny.en.bin` (actions cache; the checksum is verified on every run, so unverified cache contents are never trusted), then runs `cargo test --locked -p polyscribe-cli --test e2e_transcribe` with `POLYSCRIBE_E2E_MODEL` set: a real CPU transcription of the committed `speech.wav` fixture through the built CLI, asserting valid JSON/TOML/SRT output and sane timestamps. Locally: `POLYSCRIBE_E2E_MODEL=/path/to/ggml-tiny.en.bin cargo test -p polyscribe-cli --test e2e_transcribe`.
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
- The e2e transcription test skips itself (with a note) when `POLYSCRIBE_E2E_MODEL` is unset, so `cargo test --workspace` still needs no model; the `e2e-transcribe` job provides one.
- The reference plugin (`plugins/polyscribe-plugin-tubescribe`) is a workspace member, so it is built and exercised by the test job.
- For GPU features, set up appropriate runners and add `--features polyscribe-cli/gpu-cuda` or `--features polyscribe-cli/gpu-hip` where applicable.
- For docs-only changes, jobs still build/test to ensure doctests and examples compile when enabled.
